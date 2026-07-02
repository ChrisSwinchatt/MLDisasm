#!/usr/bin/env python3

# Copyright (C) 2018,2021,2026 Chris Swinchatt
#
# MLDisasm is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# MLDisasm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with MLDisasm.  If not, see <http://www.gnu.org/licenses/>.

"""
MLDisasm training utilities.

LLM-style autoregressive next-token prediction training.
"""

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from mldisasm.constants import ASSEMBLY_VOCAB_SIZE, PAD_IDX
from mldisasm.model import Disassembler
from mldisasm.util import log, prof


def grids(params):
    """Generate parameter sets from a parameter grid."""
    keys, values = zip(*sorted(params.items()))
    sizes = [len(v) for v in values]
    for i in range(int(np.prod(sizes))):
        grid = {}
        remaining = i
        for k, vs, size in zip(keys, values, sizes):
            remaining, off = divmod(remaining, size)
            grid[k] = vs[off]
        yield grid


def parameter_grid(params):
    """Return a list of parameter sets from a parameter grid."""
    return list(grids(params))


def kfolds(n, k=3, shuffle=True):
    """Generate k-fold cross-validation indices."""
    m = n // k
    for i in range(k):
        start = i * m
        end = (i + 1) * m
        test = np.array(list(range(start, end)), dtype=np.int32)
        train = np.array(list(range(0, start)) + list(range(end, n)), dtype=np.int32)
        if shuffle:
            np.random.shuffle(train)
            np.random.shuffle(test)
        yield train, test


def fuse_sequences(byte_indices, token_indices):
    """Fuse byte values and token indices into a single sequence.

    :param byte_indices: (batch, x_seq_len) byte values 0-255.
    :param token_indices: (batch, y_seq_len) assembly token indices.
    :returns: (batch, x_seq_len + y_seq_len) fused sequence.
    """
    return tf.concat([byte_indices, token_indices], axis=1)


def make_llm_targets(fused_seq, x_seq_len):
    """Create next-token prediction targets with byte position masking.

    Shifts the full sequence right by 1 and masks byte positions with -100
    so the loss ignores them.

    :param fused_seq: (batch, total_len) fused [bytes|tokens].
    :param x_seq_len: Number of byte positions.
    :returns: (targets, sample_weight) each (batch, total_len).
    """
    batch_size = tf.shape(fused_seq)[0]
    total_len = tf.shape(fused_seq)[1]
    y_len = total_len - x_seq_len

    # Shift right: targets[i] = fused_seq[i+1], last target is PAD.
    targets = tf.concat([
        fused_seq[:, 1:],
        tf.fill([batch_size, 1], tf.cast(PAD_IDX, fused_seq.dtype))
    ], axis=1)

    # Mask: 0 for byte positions, 1 for token positions.
    byte_mask = tf.zeros([batch_size, x_seq_len], dtype=tf.float32)
    token_mask = tf.ones([batch_size, y_len], dtype=tf.float32)
    sample_weight = tf.concat([byte_mask, token_mask], axis=1)

    return targets, sample_weight


def make_dataset(X, y, x_seq_len, batch_size, shuffle=False):
    """Create a tf.data.Dataset for LLM-style training.

    :param X: Tuple of (byte_indices, token_indices).
    :param y: Token indices (used for fused sequence construction).
    :param x_seq_len: Number of byte positions.
    :param batch_size: Batch size.
    :param shuffle: Whether to shuffle.
    :returns: A tf.data.Dataset yielding (fused_seq, targets, sample_weight).
    """
    byte_indices, token_indices = X
    n = int(byte_indices.shape[0])

    # Build fused sequences.
    fused = fuse_sequences(byte_indices, token_indices)
    targets, sample_weight = make_llm_targets(fused, x_seq_len)

    ds = tf.data.Dataset.from_tensor_slices((fused, targets, sample_weight))

    if shuffle:
        ds = ds.shuffle(n)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train_model(X, y, params, train=None, test=None, retrain=None):
    """Train a model with LLM-style next-token prediction.

    :param X: Tuple of (byte_indices, token_indices).
    :param y: Token indices.
    :param params: Model parameters.
    :param train: Training indices for CV, or None.
    :param test: Test indices for CV, or None.
    :param retrain: Existing model to continue training, or None.
    :returns: (model, accuracy, loss).
    """
    validation_data = None
    x_seq_len = params['x_seq_len']

    # Build training dataset.
    byte_indices, token_indices = X
    if train is not None and test is not None:
        # Cross-validation split.
        train_byte = tf.gather(byte_indices, train)
        train_token = tf.gather(token_indices, train)
        test_byte = tf.gather(byte_indices, test)
        test_token = tf.gather(token_indices, test)

        train_ds = make_dataset(
            (train_byte, train_token), tf.gather(y, train),
            x_seq_len, params['batch_size'], shuffle=True
        )
        val_ds = make_dataset(
            (test_byte, test_token), tf.gather(y, test),
            x_seq_len, params['batch_size'], shuffle=False
        )
    elif train is None and test is None:
        train_ds = make_dataset(
            (byte_indices, token_indices), y,
            x_seq_len, params['batch_size'], shuffle=True
        )
        val_ds = None
    else:
        raise ValueError('Either both or neither of train and test should be None')

    model = retrain
    if model is None:
        model = Disassembler(vocab_size=ASSEMBLY_VOCAB_SIZE, **params)

    log.info(f'Training model with parameters {params}')

    history = model.fit(
        train_ds,
        epochs=params['epochs'],
        validation_data=val_ds,
    )

    acc_key = 'val_accuracy' if val_ds is not None else 'accuracy'
    loss_key = 'val_loss' if val_ds is not None else 'loss'
    acc = history.history[acc_key][-1]
    loss = history.history[loss_key][-1]
    return model, acc, loss


def _train_fold(X, y, params, train, test, fold_num):
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)
    acc = -np.inf
    loss = np.inf
    with prof(
        'Fold {}/{}: acc={}%, loss={}', fold_num, params['kfolds'],
        lambda: acc * 100, lambda: loss,
        log_level='info',
        start_msg=f'Fold {fold_num}/{params["kfolds"]}'
    ):
        _, acc, loss = train_model(X, y, params, train, test)
        return acc, loss


def _running_average(value, sample, count):
    return value * (count - 1) / count + sample / count


def kfolds_train(X, y, params):
    """Perform k-folds cross-validation training.

    :param X: Tuple of (byte_indices, token_indices).
    :param y: Token indices.
    :param params: Model parameters.
    :returns: (average_accuracy, average_loss).
    """
    avg_acc = 0.0
    avg_loss = 0.0
    fold_num = 1
    n_samples = int(X[0].shape[0])
    for train, test in kfolds(n_samples, params['kfolds'], params['shuffle']):
        acc, loss = _train_fold(X, y, params, train, test, fold_num)
        avg_acc = _running_average(avg_acc, acc, fold_num)
        avg_loss = _running_average(avg_loss, loss, fold_num)
        fold_num += 1
    return avg_acc, avg_loss
