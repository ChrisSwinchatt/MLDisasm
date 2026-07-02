#!/usr/bin/env python3

# Copyright (C) 2018,2021 Chris Swinchatt
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
"""

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from mldisasm.constants import ASSEMBLY_VOCAB_SIZE
from mldisasm.model import Disassembler
from mldisasm.util import log, prof


def grids(params):
    """Generate parameter sets from a parameter grid.

    :param params: Dictionary whose values are lists.
    :yields: One parameter set for each combination of grid values.
    """
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
    """Generate k-fold cross-validation indices.

    :param n: Number of samples.
    :param k: Number of folds (must be > 1).
    :param shuffle: Whether to shuffle fold order.
    :yields: Tuple of (train_indices, test_indices).
    """
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


def cv_split(X, y, train, test):
    """Split tensors for cross-validation."""
    return (
        tf.gather(X[0], train), tf.gather(X[1], train),
        tf.gather(y, train),
        tf.gather(X[0], test), tf.gather(X[1], test),
        tf.gather(y, test),
    )


def make_training_target(y, x_seq_len):
    """Create next-token prediction target.

    Shifts token indices right by one and pads the loss mask to include
    the byte portion (masked out with -100).

    :param y: Token indices, shape (batch, y_seq_len).
    :param x_seq_len: Length of byte sequence.
    :returns: (targets, sample_weight) where targets has shape
              (batch, x_seq_len + y_seq_len) and sample_weight masks
              the byte positions.
    """
    batch_size = tf.shape(y)[0]
    y_seq_len = tf.shape(y)[1]
    total_len = x_seq_len + y_seq_len

    # Shift tokens right for next-token prediction.
    shifted = tf.roll(y, shift=1, axis=1)
    # First non-byte position is START token target, use a pad for start.
    start_col = tf.fill([batch_size, 1], tf.cast(0, y.dtype))
    shifted = tf.concat([start_col, shifted[:, 1:]], axis=1)

    # Pad byte positions with zeros, then set up sample weight.
    byte_pad = tf.zeros([batch_size, x_seq_len], dtype=y.dtype)
    targets = tf.concat([byte_pad, shifted], axis=1)

    # Sample weight: 0 for byte positions, 1 for token positions.
    byte_mask = tf.zeros([batch_size, x_seq_len], dtype=tf.float32)
    token_mask = tf.ones([batch_size, y_seq_len], dtype=tf.float32)
    sample_weight = tf.concat([byte_mask, token_mask], axis=1)

    return targets, sample_weight


def train_model(X, y, params, train=None, test=None, retrain=None):
    """Train a model.

    :param X: Tuple of (byte_indices, token_indices).
    :param y: Token indices for target shifting.
    :param params: Model parameters.
    :param train: Training indices for CV, or None.
    :param test: Test indices for CV, or None.
    :param retrain: Existing model to continue training, or None.
    :returns: (model, accuracy, loss).
    """
    validation_data = None
    if train is not None and test is not None:
        X0_train, X1_train, y_train, X0_test, X1_test, y_test = cv_split(X, y, train, test)
        X = (X0_train, X1_train)
        y = y_train
        X_test = (X0_test, X1_test)
        y_test_targets, _ = make_training_target(y_test, params['x_seq_len'])
        validation_data = (X_test, y_test_targets)
    elif train is None != test is None:
        raise ValueError('Either both or neither of train and test should be None')

    model = retrain
    if model is None:
        model = Disassembler(vocab_size=ASSEMBLY_VOCAB_SIZE, **params)

    log.info(f'Training model with parameters {params}')

    targets, sample_weight = make_training_target(y, params['x_seq_len'])

    history = model.fit(
        X, targets,
        sample_weight=sample_weight,
        epochs=params['epochs'],
        steps_per_epoch=1,
        validation_data=validation_data,
    )

    acc_key = 'val_accuracy' if validation_data is not None else 'accuracy'
    loss_key = 'val_loss' if validation_data is not None else 'loss'
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
