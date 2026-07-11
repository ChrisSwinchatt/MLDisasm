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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

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
    return torch.cat([byte_indices, token_indices], dim=1)


def make_llm_targets(fused_seq, x_seq_len):
    """Create next-token prediction targets with byte position masking.

    Shifts the full sequence right by 1 and sets byte position targets to -100
    (PyTorch CrossEntropyLoss ignore_index) so the loss ignores them.

    :param fused_seq: (batch, total_len) fused [bytes|tokens].
    :param x_seq_len: Number of byte positions.
    :returns: targets (batch, total_len), byte positions set to -100.
    """
    batch_size = fused_seq.shape[0]
    total_len = fused_seq.shape[1]

    # Shift right: targets[i] = fused_seq[i+1], last target is PAD.
    targets = torch.cat([
        fused_seq[:, 1:],
        torch.full((batch_size, 1), PAD_IDX, dtype=fused_seq.dtype)
    ], dim=1)

    # Mask byte positions with ignore_index.
    targets = targets.clone()
    targets[:, :x_seq_len] = -100

    return targets


class FusedSequenceDataset(Dataset):
    """PyTorch Dataset yielding (fused_seq, targets) pairs for LLM training."""

    def __init__(self, byte_indices, token_indices, x_seq_len):
        self.fused = fuse_sequences(byte_indices, token_indices)
        self.targets = make_llm_targets(self.fused, x_seq_len)

    def __len__(self):
        return len(self.fused)

    def __getitem__(self, idx):
        return self.fused[idx], self.targets[idx]


def _compute_accuracy(logits, targets):
    """Compute accuracy over token positions only (non-masked targets)."""
    mask = targets != -100
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    return correct, total


def train_model(X, y, params, train=None, test=None, retrain=None, device=None):
    """Train a model with LLM-style next-token prediction.

    :param X: Tuple of (byte_indices, token_indices).
    :param y: Token indices.
    :param params: Model parameters.
    :param train: Training indices for CV, or None.
    :param test: Test indices for CV, or None.
    :param retrain: Existing model to continue training, or None.
    :param device: torch device (auto-detected if None).
    :returns: (model, accuracy, loss).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    byte_indices, token_indices = X

    # Build dataset.
    full_dataset = FusedSequenceDataset(byte_indices, token_indices, params['x_seq_len'])

    if train is not None and test is not None:
        train_ds = Subset(full_dataset, train)
        val_ds = Subset(full_dataset, test)
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)
    elif train is None and test is None:
        train_loader = DataLoader(full_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = None
    else:
        raise ValueError('Either both or neither of train and test should be None')

    model = retrain
    if model is None:
        model = Disassembler(vocab_size=ASSEMBLY_VOCAB_SIZE, **params)
    model.to(device)

    # Optimizer: match Keras Adam defaults.
    # Keras Adam: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7.
    # Note: Keras places epsilon inside sqrt; PyTorch places it outside.
    # This is a known non-equivalence (see residual differences doc).
    opt_params = params.get('opt_params', {})
    lr = opt_params.get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7
    )

    # Loss: sparse_categorical_crossentropy → CrossEntropyLoss with ignore_index.
    # Byte position targets are set to -100 so they don't contribute to loss,
    # equivalent to TF sample_weight=0 on byte positions.
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    log.info(f'Training model with parameters {params}')

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(params['epochs']):
        # Training.
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_token_total = 0
        for fused_seq, targets in train_loader:
            fused_seq = fused_seq.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(fused_seq)
            # CrossEntropyLoss expects (N, C, ...) i.e. (batch, vocab, seq).
            loss = criterion(logits.permute(0, 2, 1), targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            corr, tot = _compute_accuracy(logits, targets)
            epoch_correct += corr
            epoch_token_total += tot

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_correct / max(epoch_token_total, 1)
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)

        # Validation.
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_token_total = 0
            with torch.no_grad():
                for fused_seq, targets in val_loader:
                    fused_seq = fused_seq.to(device)
                    targets = targets.to(device)
                    logits = model(fused_seq)
                    loss = criterion(logits.permute(0, 2, 1), targets)
                    val_loss += loss.item()
                    corr, tot = _compute_accuracy(logits, targets)
                    val_correct += corr
                    val_token_total += tot
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_accuracy'].append(val_correct / max(val_token_total, 1))

    acc_key = 'val_accuracy' if val_loader is not None else 'accuracy'
    loss_key = 'val_loss' if val_loader is not None else 'loss'
    acc = history[acc_key][-1]
    loss = history[loss_key][-1]
    return model, acc, loss


def _train_fold(X, y, params, train, test, fold_num, device=None):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    acc = -float('inf')
    loss = float('inf')
    with prof(
        'Fold {}/{}: acc={}%, loss={}', fold_num, params['kfolds'],
        lambda: acc * 100, lambda: loss,
        log_level='info',
        start_msg=f'Fold {fold_num}/{params["kfolds"]}'
    ):
        _, acc, loss = train_model(X, y, params, train, test, device=device)
        return acc, loss


def _running_average(value, sample, count):
    return value * (count - 1) / count + sample / count


def kfolds_train(X, y, params, device=None):
    """Perform k-folds cross-validation training.

    :param X: Tuple of (byte_indices, token_indices).
    :param y: Token indices.
    :param params: Model parameters.
    :param device: torch device.
    :returns: (average_accuracy, average_loss).
    """
    avg_acc = 0.0
    avg_loss = 0.0
    fold_num = 1
    n_samples = int(X[0].shape[0])
    for train, test in kfolds(n_samples, params['kfolds'], params['shuffle']):
        acc, loss = _train_fold(X, y, params, train, test, fold_num, device=device)
        avg_acc = _running_average(avg_acc, acc, fold_num)
        avg_loss = _running_average(avg_loss, loss, fold_num)
        fold_num += 1
    return avg_acc, avg_loss
