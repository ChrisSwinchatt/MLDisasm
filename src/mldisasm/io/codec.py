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
MLDisasm encoding and decoding with tokenization.
"""

import numpy as np
import torch

from mldisasm.constants import ASCII_MAX, START_TOKEN, STOP_TOKEN

# Vocabulary size for assembly text.
ASSEMBLY_VOCAB_SIZE = ASCII_MAX + 4  # printable ASCII + PAD, START, END, UNK

# Special token indices.
PAD_IDX   = 0
START_IDX = 1
END_IDX   = 2
UNK_IDX   = 3


class ByteIndexer:
    """Maps raw bytes to integer indices for embedding lookup."""

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def encode(self, bs, as_tensor=True):
        """Convert bytes to padded integer indices (0-255, 0 for padding)."""
        if isinstance(bs, str):
            bs = bytes.fromhex(bs)
        if not isinstance(bs, (bytes, bytearray)):
            raise TypeError(f'Expected bytes, got {type(bs).__name__}')
        indices = list(bs)[:self.seq_len]
        pad_len = self.seq_len - len(indices)
        indices = indices + [0] * pad_len
        result = np.array(indices, dtype=np.int32)
        if as_tensor:
            return torch.tensor(result, dtype=torch.int32)
        return result

    def decode(self, tensor):
        """Decode integer indices back to bytes, skipping padding."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        return bytes(b for b in tensor if b != 0)


class AssemblyTokenizer:
    """Character-level tokenizer for assembly text."""

    def __init__(self, seq_len):
        self.seq_len = seq_len
        # Build char-to-index mapping.
        self._char_to_idx = {chr(i): i + 4 for i in range(ASCII_MAX + 1)}
        # Overwrite with special characters
        self._char_to_idx['\t'] = START_IDX
        self._char_to_idx['\n'] = END_IDX
        self._idx_to_char = {v: k for k, v in self._char_to_idx.items()}

    @property
    def vocab_size(self):
        return ASSEMBLY_VOCAB_SIZE

    def encode(self, text, as_tensor=True):
        """Convert assembly text to padded integer indices."""
        if not isinstance(text, str):
            raise TypeError(f'Expected str, got {type(text).__name__}')
        # Add start/end tokens if not present.
        if not text.startswith(START_TOKEN):
            text = START_TOKEN + text
        if not text.endswith(STOP_TOKEN):
            text = text + STOP_TOKEN
        # Convert to indices.
        indices = [self._char_to_idx.get(c, UNK_IDX) for c in text]
        indices = indices[:self.seq_len]
        pad_len = self.seq_len - len(indices)
        indices = indices + [PAD_IDX] * pad_len
        result = np.array(indices, dtype=np.int32)
        if as_tensor:
            return torch.tensor(result, dtype=torch.int32)
        return result

    def decode(self, indices):
        """Convert integer indices back to assembly text."""
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        if indices.ndim > 1:
            return [self.decode(row) for row in indices]
        return ''.join(
            self._idx_to_char.get(int(i), '?')
            for i in indices
            if i not in (PAD_IDX,)
        )
