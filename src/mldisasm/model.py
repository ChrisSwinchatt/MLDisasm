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
MLDisasm disassembler model.

A decoder-only transformer trained like an LLM: takes a fused sequence of
[byte_values | assembly_tokens], predicts the next assembly token at each
position, with causal self-attention and loss masked on byte positions.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from mldisasm.constants import START_IDX, END_IDX, PAD_IDX, ASSEMBLY_VOCAB_SIZE

PARAMETER_DEFAULTS = {
    'd_model':       256,
    'num_heads':     8,
    'num_layers':    6,
    'ff_dim':        1024,
    'dropout':       0.1,
    'max_seq_len':   8192,
    'x_seq_len':     4096,
    'y_seq_len':     2048,
    'optimizer':     'Adam',
    'opt_params':    {'learning_rate': 0.001},
    'loss':          'sparse_categorical_crossentropy',
    'metrics':       ['accuracy'],
    'epochs':        100,
    'batch_size':    16,
    'kfolds':        10,
    'shuffle':       True,
}


def _make_params(**kwargs):
    params = dict(PARAMETER_DEFAULTS)
    params.update(kwargs)
    return params


def sinusoidal_position_encoding(max_len, d_model):
    """Sinusoidal positional encoding, shape (1, max_len, d_model)."""
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    angles = position / torch.pow(10000.0, 2 * (dims // 2) / d_model)
    even = torch.sin(angles[:, 0::2])
    odd = torch.cos(angles[:, 1::2])
    encoding = torch.stack([even, odd], dim=-1).reshape(max_len, d_model)
    return encoding.unsqueeze(0)


class FusedEmbedding(nn.Module):
    """Embeds a fused [bytes|tokens] sequence.

    Positions 0..x_seq_len-1 are byte values (0-255) embedded via byte_embed.
    Positions x_seq_len..end are token indices embedded via token_embed.
    """

    def __init__(self, x_seq_len, d_model, vocab_size):
        super().__init__()
        self.x_seq_len = x_seq_len
        self.byte_embed = nn.Embedding(256, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)

    def forward(self, seq_ids):
        """seq_ids: (batch, total_seq_len) int tensor."""
        batch_size = seq_ids.shape[0]
        total_len = seq_ids.shape[1]

        x_len = min(self.x_seq_len, total_len)
        byte_part = seq_ids[:, :x_len]
        token_part = seq_ids[:, x_len:]

        byte_emb = self.byte_embed(byte_part)
        tok_emb = self.token_embed(token_part)

        return torch.cat([byte_emb, tok_emb], dim=1)


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with causal self-attention.

    Uses manual multi-head attention (explicit Q/K/V/O Linear projections)
    for transparent weight mapping and parity-verifiable behavior.
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** 0.5

        # Multi-head attention projections.
        # Weight layout: nn.Linear(in, out) has weight (out, in).
        # Each is equivalent to a Keras Dense(in, out) with transposed weight.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Feed-forward network.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

        # LayerNorm: Keras default epsilon is 1e-6 (PyTorch default is 1e-5).
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout_rate = dropout
        self.dropout_attn = nn.Dropout(dropout)   # attention weights
        self.dropout1 = nn.Dropout(dropout)         # MHA output
        self.dropout2 = nn.Dropout(dropout)         # FFN output

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        d_model = self.d_model
        h = self.num_heads
        d_k = self.head_dim

        # Linear projections.
        q = self.W_q(x)  # (batch, seq, d_model)
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape to (batch, num_heads, seq, head_dim).
        q = q.view(batch_size, seq_len, h, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, h, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, h, d_k).transpose(1, 2)

        # Scaled dot-product attention.
        scores = (q @ k.transpose(-2, -1)) / self.scale

        # Causal mask: lower-triangular, -inf for future positions.
        # Keras equivalent: tf.linalg.band_part(ones, -1, -1) as bool mask
        # where True=attend. PyTorch uses additive float mask.
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )
        scores = scores + causal_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        attn_out = attn_weights @ v  # (batch, num_heads, seq, head_dim)

        # Reshape back: (batch, num_heads, seq, head_dim) → (batch, seq, d_model).
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_out = self.W_o(attn_out)

        # Residual + norm.
        x = self.layernorm1(x + self.dropout1(attn_out))

        # Feed-forward.
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_out))

        return x


class Disassembler(nn.Module):
    """Decoder-only transformer for disassembly, trained like an LLM.

    Takes a single fused sequence of [byte_values | assembly_token_indices],
    predicts the next assembly token at each position using causal attention.

    The byte portion is embedded via a 256-entry byte embedding table;
    the token portion via the assembly vocabulary embedding table.
    Loss is only computed on token positions (byte positions are masked).
    """

    def __init__(self, d_model=256, num_heads=8, num_layers=6, ff_dim=1024,
                 dropout=0.1, max_seq_len=1024, x_seq_len=512, y_seq_len=256,
                 vocab_size=None, optimizer='Adam', opt_params=None, loss=None,
                 metrics=None, epochs=100, batch_size=64, kfolds=10,
                 shuffle=True, mask_value=None, **kwargs):
        if vocab_size is None:
            vocab_size = ASSEMBLY_VOCAB_SIZE

        params = _make_params(
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
            ff_dim=ff_dim, dropout=dropout, max_seq_len=max_seq_len,
            x_seq_len=x_seq_len, y_seq_len=y_seq_len,
            optimizer=optimizer, opt_params=opt_params or {},
            loss=loss, metrics=metrics, epochs=epochs,
            batch_size=batch_size, kfolds=kfolds, shuffle=shuffle,
            mask_value=mask_value, **kwargs
        )

        super().__init__()

        self.d_model = params['d_model']
        self.x_seq_len = params['x_seq_len']
        self.y_seq_len = params['y_seq_len']
        self.max_seq_len = params['max_seq_len']
        self.vocab_size = vocab_size
        self.params = params

        # Fused embedding: byte values for first x_seq_len positions,
        # token indices for the rest.
        self.fused_embed = FusedEmbedding(
            self.x_seq_len, self.d_model, vocab_size
        )

        # Positional encoding (non-trainable buffer).
        pos_enc = sinusoidal_position_encoding(self.max_seq_len, self.d_model)
        self.register_buffer('pos_encoding', pos_enc)

        # Transformer decoder blocks.
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(self.d_model, params['num_heads'],
                                    params['ff_dim'], params['dropout'])
            for _ in range(params['num_layers'])
        ])

        # Output projection to vocabulary.
        # Keras Dense(vocab_size) weight shape (d_model, vocab_size).
        # PyTorch nn.Linear(d_model, vocab_size) weight shape (vocab_size, d_model).
        # During weight porting: transpose the Keras weight.
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, inputs):
        """Forward pass.

        :param inputs: Fused sequence (batch, total_seq_len) of ints.
                        First x_seq_len positions are byte values (0-255),
                        remaining positions are assembly token indices.
        :returns: Logits (batch, total_seq_len, vocab_size).
        """
        x = self.fused_embed(inputs)
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]

        for block in self.decoder_blocks:
            x = block(x)

        return self.output_projection(x)

    @torch.no_grad()
    def infer(self, X, max_tokens=None):
        """Autoregressive greedy decoding.

        Byte embeddings are cached since they never change during decoding.
        The full fused sequence is rebuilt and run through all decoder blocks
        each step.

        :param X: Byte values, shape (x_seq_len,) or (1, x_seq_len).
        :param max_tokens: Maximum tokens to generate (default: y_seq_len).
        :returns: Decoded assembly string.
        """
        if max_tokens is None:
            max_tokens = self.y_seq_len

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(dtype=torch.int32)
        if X.dim() == 1:
            X = X.unsqueeze(0)

        x_len = X.shape[1]
        if x_len < self.x_seq_len:
            pad = torch.zeros((1, self.x_seq_len - x_len), dtype=torch.int32)
            X = torch.cat([X, pad], dim=1)

        # Cache raw byte embeddings (no decoder block processing yet).
        cached_byte_emb = self.fused_embed.byte_embed(X)

        token_ids = [START_IDX]
        token_tensor = torch.tensor([token_ids], dtype=torch.int32)

        for _ in range(max_tokens):
            tok_emb = self.fused_embed.token_embed(token_tensor)
            x = torch.cat([cached_byte_emb, tok_emb], dim=1)
            seq_len = x.shape[1]
            x = x + self.pos_encoding[:, :seq_len, :]

            for block in self.decoder_blocks:
                x = block(x)

            logits = self.output_projection(x)
            next_token = torch.argmax(logits[0, -1, :])
            next_token_id = int(next_token.item())

            token_ids.append(next_token_id)
            token_tensor = torch.tensor([token_ids], dtype=torch.int32)

            if next_token_id == END_IDX:
                break

        return self._detokenize(token_ids)

    def _detokenize(self, token_ids):
        """Convert token indices to assembly text."""
        idx_to_char = {i + 4: chr(i) for i in range(128)}
        idx_to_char[START_IDX] = '\t'
        idx_to_char[END_IDX] = '\n'
        return ''.join(
            idx_to_char.get(t, '?')
            for t in token_ids
            if t != PAD_IDX
        )
