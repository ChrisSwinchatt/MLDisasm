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
MLDisasm disassembler model.

A decoder-only transformer that maps machine code bytes to assembly text.
"""

import inspect

import numpy as np
import tensorflow as tf
from tensorflow import keras

from mldisasm.constants import START_IDX, END_IDX, PAD_IDX, ASSEMBLY_VOCAB_SIZE

PARAMETER_DEFAULTS = {
    'd_model':       256,
    'num_heads':     8,
    'num_layers':    6,
    'ff_dim':        1024,
    'dropout':       0.1,
    'max_seq_len':   1024,
    'x_seq_len':     512,
    'y_seq_len':     256,
    'mask_value':    None,
    'optimizer':     'Adam',
    'opt_params':    {'learning_rate': 0.001},
    'loss':          'sparse_categorical_crossentropy',
    'metrics':       ['accuracy'],
    'epochs':        100,
    'batch_size':    64,
    'kfolds':        10,
    'shuffle':       True,
}


def _make_params(**kwargs):
    params = dict(PARAMETER_DEFAULTS)
    params.update(kwargs)
    return params


def sinusoidal_position_encoding(max_len, d_model):
    """Sinusoidal positional encoding, shape (1, max_len, d_model)."""
    positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
    dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angles = positions / tf.pow(10000.0, 2 * (dims // 2) / d_model)
    even = tf.sin(angles[:, 0::2])
    odd = tf.cos(angles[:, 1::2])
    # Interleave even and odd.
    encoding = tf.reshape(tf.stack([even, odd], axis=-1), [max_len, d_model])
    return encoding[tf.newaxis, :, :]


class TransformerDecoderBlock(keras.layers.Layer):
    """Transformer decoder block with causal self-attention."""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(self.ff_dim, activation='gelu'),
            keras.layers.Dense(self.d_model),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, -1)
        causal_mask = tf.cast(causal_mask, tf.bool)
        attn_out = self.attn(x, x, attention_mask=causal_mask, training=training)
        x = self.layernorm1(x + self.dropout1(attn_out, training=training))
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_out, training=training))
        return x


class Disassembler(keras.Model):
    """Decoder-only transformer for disassembly.

    Takes byte indices and token indices, embeds them, concatenates,
    applies transformer decoder blocks, and projects to vocabulary logits.
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

        super().__init__(name='disassembler')

        self.d_model = params['d_model']
        self.x_seq_len = params['x_seq_len']
        self.y_seq_len = params['y_seq_len']
        self.max_seq_len = params['max_seq_len']
        self.vocab_size = vocab_size
        self.params = params

        # Embeddings
        self.byte_embedding = keras.layers.Embedding(256, self.d_model)
        self.token_embedding = keras.layers.Embedding(vocab_size, self.d_model)

        # Positional encoding (non-trainable)
        pos_enc = sinusoidal_position_encoding(self.max_seq_len, self.d_model)
        self.pos_encoding = tf.Variable(pos_enc, trainable=False, name='pos_encoding')

        # Transformer decoder blocks
        self.decoder_blocks = [
            TransformerDecoderBlock(self.d_model, params['num_heads'], params['ff_dim'], params['dropout'])
            for _ in range(params['num_layers'])
        ]

        # Output projection
        self.output_projection = keras.layers.Dense(vocab_size)

        self._compile_model()

    def call(self, inputs, training=False):
        byte_indices, token_indices = inputs
        byte_emb = self.byte_embedding(byte_indices)
        tok_emb = self.token_embedding(token_indices)
        seq_len = tf.shape(byte_emb)[1] + tf.shape(tok_emb)[1]
        x = tf.concat([byte_emb, tok_emb], axis=1)
        x = x + self.pos_encoding[:, :seq_len, :]

        for block in self.decoder_blocks:
            x = block(x, training=training)

        return self.output_projection(x)

    def infer(self, X, max_tokens=None):
        """Autoregressive greedy decoding.

        :param X: Byte indices, shape (1, x_seq_len).
        :param max_tokens: Maximum tokens to generate (default: y_seq_len).
        :returns: Decoded assembly string.
        """
        if max_tokens is None:
            max_tokens = self.y_seq_len

        if isinstance(X, np.ndarray):
            X = tf.convert_to_tensor(X, dtype=tf.int32)
        if X.shape.rank == 1:
            X = X[tf.newaxis, :]

        byte_emb = self.byte_embedding(X)

        token_ids = [START_IDX]
        token_tensor = tf.convert_to_tensor([token_ids], dtype=tf.int32)

        for _ in range(max_tokens):
            tok_emb = self.token_embedding(token_tensor)
            seq_len = tf.shape(byte_emb)[1] + tf.shape(tok_emb)[1]
            x = tf.concat([byte_emb, tok_emb], axis=1)
            x = x + self.pos_encoding[:, :seq_len, :]

            for block in self.decoder_blocks:
                x = block(x, training=False)

            logits = self.output_projection(x)
            next_token = tf.argmax(logits[0, -1, :]).numpy()
            token_ids.append(int(next_token))
            token_tensor = tf.convert_to_tensor([token_ids], dtype=tf.int32)

            if next_token == END_IDX:
                break

        return self._detokenize(token_ids)

    def _detokenize(self, token_ids):
        """Convert token indices back to assembly text."""
        idx_to_char = {i + 4: chr(i) for i in range(128)}
        idx_to_char[START_IDX] = '\t'
        idx_to_char[END_IDX] = '\n'
        return ''.join(
            idx_to_char.get(t, '?')
            for t in token_ids
            if t != PAD_IDX
        )

    def _compile_model(self):
        optimizer = self.params['optimizer']
        if isinstance(optimizer, str):
            optimizer_cls = getattr(keras.optimizers, optimizer)
            signature = inspect.signature(optimizer_cls)
            filtered_params = {
                k: v for k, v in self.params['opt_params'].items()
                if k in signature.parameters
            }
            optimizer = optimizer_cls(**filtered_params)
        loss = self.params.get('loss', 'sparse_categorical_crossentropy')
        self.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=self.params['metrics']
        )


CUSTOM_OBJECTS = {
    'Disassembler': Disassembler,
    'TransformerDecoderBlock': TransformerDecoderBlock,
}
