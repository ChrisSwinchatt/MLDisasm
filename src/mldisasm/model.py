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

A decoder-only transformer trained like an LLM: takes a fused sequence of
[byte_values | assembly_tokens], predicts the next assembly token at each
position, with causal self-attention and loss masked on byte positions.
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
    encoding = tf.reshape(tf.stack([even, odd], axis=-1), [max_len, d_model])
    return encoding[tf.newaxis, :, :]


class FusedEmbedding(keras.layers.Layer):
    """Embeds a fused [bytes|tokens] sequence.

    Positions 0..x_seq_len-1 are byte values (0-255) embedded via byte_embed.
    Positions x_seq_len..end are token indices embedded via token_embed.
    """

    def __init__(self, x_seq_len, d_model, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.x_seq_len = x_seq_len
        self.byte_embed = keras.layers.Embedding(256, d_model, name='byte_embedding')
        self.token_embed = keras.layers.Embedding(vocab_size, d_model, name='token_embedding')

    def call(self, inputs):
        seq_ids = inputs  # (batch, total_seq_len)
        batch_size = tf.shape(seq_ids)[0]
        total_len = tf.shape(seq_ids)[1]

        # Split the sequence into byte and token portions.
        x_len = tf.minimum(self.x_seq_len, total_len)
        byte_part = seq_ids[:, :x_len]
        token_part = seq_ids[:, x_len:]

        byte_emb = self.byte_embed(byte_part)    # (batch, x_len, d_model)
        tok_emb = self.token_embed(token_part)   # (batch, y_len, d_model)

        return tf.concat([byte_emb, tok_emb], axis=1)


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

        super().__init__(name='disassembler')

        self.d_model = params['d_model']
        self.x_seq_len = params['x_seq_len']
        self.y_seq_len = params['y_seq_len']
        self.max_seq_len = params['max_seq_len']
        self.vocab_size = vocab_size
        self.params = params

        # Fused embedding: byte values for first x_seq_len positions,
        # token indices for the rest.
        self.fused_embed = FusedEmbedding(
            self.x_seq_len, self.d_model, vocab_size, name='fused_embed'
        )

        # Positional encoding (non-trainable).
        pos_enc = sinusoidal_position_encoding(self.max_seq_len, self.d_model)
        self.pos_encoding = tf.Variable(pos_enc, trainable=False, name='pos_encoding')

        # Transformer decoder blocks.
        self.decoder_blocks = [
            TransformerDecoderBlock(self.d_model, params['num_heads'],
                                    params['ff_dim'], params['dropout'])
            for _ in range(params['num_layers'])
        ]

        # Output projection to vocabulary.
        self.output_projection = keras.layers.Dense(vocab_size, name='output_projection')

        self._compile_model()

    def call(self, inputs, training=False):
        """Forward pass.

        :param inputs: Fused sequence (batch, total_seq_len) of ints.
                        First x_seq_len positions are byte values (0-255),
                        remaining positions are assembly token indices.
        :returns: Logits (batch, total_seq_len, vocab_size).
        """
        x = self.fused_embed(inputs)
        seq_len = tf.shape(x)[1]
        x = x + self.pos_encoding[:, :seq_len, :]

        for block in self.decoder_blocks:
            x = block(x, training=training)

        return self.output_projection(x)

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
            X = tf.convert_to_tensor(X, dtype=tf.int32)
        if X.shape.rank == 1:
            X = X[tf.newaxis, :]

        x_len = tf.shape(X)[1]
        if x_len < self.x_seq_len:
            pad = tf.zeros((1, self.x_seq_len - x_len), dtype=tf.int32)
            X = tf.concat([X, pad], axis=1)

        # Cache raw byte embeddings (no decoder block processing yet).
        cached_byte_emb = self.fused_embed.byte_embed(X)

        token_ids = [START_IDX]
        token_tensor = tf.convert_to_tensor([token_ids], dtype=tf.int32)

        for _ in range(max_tokens):
            tok_emb = self.fused_embed.token_embed(token_tensor)
            x = tf.concat([cached_byte_emb, tok_emb], axis=1)
            seq_len = tf.shape(x)[1]
            x = x + self.pos_encoding[:, :seq_len, :]

            for block in self.decoder_blocks:
                x = block(x, training=False)

            logits = self.output_projection(x)
            next_token = tf.argmax(logits[0, -1, :])
            next_token_id = tf.cast(next_token, tf.int32)

            token_ids.append(int(next_token_id))
            token_tensor = tf.convert_to_tensor([token_ids], dtype=tf.int32)

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
    'FusedEmbedding': FusedEmbedding,
}
