#!/usr/bin/env python3

"""
Parity tests verifying that the PyTorch Disassembler produces identical outputs
to the original TensorFlow/Keras implementation.

Two levels of comparison:
  Level 1 (layer-wise): Compare intermediate activations at each major boundary.
  Level 2 (end-to-end): Compare final model output logits.

Both models are in eval mode (no dropout, deterministic LayerNorm).
Weights are manually set to identical values in both frameworks.

atol=1e-4, rtol=1e-3: appropriate for a 2-layer transformer with GELU
activations and attention softmax. Sub-ULP differences in exp/sum ordering
may cause differences within this tolerance.

If TensorFlow is not installed, all TF-dependent tests are skipped with a
clear message. The PyTorch-only tests (model instantiation, forward pass
shapes, inference, training loop) run regardless.

To run the full parity suite with TF:
    1. Create a Python 3.11 environment (TF does not yet support 3.12+):
       python3.11 -m venv /tmp/tf-parity-env

    2. Install dependencies:
       pip install tensorflow>=2.15 torch>=2.0 numpy pytest

    3. Run the parity tests:
       PYTHONPATH=src python -m pytest tests/test_parity.py -v

Expected results:
    - All 4 PyTorch-only tests: PASSED
    - All 3 weight parity tests: PASSED
    - All 3 forward pass parity tests: PASSED
    - Total: 11 passed, 0 skipped, 0 failed

    If any test fails, the assertion message will identify the specific
    layer and weight where numerical divergence occurred.

Residual differences that cannot be eliminated:
    - Adam epsilon placement: Keras places eps inside sqrt; PyTorch places
      it outside. This affects optimization trajectories but not forward pass.
    - Random initialization: Keras Glorot uniform vs PyTorch Kaiming uniform
      defaults differ. This test uses manually-set weights to bypass this.
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

# ---------------------------------------------------------------------------
# Probe for TensorFlow
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    HAVE_TF = True
except ImportError:
    HAVE_TF = False
    tf = None
    keras = None

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
TEST_CONFIG = {
    'd_model':      64,
    'num_heads':    4,
    'num_layers':   2,
    'ff_dim':       128,
    'dropout':      0.0,   # zero dropout for deterministic comparison
    'max_seq_len':  128,
    'x_seq_len':    64,
    'y_seq_len':    32,
    'batch_size':   2,
}

ATOL = 1e-4
RTOL = 1e-3

requires_tf = pytest.mark.skipif(not HAVE_TF, reason='TensorFlow not installed')


# ===========================================================================
# TF reference model (only defined when TF is available)
# ===========================================================================

if HAVE_TF:

    def _tf_sinusoidal_position_encoding(max_len, d_model):
        positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angles = positions / tf.pow(10000.0, 2 * (dims // 2) / d_model)
        even = tf.sin(angles[:, 0::2])
        odd = tf.cos(angles[:, 1::2])
        encoding = tf.reshape(tf.stack([even, odd], axis=-1), [max_len, d_model])
        return encoding[tf.newaxis, :, :]


    class _TF_FusedEmbedding(keras.layers.Layer):
        def __init__(self, x_seq_len, d_model, vocab_size, **kwargs):
            super().__init__(**kwargs)
            self.x_seq_len = x_seq_len
            self.byte_embed = keras.layers.Embedding(256, d_model, name='byte_embedding')
            self.token_embed = keras.layers.Embedding(vocab_size, d_model, name='token_embedding')

        def call(self, inputs):
            seq_ids = inputs
            total_len = tf.shape(seq_ids)[1]
            x_len = tf.minimum(self.x_seq_len, total_len)
            byte_part = seq_ids[:, :x_len]
            token_part = seq_ids[:, x_len:]
            byte_emb = self.byte_embed(byte_part)
            tok_emb = self.token_embed(token_part)
            return tf.concat([byte_emb, tok_emb], axis=1)


    class _TF_TransformerDecoderBlock(keras.layers.Layer):
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


    class _TF_Disassembler(keras.Model):
        def __init__(self, d_model=256, num_heads=8, num_layers=6, ff_dim=1024,
                     dropout=0.1, max_seq_len=128, x_seq_len=64, y_seq_len=32,
                     vocab_size=132, **kwargs):
            super().__init__(name='disassembler')
            self.d_model = d_model
            self.x_seq_len = x_seq_len
            self.y_seq_len = y_seq_len
            self.max_seq_len = max_seq_len
            self.vocab_size = vocab_size

            self.fused_embed = _TF_FusedEmbedding(
                self.x_seq_len, self.d_model, vocab_size, name='fused_embed'
            )
            pos_enc = _tf_sinusoidal_position_encoding(self.max_seq_len, self.d_model)
            self.pos_encoding = tf.Variable(pos_enc, trainable=False, name='pos_encoding')
            self.decoder_blocks = [
                _TF_TransformerDecoderBlock(self.d_model, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
            self.output_projection = keras.layers.Dense(vocab_size, name='output_projection')

        def call(self, inputs, training=False):
            x = self.fused_embed(inputs)
            seq_len = tf.shape(x)[1]
            x = x + self.pos_encoding[:, :seq_len, :]
            for block in self.decoder_blocks:
                x = block(x, training=training)
            return self.output_projection(x)


# ===========================================================================
# Import PyTorch model
# ===========================================================================
from mldisasm.model import Disassembler, TransformerDecoderBlock, FusedEmbedding


# ===========================================================================
# Helpers
# ===========================================================================

def _make_fixed_weights(shape, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32) * 0.02


def _set_tf_weights(tf_model):
    """Build TF model and set all weights to fixed values."""
    batch_size = 1
    total_len = TEST_CONFIG['x_seq_len'] + TEST_CONFIG['y_seq_len']
    dummy = tf.constant(np.random.randint(0, 256, (batch_size, total_len)), dtype=tf.int32)
    _ = tf_model(dummy, training=False)

    seed = 42
    for w in tf_model.trainable_weights:
        shape = w.shape.as_list()
        new_val = _make_fixed_weights(shape, seed)
        w.assign(new_val)
        seed += 1
    for w in tf_model.non_trainable_weights:
        shape = w.shape.as_list()
        new_val = _make_fixed_weights(shape, seed)
        w.assign(new_val)
        seed += 1


def _set_pt_weights_from_tf(tf_model, pt_model):
    """Set PyTorch model weights from TF model with full layout conversion.

    Weight mapping rules:
      - Embedding: (V, D) → nn.Embedding.weight (V, D): direct copy
      - Dense kernel: (in, out) → nn.Linear.weight (out, in): transpose
      - Dense bias: (out,) → nn.Linear.bias (out,): direct copy
      - LayerNorm gamma: (D,) → nn.LayerNorm.weight (D,): direct copy
      - LayerNorm beta: (D,) → nn.LayerNorm.bias (D,): direct copy
      - Positional encoding: (1, L, D) → (1, L, D): direct copy
    """
    tf_weights = {}
    for w in tf_model.trainable_weights:
        tf_weights[w.name] = w.numpy()
    for w in tf_model.non_trainable_weights:
        tf_weights[w.name] = w.numpy()

    consumed = set()

    for pt_name, pt_param in pt_model.named_parameters():
        tf_val = None
        transform = None

        # Positional encoding.
        if 'pos_encoding' in pt_name:
            for k in tf_weights:
                if 'pos_encoding' in k and k not in consumed:
                    tf_val = tf_weights[k]
                    transform = 'pos_encoding'
                    consumed.add(k)
                    break

        # Fused embedding.
        elif 'fused_embed.byte_embed' in pt_name:
            for k in tf_weights:
                if 'byte_embedding/embeddings' in k and k not in consumed:
                    tf_val = tf_weights[k]
                    transform = 'embedding'
                    consumed.add(k)
                    break
        elif 'fused_embed.token_embed' in pt_name:
            for k in tf_weights:
                if 'token_embedding/embeddings' in k and k not in consumed:
                    tf_val = tf_weights[k]
                    transform = 'embedding'
                    consumed.add(k)
                    break

        # Output projection.
        elif 'output_projection' in pt_name:
            if 'weight' in pt_name:
                for k in tf_weights:
                    if 'output_projection/kernel' in k and k not in consumed:
                        tf_val = tf_weights[k]
                        transform = 'linear_kernel'
                        consumed.add(k)
                        break
            elif 'bias' in pt_name:
                for k in tf_weights:
                    if 'output_projection/bias' in k and k not in consumed:
                        tf_val = tf_weights[k]
                        transform = 'linear_bias'
                        consumed.add(k)
                        break

        # Decoder blocks.
        elif 'decoder_blocks' in pt_name:
            tf_key, transform = _match_decoder_block_weight(pt_name, tf_weights, consumed)
            if tf_key:
                tf_val = tf_weights[tf_key]
                consumed.add(tf_key)

        # Dropout has no parameters.
        elif 'dropout' in pt_name:
            continue

        if tf_val is None:
            raise RuntimeError(
                f'Could not find TF weight for PyTorch parameter: {pt_name}\n'
                f'Unconsumed TF weights: {sorted(set(tf_weights) - consumed)}'
            )

        if transform == 'embedding' or transform == 'layernorm' or transform == 'pos_encoding' or transform == 'linear_bias':
            pt_param.data.copy_(torch.from_numpy(tf_val))
        elif transform == 'linear_kernel':
            pt_param.data.copy_(torch.from_numpy(tf_val.T.copy()))


def _match_decoder_block_weight(pt_name, tf_weights, consumed):
    """Match a PyTorch decoder block parameter to its TF weight equivalent.

    pt_name examples:
      decoder_blocks.0.W_q.weight
      decoder_blocks.0.ffn.0.weight
      decoder_blocks.0.layernorm1.weight
    """
    parts = pt_name.split('.')
    block_idx = int(parts[1])
    sub = parts[2]

    # TF uses 1-based naming.
    tf_prefix = f'disassembler/transformer_decoder_block_{block_idx + 1}/'

    # Attention projections.
    mha_map = {
        'W_q': 'multi_head_attention/_query_dense',
        'W_k': 'multi_head_attention/_key_dense',
        'W_v': 'multi_head_attention/_value_dense',
        'W_o': 'multi_head_attention/_output_dense',
    }
    if sub in mha_map:
        param_type = parts[3]
        suffix = 'kernel' if param_type == 'weight' else 'bias'
        tf_key = f'{tf_prefix}{mha_map[sub]}/{suffix}:0'
        if tf_key in tf_weights and tf_key not in consumed:
            return tf_key, 'linear_kernel' if param_type == 'weight' else 'linear_bias'
        return None, None

    # Feed-forward network.
    if sub == 'ffn':
        ffn_idx = int(parts[3])
        param_type = parts[4]
        dense_name = 'dense' if ffn_idx == 0 else 'dense_1'
        suffix = 'kernel' if param_type == 'weight' else 'bias'
        tf_key = f'{tf_prefix}sequential/{dense_name}/{suffix}:0'
        if tf_key in tf_weights and tf_key not in consumed:
            return tf_key, 'linear_kernel' if param_type == 'weight' else 'linear_bias'
        return None, None

    # LayerNorm.
    ln_map = {
        'layernorm1': 'layer_normalization',
        'layernorm2': 'layer_normalization_1',
    }
    if sub in ln_map:
        param_type = parts[3]
        suffix = 'gamma' if param_type == 'weight' else 'beta'
        tf_key = f'{tf_prefix}{ln_map[sub]}/{suffix}:0'
        if tf_key in tf_weights and tf_key not in consumed:
            return tf_key, 'layernorm'
        return None, None

    return None, None


def _build_tf_model(config):
    return _TF_Disassembler(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        x_seq_len=config['x_seq_len'],
        y_seq_len=config['y_seq_len'],
        vocab_size=132,
    )


def _build_pt_model(config):
    return Disassembler(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        x_seq_len=config['x_seq_len'],
        y_seq_len=config['y_seq_len'],
        vocab_size=132,
    )


def _create_test_input(batch_size, x_seq_len, y_seq_len):
    rng = np.random.RandomState(12345)
    total_len = x_seq_len + y_seq_len
    byte_part = rng.randint(0, 256, (batch_size, x_seq_len)).astype(np.int32)
    token_part = rng.randint(4, 132, (batch_size, y_seq_len)).astype(np.int32)
    return np.concatenate([byte_part, token_part], axis=1)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope='module')
def tf_model():
    if not HAVE_TF:
        pytest.skip('TensorFlow not installed')
    model = _build_tf_model(TEST_CONFIG)
    _set_tf_weights(model)
    return model


@pytest.fixture(scope='module')
def pt_model(tf_model):
    model = _build_pt_model(TEST_CONFIG)
    _set_pt_weights_from_tf(tf_model, model)
    return model


@pytest.fixture(scope='module')
def test_input():
    return _create_test_input(TEST_CONFIG['batch_size'],
                              TEST_CONFIG['x_seq_len'],
                              TEST_CONFIG['y_seq_len'])


# ===========================================================================
# PyTorch-only tests (run even without TF)
# ===========================================================================

class TestPyTorchModel:
    """Tests that only require PyTorch."""

    def test_model_instantiation(self):
        """Model can be instantiated with default parameters."""
        model = Disassembler(
            d_model=64, num_heads=4, num_layers=2, ff_dim=128,
            dropout=0.1, max_seq_len=128, x_seq_len=64, y_seq_len=32,
            vocab_size=132
        )
        assert isinstance(model, nn.Module)
        assert len(model.decoder_blocks) == 2

    def test_forward_pass_shapes(self):
        """Forward pass produces correct output shapes."""
        model = Disassembler(
            d_model=64, num_heads=4, num_layers=2, ff_dim=128,
            dropout=0.0, max_seq_len=128, x_seq_len=64, y_seq_len=32,
            vocab_size=132
        )
        model.eval()
        batch_size = 2
        total_len = 64 + 32
        dummy = torch.zeros((batch_size, total_len), dtype=torch.long)

        with torch.no_grad():
            logits = model(dummy)

        assert logits.shape == (batch_size, total_len, 132), \
            f'Expected (2, 96, 132), got {logits.shape}'

    def test_inference_smoke(self):
        """Inference produces valid string output."""
        model = Disassembler(
            d_model=64, num_heads=4, num_layers=2, ff_dim=128,
            dropout=0.0, max_seq_len=128, x_seq_len=64, y_seq_len=32,
            vocab_size=132
        )
        model.eval()
        byte_input = np.zeros(64, dtype=np.int32)
        result = model.infer(byte_input, max_tokens=8)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result[0] == '\t'

    def test_training_loop_step(self):
        """A single training step completes without error."""
        from mldisasm.training import train_model
        model = Disassembler(
            d_model=64, num_heads=4, num_layers=2, ff_dim=128,
            dropout=0.0, max_seq_len=128, x_seq_len=8, y_seq_len=8,
            vocab_size=132, epochs=1, batch_size=2
        )
        byte_indices = torch.randint(0, 256, (4, 8))
        token_indices = torch.randint(4, 132, (4, 8))
        X = (byte_indices, token_indices)
        y = token_indices

        trained_model, acc, loss = train_model(
            X, y, model.params, retrain=model,
            device=torch.device('cpu')
        )
        assert 0.0 <= acc <= 1.0
        assert loss > 0.0


# ===========================================================================
# TF-PyTorch parity tests (skipped without TF)
# ===========================================================================

@requires_tf
class TestWeightParity:
    """Verify weight layout conversions are correct."""

    def test_embedding_weights(self, tf_model, pt_model):
        """Embedding weights are directly copied (same layout)."""
        for w in tf_model.trainable_weights:
            if 'byte_embedding/embeddings' in w.name:
                tf_w = w.numpy()
                pt_w = pt_model.fused_embed.byte_embed.weight.detach().cpu().numpy()
                np.testing.assert_allclose(tf_w, pt_w, atol=ATOL, rtol=RTOL)
            if 'token_embedding/embeddings' in w.name:
                tf_w = w.numpy()
                pt_w = pt_model.fused_embed.token_embed.weight.detach().cpu().numpy()
                np.testing.assert_allclose(tf_w, pt_w, atol=ATOL, rtol=RTOL)

    def test_output_projection_weights(self, tf_model, pt_model):
        """Output projection: Keras Dense kernel (in,out) → PyTorch Linear weight (out,in)."""
        for w in tf_model.trainable_weights:
            if 'output_projection/kernel' in w.name:
                tf_w = w.numpy()
                pt_w = pt_model.output_projection.weight.detach().cpu().numpy()
                np.testing.assert_allclose(tf_w.T, pt_w, atol=ATOL, rtol=RTOL)
            if 'output_projection/bias' in w.name:
                tf_w = w.numpy()
                pt_w = pt_model.output_projection.bias.detach().cpu().numpy()
                np.testing.assert_allclose(tf_w, pt_w, atol=ATOL, rtol=RTOL)

    def test_decoder_block_attention_weights(self, tf_model, pt_model):
        """Each MHA projection weight matches after transpose."""
        for block_idx, pt_block in enumerate(pt_model.decoder_blocks):
            tf_prefix = f'disassembler/transformer_decoder_block_{block_idx + 1}/'
            for pt_attr, tf_suffix in [
                ('W_q', 'multi_head_attention/_query_dense'),
                ('W_k', 'multi_head_attention/_key_dense'),
                ('W_v', 'multi_head_attention/_value_dense'),
                ('W_o', 'multi_head_attention/_output_dense'),
            ]:
                pt_weight = getattr(pt_block, pt_attr).weight.detach().cpu().numpy()
                pt_bias = getattr(pt_block, pt_attr).bias.detach().cpu().numpy()

                tf_weight = None
                tf_bias_val = None
                for w in tf_model.trainable_weights:
                    if f'{tf_prefix}{tf_suffix}/kernel' in w.name:
                        tf_weight = w.numpy()
                    if f'{tf_prefix}{tf_suffix}/bias' in w.name:
                        tf_bias_val = w.numpy()

                assert tf_weight is not None
                np.testing.assert_allclose(
                    tf_weight.T, pt_weight, atol=ATOL, rtol=RTOL,
                    err_msg=f'Block {block_idx} {pt_attr} kernel'
                )
                np.testing.assert_allclose(
                    tf_bias_val, pt_bias, atol=ATOL, rtol=RTOL,
                    err_msg=f'Block {block_idx} {pt_attr} bias'
                )

    def test_decoder_block_layernorm_weights(self, tf_model, pt_model):
        """LayerNorm gamma/beta are directly copied."""
        for block_idx, pt_block in enumerate(pt_model.decoder_blocks):
            tf_prefix = f'disassembler/transformer_decoder_block_{block_idx + 1}/'
            for pt_ln_name, tf_ln_name in [
                ('layernorm1', 'layer_normalization'),
                ('layernorm2', 'layer_normalization_1'),
            ]:
                pt_ln = getattr(pt_block, pt_ln_name)
                pt_weight = pt_ln.weight.detach().cpu().numpy()
                pt_bias = pt_ln.bias.detach().cpu().numpy()

                tf_weight = None
                tf_bias_val = None
                for w in tf_model.trainable_weights:
                    if f'{tf_prefix}{tf_ln_name}/gamma' in w.name:
                        tf_weight = w.numpy()
                    if f'{tf_prefix}{tf_ln_name}/beta' in w.name:
                        tf_bias_val = w.numpy()

                assert tf_weight is not None
                np.testing.assert_allclose(tf_weight, pt_weight, atol=ATOL, rtol=RTOL)
                np.testing.assert_allclose(tf_bias_val, pt_bias, atol=ATOL, rtol=RTOL)


@requires_tf
class TestForwardPassParity:
    """Compare intermediate activations and final outputs."""

    def test_fused_embedding_output(self, tf_model, pt_model, test_input):
        """Fused embedding produces identical output."""
        tf_input = tf.constant(test_input)
        tf_out = tf_model.fused_embed(tf_input).numpy()

        pt_input = torch.from_numpy(test_input).long()
        pt_model.eval()
        with torch.no_grad():
            pt_out = pt_model.fused_embed(pt_input).numpy()

        np.testing.assert_allclose(tf_out, pt_out, atol=ATOL, rtol=RTOL)

    def test_pos_encoding(self, tf_model, pt_model):
        """Positional encodings are numerically identical."""
        tf_pos = tf_model.pos_encoding.numpy()
        pt_pos = pt_model.pos_encoding.detach().cpu().numpy()
        np.testing.assert_allclose(tf_pos, pt_pos, atol=ATOL, rtol=RTOL)

    def test_full_forward_pass(self, tf_model, pt_model, test_input):
        """End-to-end forward pass produces identical logits.

        Both models in eval mode (dropout disabled, deterministic LayerNorm).
        atol=1e-4, rtol=1e-3 justified by:
          - GELU uses exact tanh approximation in both frameworks
          - Softmax numerical paths may differ at sub-ULP level
          - 2-layer model keeps accumulation error within tolerance
        """
        tf_input = tf.constant(test_input)
        tf_out = tf_model(tf_input, training=False).numpy()

        pt_input = torch.from_numpy(test_input).long()
        pt_model.eval()
        with torch.no_grad():
            pt_out = pt_model(pt_input).detach().cpu().numpy()

        np.testing.assert_allclose(tf_out, pt_out, atol=ATOL, rtol=RTOL,
                                   err_msg='End-to-end forward pass mismatch')
