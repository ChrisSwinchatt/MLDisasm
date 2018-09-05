#!/usr/bin/env python3

'''
MLDisasm disassembler model.

Based on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
'''

import inspect

import numpy as np

import tensorflow       as tf
import tensorflow.keras as keras

from mldisasm.constants import START_TOKEN, STOP_TOKEN

# Default values for model parameters.
PARAMETER_DEFAULTS = {
    'x_seq_len':             None,
    'y_seq_len':             None,
    'mask_value':            None,
    'recurrent_unit':        'lstm',
    'recurrent_layers':      1,
    'recurrent_activation':  'tanh',
    'recurrent_use_bias':    True,
    'recurrent_forget_bias': True,
    'dropout':               0.0,
    'recurrent_dropout':     0.0,
    'dense_activation':      'softmax',
    'use_softmax':           True,
    'optimizer':             'SGD',
    'opt_params':            dict(),
    'loss':                  'categorical_crossentropy',
    'metrics':               'acc'
}

def _make_params(**kwargs):
    '''
    Create parameter set from keyword args and defaults.
    '''
    params = dict(PARAMETER_DEFAULTS)
    params.update(kwargs)
    return params

def _split_recurrent_states(outputs):
    '''
    Split the output of a recurrent unit with return_state enabled into output and state tensors. This is needed
    because the LSTM returns two hidden states, the GRU and RNN return one hidden state, and a unit with return_state
    disabled returns only the output tensor.
    '''
    n = len(outputs)
    if n == 1:
        return outputs, None
    elif n == 2:
        return outputs[0], outputs[1]
    elif n == 3:
        return outputs[0], [outputs[1],outputs[2]]
    raise TypeError('Expected tuple with 1 output and either 1 or 2 states, got {} of length {} instead'.format(
        type(outputs).__name__,
        n
    ))

def _make_recurrent_layer(params, **kwargs):
    '''
    Make a recurrent layer depending on `params` and `kwargs`.
    '''
    # Select only those parameters which are valid for the recurrent unit we're going to use. Fortunately SimpleRNN, GRU
    # and LSTM take mostly the same parameters, with the exception that only LSTM takes the unit_forget_bias parameter.
    layer_params = {
        'units':             params['hidden_size'],
        'activation':        params['recurrent_activation'],
        'dropout':           params['dropout'],
        'use_bias':          params['recurrent_use_bias'],
        'recurrent_dropout': params['recurrent_dropout']
    }
    layer_params.update(kwargs)
    if params['recurrent_unit'] == 'rnn':
        return keras.layers.SimpleRNN(**layer_params)
    elif params['recurrent_unit'] == 'gru':
        return keras.layers.GRU(**layer_params)
    elif params['recurrent_unit'] == 'lstm':
        layer_params['unit_forget_bias'] = params['recurrent_forget_bias']
        return keras.layers.LSTM(**layer_params)
    else:
        raise ValueError('Invalid value for recurrent_unit: {}'.format(params['recurrent_unit']))

class RecurrentStack(keras.layers.Layer):
    '''
    A stack of recurrent layers.
    '''
    def __init__(self, params, **kwargs):
        super().__init__()
        # Create a copy of params so we can modify it.
        self.params = params
        # Create layer(s).
        self.layers = []
        if self.params['recurrent_layers'] > 1:
            # Add first recurrent layer(s), which have return_sequences=True and return_state=False.
            return_sequences           = kwargs['return_sequences']
            return_state               = kwargs['return_state']
            kwargs['return_sequences'] = True
            kwargs['return_state']     = False
            for _ in range(self.params['recurrent_layers'] - 1):
                self.layers.append(_make_recurrent_layer(self.params, **kwargs))
            # Restore original values.
            kwargs['return_sequences'] = return_sequences
            kwargs['return_state']     = return_state
        # Add the last layer, which gets the original values of return_sequences and return_state.
        self.layers.append(_make_recurrent_layer(self.params, **kwargs))

    def compute_output_shape(self, input_shape):
        return self.layers[-1].compute_output_shape(input_shape)

    def call(self, X, **kwargs):
        '''
        Compute the forwards pass.
        '''
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        X = self.layers[0](X, **kwargs)
        if 'initial_state' in kwargs:
            del kwargs['initial_state'] # Pass initial_state only to the first layer.
        for layer in self.layers[1:]:
            if isinstance(X, tuple):
                X = X[0]
            X = layer(X, **kwargs)
        return X

class DisassemblyEncoder:
    '''
    Encoder model.
    '''
    def __init__(self, params):
        # Create the input layer.
        self.input_shape = (params['x_seq_len'], params['input_size'])
        self.inputs      = keras.layers.Input(shape=self.input_shape)
        # Create the recurrent layer(s).
        params['recurrent_layers'] = params['encoder_layers']
        self.recurrent = RecurrentStack(
            params,
            return_sequences = False,
            return_state     = True
        )
        del params['recurrent_layers']
        # Create the output. The output of encoder is the predictions and either one (RNN, GRU) or two (LSTM) internal
        # states. We just want the states to pass to the decoder.
        outputs = self.recurrent(self.inputs)
        self.outputs, self.state = _split_recurrent_states(outputs)
        assert self.state is not None
        # Create the model.
        self.model = keras.Model(self.inputs, self.state)

    def call(self, X, **kwargs):
        '''
        Forwards pass. See `keras.Model.call`
        '''
        return self.model(X, **kwargs)

    def __call__(self, X, **kwargs):
        return self.call(X, **kwargs)

class DisassemblyDecoder:
    '''
    Decoder model.
    '''
    def __init__(self, params, initial_state):
        # Create the input layer.
        self.input_shape = (params['y_seq_len'], params['output_size'])
        self.inputs      = keras.layers.Input(shape=self.input_shape)
        # Create the hidden layer(s).
        params['recurrent_layers'] = params['decoder_layers']
        self.recurrent = RecurrentStack(
            params,
            return_sequences = True,
            return_state     = True
        )
        del params['recurrent_layers']
        # Create the outputs with the hidden state of the encoder as the decoder's initial state.
        outputs    = self.recurrent(self.inputs, initial_state=initial_state)
        outputs, _ = _split_recurrent_states(outputs)
        # Create the dense layer which maps from hidden space to output space.
        self.dense   = keras.layers.Dense(params['output_size'], params['dense_activation'])
        self.outputs = self.dense(outputs)
        # Set up model for inference. During inference the decoder's states are appended to its inputs.
        state_shape       = (params['hidden_size'],)
        inf_states_inputs = []
        inf_states_inputs.append(keras.layers.Input(shape=state_shape))
        if params['recurrent_unit'] == 'lstm':
            # Add second state for LSTM.
            inf_states_inputs.append(keras.layers.Input(shape=state_shape))
        inf_outputs             = self.recurrent(self.inputs, initial_state=inf_states_inputs)
        inf_outputs, inf_states = _split_recurrent_states(inf_outputs)
        # Create the inference model.
        self.model = keras.Model(
            [self.inputs] + inf_states_inputs,
            [inf_outputs] + inf_states
        )

    def call(self, X, **kwargs):
        '''
        Forwards pass. See `keras.Model.call`
        '''
        return self.model(X, **kwargs)

    def __call__(self, X, **kwargs):
        return self.call(X, **kwargs)

class Disassembler(keras.Model):
    '''
    Disassembler model.
    '''
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        '''
        Create a disassembler model. After creating the disassembler, you can train it with any of the methods of
        keras.Model, such as fit(). The inputs to the model (X) is a tuple of the training inputs and targets; the
        targets of the model (y) are the same as the second input but shifted one timestep along:

        :example:
            model = Disassembler(...)
            model.fit([X, y], tf.manip.roll(y, 1, 1), ...)

        :param input_size: Dimensionality of the input space.
        :param hidden_size: Number of units per recurrent layer.
        :param output_size: Dimensionality of the output space.

        :note: The following must be passed as keyword arguments if given.
        :note: Unless otherwise specified all parameters apply to both the encoder and decoder, so e.g. setting
        recurrent_layers to 2 creates two recurrent layers in *both* the encoder and the decoder (4 layers total).

        :param x_seq_len: Length of the input sequence. Default: None (variable).
        :param y_seq_len: Length of the output sequence. Default: None (variable).
        :param mask_value: Mask value. Timesteps with the mask value are skipped. Default: None (no masking).
        :param recurrent_unit: Type of recurrent unit. Possible values are: 'gru', 'rnn', 'lstm'. Default: lstm.
        :param recurrent_layers: Number of recurrent layers. Default: 1.
        :param recurrent_activation: Recurrent activation function. Default: tanh.
        :param recurrent_use_bias: Whether to use bias in the recurrent layers. Default: True
        :param recurrent_forget_bias: Whether to use forget bias in LSTM layers. Default: True.
        :param dropout: Dropout rate between sequences. Default: 0.
        :param recurrent_dropout: Dropout rate within sequences. Default: 0.
        :param dense_activation: Activation function for the decoder's dense layer. Default: softmax.
        :param optimizer: Which optimizer to use. Default: SGD.
        :param opt_params: Parameters to the optimizer. A common one is 'lr' for learning rate. Default: {}.
        :param loss: Loss function. Default: categorical_crossentropy.
        :param metrics: Metrics. Default: accuracy.

        :returns: A trainable disassembler model.
        '''
        params = _make_params(
            input_size  = input_size,
            hidden_size = hidden_size,
            output_size = output_size,
            **kwargs
        )
        encoder = DisassemblyEncoder(params)
        decoder = DisassemblyDecoder(params, encoder.state)
        super().__init__([encoder.inputs, decoder.inputs], decoder.outputs)
        self.params  = params
        self.encoder = encoder
        self.decoder = decoder
        self._compile()

    def call(self, X, **kwargs):
        '''
        Forwards pass. See `keras.Model.call`
        '''
        if kwargs.get('training', False):
            return super().call(X, **kwargs)
        return self.infer(X, **kwargs)

    def infer(self, X, **kwargs):
        '''
        Decode an input sequence.
        :param args: Arguments to predict().
        :param kwargs: Keyword arguments to predict().
        :returns: The decoded string.
        '''
        # Promote to float32.
        X = tf.cast(X, tf.float32)
        # Get the encoded vector.
        states = self.encoder(X, **kwargs)
        # Create the target sequence with the start token.
        y = np.zeros((1, self.params['y_seq_len'], self.params['output_size']), dtype=np.float32)
        y[0, 0, ord(START_TOKEN)] = 1
        y = tf.Variable(y)
        # Decode the sequence(s) in a loop.
        result = ''
        while True:
            # Decoded sequence.
            outputs = self.decoder([y] + states)
            tokens, states = _split_recurrent_states(outputs)
            # Store the token. Set 'stop' flag if the stop token (LF) was generated.
            index   = np.argmax(tokens[0, -1, :])
            token   = chr(index)
            result += token
            # Update the target sequence.
            y = np.zeros((1, self.params['y_seq_len'], self.params['output_size']), dtype=np.float32)
            y[0, 0, index] = 1
            y = tf.Variable(y)
            # Break on stop token or when the maximum string length is exceeded.
            if token == STOP_TOKEN or len(result) >= self.params['y_seq_len']:
                break
        return result

    def _compile(self):
        optimizer = self.params['optimizer']
        if isinstance(optimizer, str):
            # If optimizer is str, interpret it as the name of a Keras optimizer.
            optimizer = getattr(keras.optimizers, optimizer)
            # Filter out parameters which aren't found in the optimizer's signature. This is needed for gridsearch
            # because the opt_params grid contains parameter values for all optimizers being searched.
            signature  = inspect.signature(optimizer)
            opt_params = dict(filter(
                lambda kv: kv[0] in signature.parameters,
                self.params['opt_params']
            ))
            # Instantiate the optimizer with the chosen parameters.
            optimizer = optimizer(**opt_params)
        self.compile(
            optimizer,
            self.params['loss'],
            metrics=self.params['metrics']
        )

# Custom objects
CUSTOM_OBJECTS = {
    'Disassembler':       Disassembler,
    'DisassemblyDecoder': DisassemblyDecoder,
    'DisassemblyEncoder': DisassemblyEncoder,
    'RecurrentStack':     RecurrentStack
}
