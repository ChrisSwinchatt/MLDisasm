#!/usr/bin/env python3

'''
MLDisasm disassembler model.

See: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
'''

import inspect

import numpy as np

import tensorflow.keras as keras

from mldisasm.constants import START_TOKEN, STOP_TOKEN

DEFAULT_PARAMS = {
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
    'dense_activation':      'sigmoid',
    'use_softmax':           True,
    'optimizer':             'SGD',
    'opt_params':            dict(),
    'loss':                  'categorical_crossentropy',
    'metrics':               'acc'
}

def _make_params(**kwargs):
    params = dict(DEFAULT_PARAMS)
    params.update(kwargs)
    return params

def _make_recurrent_layer(params, **kwargs):
    if params['recurrent_layers'] > 1:
        num_layers = params['recurrent_layers']
        params = dict(params)
        del params['recurrent_layers']
        model = keras.Sequential()
        for _ in range(num_layers):
            model.add(_make_recurrent_layer(params))
        return model
    layer_params = {
        'units':             params['hidden_size'],
        'activation':        params['recurrent_activation'],
        'dropout':           params['dropout'],
        'use_bias':          params['recurrent_use_bias'],
        'recurrent_dropout': params['recurrent_dropout'],
        'return_sequences':  True,
        'return_state':      True
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

class DisassemblyEncoder(keras.Model):
    '''
    Encoder model.
    '''
    def __init__(self, params):
        # Create the input layer.
        self._input_shape = (params['x_seq_len'], params['input_size'])
        self.inputs      = keras.layers.Input(shape=self._input_shape)
        # Create the recurrent layer(s).
        self.layer       = _make_recurrent_layer(params, return_sequences=False)
        # Create the output.
        self.outputs     = self.layer(self.inputs)
        # The output of encoder is the predictions and either one (RNN, GRU) or two (LSTM) internal states.
        self.state = None
        if len(self.outputs) == 2:
            state = self.outputs[1]
        elif len(self.outputs) == 3:
            state = [self.outputs[1], self.outputs[2]]
        else:
            raise TypeError('Expected tuple with length of either two or three, got {} of length {}'.format(
                type(self.outputs).__name__,
                len(self.outputs)
            ))
        assert state is not None
        super().__init__(self.inputs, self.state)

class DisassemblyDecoder(keras.Model):
    '''
    Decoder model.
    '''
    def __init__(self, params, input_state):
        # Create the input layer.
        self._input_shape = (params['y_seq_len'], params['output_size'])
        self.inputs      = keras.layers.Input(shape=self._input_shape)
        # Create the hidden layer(s).
        self.layer = _make_recurrent_layer(params)
        # Create the outputs with the hidden state of the encoder as the decoder's initial state.
        self.input_state   = input_state
        self.outputs, _, _ = self.layer(self.inputs, initial_state=self.input_state)
        # Create the dense layer which maps from hidden space to output space.
        self.dense   = keras.layers.Dense(params['output_size'], params['dense_activation'])
        self.outputs = self.dense(self.outputs)
        # Set up model for inference. During inference the decoder's states are appended to its inputs.
        self.inf_states_inputs = []
        self.inf_states_inputs.append(keras.layers.Input(shape=(params['hidden_size'])))
        if params['recurrent_unit'] == 'lstm':
            # Add second state for LSTM.
            self.inf_states_inputs.append(keras.layers.Input(shape=(params['hidden_size'])))
        self.inf_outputs = self.layer(self.inputs, initial_state=self.inf_states_inputs)
        self.inf_states  = self.inf_outputs[1:]
        super().__init__(
            [self.inputs]      + self.inf_states_inputs,
            [self.inf_outputs] + self.inf_states
        )

class Disassembler(keras.Model):
    '''
    Disassembler model.
    '''
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        '''
        Create a disassembler model. After creating the disassembler, you can train it with any of the
        methods of keras.Model, such as fit(). The inputs to the model (X) is a tuple of the training inputs and
        targets; the targets of the model (y) are the same as the second input but shifted one timestep along:

        :example:
            model = Disassembler(...)
            model.fit([X, y], tf.manip.roll(y, 1, 1), ...)

        :param input_size: Dimensionality of the input space.
        :param hidden_size: Number of units per recurrent layer.
        :param output_size: Dimensionality of the output space.

        :note: The following must be passed as keyword arguments if given.

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
        :param dense_activation: Activation function for dense layer. Default: sigmoid.
        :param use_softmax: Whether to use a softmax layer. Default: True.
        :param optimizer: Which optimizer to use. Default: SGD.
        :param opt_params: Parameters to the optimizer. A common one is 'lr' for learning rate. Default: {}.
        :param loss: Loss function. Default: categorical_crossentropy.
        :param metrics: Metrics. Default: accuracy.

        :returns: A trainable disassembler model.
        '''
        self.params = _make_params(
            input_size  = input_size,
            hidden_size = hidden_size,
            output_size = output_size,
            **kwargs
        )
        self.encoder = DisassemblyEncoder(self.params)
        self.decoder = DisassemblyDecoder(self.params, self.encoder.state)
        super().__init__([self.encoder.inputs,self.decoder.inputs], self.decoder.outputs)
        self.train_mode = True
        self._compile()

    def __call__(self, *args, **kwargs):
        if self.train_mode:
            return super().__call__(*args, **kwargs)
        else:
            return self.infer(*args, **kwargs)

    def training_mode(self):
        '''
        Enable training mode.
        '''
        self.train_mode = True

    def inference_mode(self):
        '''
        Enable inference mode.
        '''
        self.train_mode = False

    def infer(self, *args, **kwargs):
        '''
        Decode an input sequence.
        :param args: Arguments to predict().
        :param kwargs: Keyword arguments to predict().
        :returns: The decoded string.
        '''
        # Get the encoded vector.
        states = self.encoder(*args, **kwargs)
        # Create the target sequence with the start token.
        y = np.zeros((1, 1, self.params['y_seq_len']))
        y[0, 0, ord(START_TOKEN)] = 1
        # Decode the sequence(s) in a loop.
        stop   = False
        result = ''
        while not stop:
            # Decoded sequence.
            outputs = self.decoder([y] + states)
            tokens  = outputs[0]
            states  = outputs[1:]
            # Store the token. Set 'stop' flag if the stop token (LF) was generated.
            index = np.argmax(tokens[0, -1, :])
            token = chr(index)
            result += token
            if token == STOP_TOKEN or len(result) >= self.params['y_seq_len']:
                stop = True
            # Update the target sequence.
            y = np.zeros((1, 1, self.params['y_seq_len']))
            y[0, 0, index] = 1
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
