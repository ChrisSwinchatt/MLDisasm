#!/usr/bin/env python3

'''
MLDisasm disassembler model.

See: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
'''

import inspect

import tensorflow.keras as keras

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

def _make_encoder(params):
    input_shape = (params['x_seq_len'], params['input_size'])
    inputs      = keras.layers.Input(shape=input_shape)
    encoder     = _make_recurrent_layer(params, return_sequences=False)
    outputs     = encoder(inputs)
    # The output of encoder is the predictions and either one (RNN, GRU) or two (LSTM) internal states.
    state = None
    if len(outputs) == 2:
        state = outputs[1]
    elif len(outputs) == 3:
        state = [outputs[1], outputs[2]]
    else:
        raise TypeError('Expected tuple with length of either two or three, got {} of length {}'.format(
            type(outputs).__name__,
            len(outputs)
        ))
    assert state is not None
    return inputs, state

def _make_decoder(params, state):
    input_shape   = (params['y_seq_len'], params['output_size'])
    inputs        = keras.layers.Input(shape=input_shape)
    decoder       = _make_recurrent_layer(params)
    outputs, _, _ = decoder(inputs, initial_state=state)
    dense         = keras.layers.Dense(params['output_size'], params['dense_activation'])
    outputs       = dense(outputs)
    return inputs, outputs

def _compile(model, params):
    optimizer = params['optimizer']
    if isinstance(optimizer, str):
        # If optimizer is str, interpret it as the name of a Keras optimizer.
        optimizer = getattr(keras.optimizers, optimizer)
        # Filter out parameters which aren't found in the optimizer's signature. This is needed for gridsearch
        # because the opt_params grid contains parameter values for all optimizers being searched.
        signature  = inspect.signature(optimizer)
        opt_params = dict(filter(
            lambda kv: kv[0] in signature.parameters,
            params['opt_params']
        ))
        # Instantiate the optimizer with the chosen parameters.
        optimizer = optimizer(**opt_params)
    model.compile(
        optimizer,
        params['loss'],
        metrics=params['metrics']
    )
    return model

def trainable_disassembler(input_size, hidden_size, output_size, **kwargs):
    '''
    Create a disassembler model for training. After creating the disassembler, you can train it with any of the methods
    of keras.Model, such as fit(). The inputs to the model (X) is a tuple of the training inputs and targets; the
    targets of the model (y) are the same as the second input but shifted one timestep along:

    :example:
        model = trainable_disassembler(...)
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
    params = _make_params(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        **kwargs
    )
    encoder_inputs, encoder_state   = _make_encoder(params)
    decoder_inputs, decoder_outputs = _make_decoder(params, encoder_state)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return _compile(model, params)
