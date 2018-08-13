#!/usr/bin/env python3

'''
MLDisasm disassembler model.
'''

import tensorflow.keras as keras

def make_disassembler(hidden_size, **kwargs):
    '''
    Create a disassembler model.
    :param hidden_size: How many hidden units to use in each LSTM layer.
    :note: The following optional arguments must be passed as keyword arguments.
    :param lstm_layers: How many LSTM layers to use. Default value is 1.
    :param lstm_activation: Name of the LSTM activation function. Default is tanh.
    :param lstm_dropout: Dropout rate between LSTM sequences. Default is 0.
    :param lstm_r_dropout: Dropout rate between steps within each sequence. Default is 0.
    :param use_bias: Whether to use bias vectors in LSTM layers. Default is true.
    :param lstm_forget_bias: Whether to apply unit forget bias. Default is True.
    :param dense_activation: Name of the dense layer activation function. Default is sigmoid.
    :param use_softmax: Whether to use a softmax layer. Output of a softmax layer is a vector whose values sum to 1.
    :param loss: Name of the loss function to minimise during training. Default is mean squared error.
    :param optimizer: Name of the optimiser. Default is SGD.
    :param batch_size: Batch size. Default is None (variable size).
    :param seq_len: Sequence length. Default is None (variable length).
    :param mask_value: Mask value. Default is None (no masking).
    :param vocab_size: The size of the vocabulary so this can be checked before making predictions. Default is None.
    :returns: The model.
    '''
    model = keras.Sequential()
    # Add input layer.
    input_shape = (kwargs.get('seq_len', None), 1)
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    # Add masking layer.
    if kwargs.get('mask_value', None) is not None:
        model.add(keras.layers.Masking(
            mask_value=kwargs.get('mask_value'),
            input_shape=input_shape
        ))
    # Append LSTM layers.
    for _ in range(kwargs.get('lstm_layers', 1)):
        model.add(keras.layers.LSTM(
            units             = hidden_size,
            activation        = kwargs.get('lstm_activation',  'tanh'),
            dropout           = kwargs.get('lstm_dropout',     0.0),
            use_bias          = kwargs.get('use_bias',         True),
            unit_forget_bias  = kwargs.get('lstm_forget_bias', True),
            recurrent_dropout = kwargs.get('lstm_r_dropout',   0.0),
            return_sequences  = True
        ))
    # Append dense layer.
    model.add(keras.layers.Dense(
        units      = 1,
        activation = kwargs.get('dense_activation', 'sigmoid')
    ))
    # Compile and return the model with optimiser and loss function.
    model.compile(
        kwargs.get('optimizer', 'SGD'),
        kwargs.get('loss',      'mean_squared_error')
    )
    return model

def _validate_training_inputs(inputs, targets):
    '''
    Validate shape of training inputs/targets.
    '''
    if inputs.shape.ndims != 3:
        raise ValueError('Incorrect dimensionality {} of input tensor (wanted 3)'.format(inputs.shape.ndims))
    if inputs.shape != targets.shape:
        raise ValueError('Dimension 0 (batch size) of inputs and targets must match (got {} and {})'.format(
            inputs.shape,
            targets.shape
        ))
