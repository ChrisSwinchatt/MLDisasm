#!/usr/bin/env python3

'''
MLDisasm disassembler.
'''

import functools

import tensorflow.keras as keras

import mldisasm.io.log        as     log
from   mldisasm.training.loss import LOSS_FUNCTIONS

class Disassembler:
    '''
    Machine learning disassembler.
    '''
    def __init__(self, hidden_size, tokens, codec=None, **kwargs):
        '''
        Initialise Disassembler.
        :param hidden_size: How many hidden units to use in each LSTM layer.
        :param tokens: The token vocabulary.
        :param codec: If passed, decodes target strings during training.
        :note: The following optional arguments must be passed as keyword arguments.
        :param lstm_layers: How many LSTM layers to use. Default value is 1.
        :param lstm_activation: Name of the LSTM activation function. Default is tanh.
        :param lstm_dropout: Dropout rate between LSTM sequences. Default is 0.
        :param lstm_r_dropout: Dropout rate between steps within each sequence. Default is 0.
        :param use_bias: Whether to use bias vectors in LSTM layers. Default is true.
        :param lstm_forget_bias: Whether to apply unit forget bias. Default is True.
        :param dense_activation: Name of the dense layer activation function. Default is sigmoid.
        :param use_softmax: Whether to use a softmax layer. Output of a softmax layer is a vector whose values sum to 1.
        :param loss: Name of the loss function to minimise during training. Default is levenshtein.
        :param optimizer: Name of the optimiser. Default is SGD.
        :param batch_size: Batch size. Default is None (unknown).
        :param seq_len: Sequence length. Default is None (unknown).
        '''
        # Create sequential model.
        self.tokens = tokens
        self.model  = keras.Sequential(name='sequential')
        # Append LSTM layers.
        for _ in range(kwargs.get('lstm_layers', 1)):
            self.model.add(keras.layers.LSTM(
                units             = hidden_size,
                activation        = kwargs.get('lstm_activation',  'tanh'),
                dropout           = kwargs.get('lstm_dropout',     0.0),
                use_bias          = kwargs.get('use_bias',         True),
                unit_forget_bias  = kwargs.get('lstm_forget_bias', True),
                recurrent_dropout = kwargs.get('lstm_r_dropout',   0.0),
                return_sequences  = True
            ))
        # Append dense layer.
        self.model.add(keras.layers.Dense(
            units      = len(self.tokens),
            activation = kwargs.get('dense_activation', 'sigmoid')
        ))
        # Compile the model with an optimiser and loss function.
        loss = kwargs.get('loss', 'levenshtein')
        if loss in LOSS_FUNCTIONS:
            loss = functools.partial(
                LOSS_FUNCTIONS[loss],
                codec
            )
        self.model.compile(kwargs.get('optimizer', 'SGD'), loss)

    def train(self, inputs, targets):
        '''
        Train the model.
        :param inputs: A tensor of one-hot encoded real numbers.
        :param targets: A tensor of one-hot encoded strings.
        :returns: The training history, see tensorflow.keras.Model.fit().
        '''
        _validate_training_inputs(inputs, targets)
        return self.model.fit(inputs, targets, steps_per_epoch=inputs.shape[0])

    def disassemble(self, inputs):
        '''
        Produce disassembly.
        :param inputs: The input tensor.
        '''
        self.model(inputs)

    def __str__(self):
        return self.model.summary()

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
