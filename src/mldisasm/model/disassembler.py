#!/usr/bin/env python3

'''
MLDisasm disassembler.
'''

import tensorflow       as tf
import tensorflow.keras as keras

from mldisasm.training.loss import levenshtein_loss

class Disassembler:
    '''
    Machine learning disassembler.
    '''
    LOSS_FUNCTIONS = {
        'levenshtein': levenshtein_loss
    }

    def __init__(self, **kwargs):
        '''
        Initialise Disassembler.
        :param num_units: How many hidden units to use in each layer.
        :param num_layers: How many hidden layers to use. Default value is 1.
        '''
        self.model = keras.Sequential()
        # Add LSTM layers.
        for _ in range(kwargs.get('lstm_layers', d=1)):
            self.model.add(keras.layers.LSTM(
                units             = kwargs.get('lstm_units'),
                activation        = kwargs.get('lstm_activation',  d='tanh'),
                dropout           = kwargs.get('lstm_dropout',     d=0.0),
                unit_forget_bias  = kwargs.get('lstm_forget_bias', d=True),
                recurrent_dropout = kwargs.get('lstm_r_dropout',   d=0.0)
            ))
        # Add linear layer.
        self.model.add(keras.layers.Dense(
            units      = kwargs.get('dense_units'),
            activation = kwargs.get('dense_activation', d='sigmoid')
        ))
        # Add softmax if configured.
        if kwargs.get('use_softmax'):
            self.model.add(keras.layers.Softmax())
        # Compile the model with an optimiser and loss function.
        loss = kwargs.get('loss', d='levenshtein')
        if loss in Disassembler.LOSS_FUNCTIONS:
            loss = Disassembler.LOSS_FUNCTIONS[loss]
        self.model.compile(kwargs.get('optimizer'), loss)

    def train(self, inputs, targets):
        '''
        Train the model.
        :param inputs: A tensor of one-hot encoded real numbers.
        :param targets: A tensor of one-hot encoded strings.
        :returns: The training history, see tensorflow.keras.Model.fit().
        '''
        return self.model.fit(inputs, targets)

    def disassemble(self, inputs):
        '''
        Produce disassembly.
        :param inputs: The input tensor.
        '''
        self.model(inputs)
