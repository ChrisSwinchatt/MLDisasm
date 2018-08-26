#!/usr/bin/env python3

'''
MLDisasm disassembler model.
'''

import inspect

import tensorflow.keras as keras

class Disassembler(keras.Sequential):
    '''
    Disassembler.
    '''
    def __init__(self, hidden_size, output_size, **kwargs):
        '''
        Create a Disassembler model.
        :param hidden_size: How many hidden units to use in each LSTM layer.
        :param output_size: Dimensionality of the output.
        :note: The following parameters must be passed as keyword arguments.
        :param lstm_layers: How many LSTM layers to use. Default value is 1.
        :param lstm_activation: Name of the LSTM activation function. Default is 'tanh'.
        :param lstm_dropout: Dropout rate between LSTM sequences. Default is 0.
        :param lstm_r_dropout: Dropout rate between steps within each sequence. Default is 0.
        :param lstm_use_bias: Whether to use bias vectors in LSTM layers. Default is true.
        :param lstm_forget_bias: Whether to apply unit forget bias. Default is True.
        :param dense_units: Number of units in the dense layer. Default is 1.
        :param dense_activation: Name of the dense layer activation function. Default is 'sigmoid'.
        :param loss: Name of the loss function to minimise during training. Default is 'mean_squared_error'.
        :param optimizer: Name of the optimiser. Default is 'SGD' (stochastic gradient descent).
        :param opt_params: Dictionary of optimiser parameters. Default is empty dict. Values are optimizer dependent, but a
        common one is 'lr' (learning rate).
        :param batch_size: Batch size. Default is None (variable size).
        :param seq_len: Sequence length. Default is None (variable length).
        :param mask_value: Mask value. Default is None (no masking).
        '''
        super().__init__()
        # Save parameters.
        self.params = {
            'hidden_size':      hidden_size,
            'output_size':      output_size,
            'seq_len':          kwargs.get('seq_len',          None),
            'use_masking':      kwargs.get('use_masking',      False),
            'mask_value':       kwargs.get('mask_value',       None),
            'lstm_layers':      kwargs.get('lstm_layers',      1),
            'lstm_activation':  kwargs.get('lstm_activation',  'tanh'),
            'lstm_dropout':     kwargs.get('lstm_dropout',     0.0),
            'lstm_use_bias':    kwargs.get('lstm_use_bias',    True),
            'lstm_forget_bias': kwargs.get('lstm_forget_bias', True),
            'lstm_r_dropout':   kwargs.get('lstm_r_dropout',   0.0),
            'dense_activation': kwargs.get('dense_activation', 'sigmoid'),
            'use_softmax':      kwargs.get('use_softmax',      False),
            'optimizer':        kwargs.get('optimizer',        'SGD'),
            'opt_params':       kwargs.get('opt_params',       dict()),
            'loss':             kwargs.get('loss',             'categorical_crossentropy'),
            'metrics':          kwargs.get('metrics',          ['acc'])
        }
        # Add input layer.
        input_shape = (self.params['seq_len'], 1)
        self.add(keras.layers.InputLayer(input_shape))
        # Add masking layer.
        if self.params['use_masking'] and self.params['mask_value'] is not None:
            self.add(keras.layers.Masking(self.params['mask_value'], input_shape=input_shape))
        # Append LSTM layers.
        for _ in range(self.params['lstm_layers']):
            self.add(keras.layers.LSTM(
                units             = self.params['hidden_size'],
                activation        = self.params['lstm_activation'],
                dropout           = self.params['lstm_dropout'],
                use_bias          = self.params['lstm_use_bias'],
                unit_forget_bias  = self.params['lstm_forget_bias'],
                recurrent_dropout = self.params['lstm_r_dropout'],
                return_sequences  = True
            ))
        # Append dense layer.
        self.add(keras.layers.Dense(
            self.params['output_size'],
            self.params['dense_activation']
        ))
        # Append softmax layer.
        if self.params['use_softmax']:
            self.add(keras.layers.Softmax(input_shape=self.input_shape))
        # Compile the model with optimiser and loss function.
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
