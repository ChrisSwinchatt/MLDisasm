#!/usr/bin/env python3

'''
MLDisasm training set.
'''

import numpy as np

import tensorflow as tf

from   mldisasm.io.codec import default_ascii_codec, default_bytes_codec, BYTEORDER
import mldisasm.io.log   as     log

# Training set delimiter.
DELIMITER = '|'

# Training set encoding.
ENCODING = 'ascii'

# Delimiter decoded to bytes.
DELIMITER_BYTES = bytes(DELIMITER, ENCODING)

# Every two hex chars represent one byte.
CHARS_PER_BYTE = 2

class TrainingSet:
    '''
    Allows iterating over training set data.
    '''

    def __init__(self, file, batch_size=1, x_encoder=default_bytes_codec, y_encoder=default_ascii_codec, shuffled=False):
        '''
        Initialise TrainingSet.
        :param file: A path or handle to the file containing the training set.
        :param batch_size: Size of a batch of training examples. If this is not a clean divisor of the total training
        set size, the last batch will be smaller than the others.
        :param x_encoder: A callable which encodes the input bytes into a tensor. Default is to use one-hot encoding.
        :param y_encoder: A callable which encodes the target string into a tensor. Default is to use one-hot encoding.
        :param shuffled: Whether to shuffle the examples.
        '''
        if batch_size < 1:
            batch_size = 1
        if isinstance(file, str):
            file = open(file, 'r')
        self.batch_size  = batch_size
        self._file       = file
        self._x_encoder  = x_encoder
        self._y_encoder  = y_encoder
        self._randomise  = False
        self._max_seek   = 0
        self._index      = 0
        if shuffled:
            self.shuffle()

    def __len__(self):
        '''
        Get the number of examples in the training set.
        :returns: The number of examples in the training set.
        '''
        if self._max_seek > 0:
            return self._max_seek
        self._file.seek(0)
        self._max_seek = len([_ for _ in self._file])
        self._file.seek(self._index)
        return self._max_seek

    def shuffle(self):
        '''
        Return the examples in shuffled order from now on. Note: Depending on the size of the dataset, there is a chance
        this can return examples multiple times. If `N` is the number of examples, the probability of returning an
        example `k` times is ~1/(N^k).
        '''
        self._randomise = True
        self._max_seek  = len([_ for _ in self._file])
        self._seek()

    def __iter__(self):
        '''
        Get an iterator to the training set.
        '''
        self._seek()
        return self

    def __next__(self):
        '''
        Get the next item in the set.
        :returns: A tuple of (example,targets)
        '''
        # Seek to a random position in the file in shuffle mode.
        if self._randomise:
            self._seek()
        # Retrieve a batch of examples.
        log.debug('Loading a batch of {}'.format(self.batch_size))
        examples = [None]*self.batch_size
        targets  = [None]*self.batch_size
        for i in range(self.batch_size):
            X, y = self._get_single_pair()
            examples[i] = X
            targets[i]  = y
        # Convert lists to tensors.
        return tf.stack(examples), tf.stack(targets)

    def _seek(self):
        '''
        Seek to the beginning or a random position in the file.
        '''
        self._index = 0
        if self._randomise:
            self._index = np.random.randint(self._max_seek)
        self._file.seek(self._index)

    def _get_single_pair(self):
        '''
        Get a single (example,target) pair from the training set.
        '''
        line = next(self._file)
        # Split on |.
        elems = line.split(DELIMITER)
        if len(elems) != 2:
            raise ValueError('training:{}: Bad training example: {}'.format(self._index, line))
        try:
            # Return encoded tensors.
            target       = elems[1]
            opcode       = int(elems[0], 16)
            opcode_len   = int(0.5 + len(elems[0])/CHARS_PER_BYTE)
            opcode_bytes = opcode.to_bytes(opcode_len, BYTEORDER)
            X = self._x_encoder.encode(opcode_bytes)
            y = self._y_encoder.encode(target)
            self._index += len(line)
            return X, y
        except ValueError as e:
            # Re-raise the exception with a better message. 'raise ... from None' tells Python not to produce output
            # like 'while handling ValueError, another exception occurred'.
            raise ValueError('training:{}:{}: {}'.format(
                self._index,
                elems[0],
                str(e)
            )) from None
