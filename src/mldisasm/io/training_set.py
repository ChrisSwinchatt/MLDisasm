#!/usr/bin/env python3

'''
MLDisasm training set.
'''

import numpy as np

import tensorflow as tf

from mldisasm.io.codec import ascii_to_one_hot, bytes_to_one_hot

class TrainingSet:
    '''
    Allows iterating over training set data.
    '''

    def __init__(self, file, x_encoder=bytes_to_one_hot, y_encoder=ascii_to_one_hot, shuffled=False):
        '''
        Initialise TrainingSet.
        :param file: A path or handle to the file containing the training set.
        :param x_encoder: A callable which encodes the input bytes into a tensor. Default is to use one-hot encoding.
        :param y_encoder: A callable which encodes the target string into a tensor. Default is to use one-hot encoding.
        :param shuffled: Whether to shuffle the examples.
        '''
        if isinstance(file, str):
            file = open(file, 'rb')
        self._file      = file
        self._x_encoder = x_encoder
        self._y_encoder = y_encoder
        self._randomise = False
        self._max_seek  = 0
        if shuffled:
            self.shuffle()

    def __len__(self):
        '''
        Get the number of examples in the training set.
        :returns: The number of examples in the training set.
        '''
        if self._max_seek > 0:
            return self._max_seek
        pos = self._file.tell()
        self._file.seek(0)
        self._max_seek = len([_ for _ in self._file])
        self._file.seek(pos)
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
        '''
        # Seek to a random position in the file in shuffle mode.
        if self._randomise:
            self._seek()
        line = next(self._file)
        # Split on |.
        elems = line.split('|')
        if len(elems) != 2:
            raise ValueError('Bad line in training file: {}'.format(line))
        # Return encoded tensors.
        opcode = bytes([int(elems[0], 16)])
        X = self._x_encoder(opcode)
        y = self._y_encoder(elems[1])
        return X, y

    def _seek(self):
        '''
        Seek to the beginning or a random position in the file.
        '''
        if self._randomise:
            self._file.seek(np.random.randint(0, self._max_seek))
        else:
            self._file.seek(0)
