#!/usr/bin/env python3

'''
MLDisasm training set.
'''

import numpy as np

import tensorflow as tf

# Maximum value of a byte.
BYTE_MAX  = 0xFF

# Maximum value of an ASCII character.
ASCII_MAX = 0x7F

# Size of an input vector (one-hot).
INPUT_SIZE = BYTE_MAX + 1

# Size of a target vector (one-hot).
TARGET_SIZE = ASCII_MAX + 1

def one_hot_bytes(bs):
    '''
    One-hot encodes the contents of a bytes object.
    :param bs: The bytes object.
    :returns: A tensor with one row per byte in the input object, and INPUT_SIZE (256) elements per row. One element
    per row will have the value 1 and the others will be zero.
    '''
    indices = list(bs)
    return tf.one_hot(indices, depth=INPUT_SIZE)

def one_hot_ascii(s):
    '''
    One-hot encodes the contents of an ASCII string.
    :param s: The ASCII string.
    :returns: A tensor with one row per character in the input string, and TARGET_SIZE (128) elements per row. One
    element per row will have the value 1 and the others will be zero.
    '''
    indices = [ord(c) for c in s]
    return tf.one_hot(indices, depth=TARGET_SIZE)

class TrainingSet:
    '''
    Allows iterating over training set data.
    '''

    def __init__(self, file, x_encoder=one_hot_bytes, y_encoder=one_hot_ascii, shuffled=False):
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
