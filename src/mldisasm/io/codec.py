#!/usr/bin/env python3

'''
MLDisasm encoding and decoding.
'''

from abc import ABCMeta, abstractmethod

import numpy as np

import tensorflow as tf

import mldisasm.io.log as log

# Maximum value of a byte.
BYTE_MAX  = 0xFF

class Codec(metaclass=ABCMeta):
    '''
    Coder/decoder.
    '''
    @abstractmethod
    def encode(self, seq):
        '''
        Encode a sequence. Sequence will be padded to seq_len if necessary.
        '''
        raise NotImplementedError

    @abstractmethod
    def decode(self, tensor):
        '''
        Decode a sequence of length 'seq_len'.
        '''
        raise NotImplementedError

def _recursive_map(f, t, *args, axis=0, **kwargs):
    '''
    Recursively map a function over a tensor until reaching the given axis.
    '''
    if t.shape.ndims == axis + 1:
        return tf.map_fn(f, t, *args, **kwargs)
    return tf.map_fn(lambda x: _recursive_map(f, x, *args, **kwargs), t, *args, **kwargs)

class AsciiCodec(Codec):
    '''
    Encode ASCII as token indices, or decode token indices into ASCII.
    '''
    def __init__(self, seq_len, mask_value, tokens):
        self._seq_len    = seq_len
        self._mask_value = mask_value
        self._tokens     = tokens

    def encode(self, seq, as_tensor=True):
        '''
        Encodes the contents of an ASCII string as a vector of token indices.
        :param s: The ASCII string.
        :param checked: Set to True if the input has already been checked for validity.
        :param as_tensor: Whether to encode to a tensor or return a list.
        :returns: A tensor with one element per token in the string.
        '''
        # Check parameter.
        if not isinstance(seq, str):
            raise TypeError('Expected str, not {}'.format(type(seq).__name__))
        if not seq:
            raise ValueError('Received empty string')
        # Tokenise the string and convert TokenList indices to reals between 0 and 1.
        tokens  = self._tokens.tokenize(seq)
        indices = [[float(self._tokens.index(t))/len(self._tokens)] for t in tokens]
        # Pad to seq_len and convert to tensor.
        if len(indices) > self._seq_len:
            log.warning('Expected {} elements or fewer, got {}'.format(self._seq_len, len(indices)))
        while len(indices) < self._seq_len:
            indices.append([self._mask_value])
        if as_tensor:
            return tf.convert_to_tensor(indices, dtype=tf.int64)
        return indices

    def decode(self, indices):
        '''
        Decode a tensor of token indices into an ASCII string tensor.
        :param indices: The token indices.
        :returns: An ASCII string tensor.
        '''
        # Check parameters.
        if not isinstance(indices, tf.Tensor) and not isinstance(indices, np.ndarray):
            raise TypeError('Expected Tensor or ndarray, not {}'.format(type(indices).__name__))
        # Convert real valued outputs into TokenList indices.
        indices = tf.cast(tf.round(indices*len(self._tokens)), tf.int32)
        # Convert indices into tokens and join into a string per example in the batch. This has to be done on the CPU as
        # there is no GPU kernel for string ops.
        with tf.device('/cpu:0'):
            return tf.reduce_join(
                _recursive_map(
                    lambda i: tf.cond(
                        i >= 0,
                        true_fn  = lambda: self._tokens.as_tensor[i],
                        false_fn = lambda: tf.convert_to_tensor('')
                    ),
                    indices,
                    dtype=tf.string
                ),
                axis=[1,0]
            )

class BytesCodec(Codec):
    '''
    Encodes bytes to one-hot, or decodes one hot into bytes.
    '''
    def __init__(self, seq_len, mask_value):
        '''
        Initialise BytesCodec.
        :param seq_len: The sequence length.
        '''
        self._seq_len    = seq_len
        self._mask_value = mask_value

    def encode(self, bs, as_tensor=True):
        '''
        Encode a bytes object as a tensor of float values.
        :param bs: The bytes.
        :param as_tensor: Whether to return a tensor (True) or a list (False).
        :returns: A tensor or list of float values.
        '''
        if not isinstance(bs, bytes):
            raise TypeError('Expected bytes, not {}'.format(type(bs).__name__))
        bslen = len(bs)
        if bslen > self._seq_len:
            raise ValueError('Length of bytes ({}) is larger than sequence length ({})'.format(bslen, self._seq_len))
        xs = [[float(byte)/BYTE_MAX] for byte in bs]
        if len(xs) > self._seq_len:
            log.warning('Expected {} elements or fewer, got {}'.format(self._seq_len, len(xs)))
        while len(xs) < self._seq_len:
            xs.append([self._mask_value])
        if as_tensor:
            return tf.convert_to_tensor(xs)
        return xs

    def decode(self, tensor):
        '''
        Decode a float tensor into a bytes object.
        '''
        if not isinstance(tensor, tf.Tensor):
            raise TypeError('Expected Tensor, not {}'.format(type(tensor).__name__))
        # Convert float tensor into array of bytes.
        xs = (tensor*BYTE_MAX).eval()
        # Filter out infinity values.
        xs = filter(lambda x: x != 0, xs)
        # Convert to bytes.
        return bytes(map(int, xs))
