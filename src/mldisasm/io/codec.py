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

class AsciiCodec(Codec):
    '''
    Encode ASCII as token indices, or decode token indices into ASCII.
    '''
    def __init__(self, seq_len, tokens):
        self.seq_len = seq_len
        self.tokens  = tokens

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
        # Tokenise the string and convert to TokenList indices.
        tokens  = self.tokens.tokenize(seq)
        indices = [[self.tokens.index(t)] for t in tokens]
        # Pad to seq_len and convert to tensor.
        while len(indices) < self.seq_len:
            indices.append([-1])
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
        if not isinstance(indices, tf.Tensor):
            raise TypeError('Expected Tensor, not {}'.format(type(indices).__name__))
        ndim = len(indices.shape)
        if ndim > 1:
            # Recursively map and reduce over nD tensor until n=1.
            return tf.reduce_join(tf.map_fn(self.decode, indices, dtype=tf.string))
        # Convert indices into tokens and join into a single string. This has to be done on the CPU as there is no GPU
        # kernel for string ops.
        with tf.device('/cpu:0'):
            return tf.reduce_join(tf.map_fn(
                lambda idx: tf.cond(
                    idx >= 0,
                    true_fn  = lambda: self.tokens.as_tensor[idx],
                    false_fn = lambda: tf.convert_to_tensor('')
                ),
                indices,
                dtype=tf.string
            ))

class BytesCodec(Codec):
    '''
    Encodes bytes to one-hot, or decodes one hot into bytes.
    '''
    def __init__(self, seq_len):
        '''
        Initialise BytesCodec.
        :param seq_len: The sequence length.
        '''
        self.seq_len = seq_len

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
        if bslen > self.seq_len:
            raise ValueError('Length of bytes ({}) is larger than sequence length ({})'.format(bslen, self.seq_len))
        xs = [[float(byte)/BYTE_MAX] for byte in bs]
        while len(xs) < self.seq_len:
            xs.append([np.inf])
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
        xs = filter(lambda x: x != np.inf, xs)
        # Convert to bytes.
        return bytes(map(int, xs))
