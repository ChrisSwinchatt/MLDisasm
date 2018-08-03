#!/usr/bin/env python3

'''
MLDisasm encoding and decoding.
'''

from abc import ABCMeta, abstractmethod
import time

import tensorflow as tf

import mldisasm.io.log as log

# Maximum value of a byte.
BYTE_MAX  = 0xFF

# Maximum value of an ASCII character.
ASCII_MAX = 0x7F

# Size of an input vector (one-hot).
INPUT_SIZE = BYTE_MAX + 1

# Size of a target vector (one-hot).
TARGET_SIZE = ASCII_MAX + 1

# Sequence length.
SEQ_LEN   = 100

# Data byte-order.
BYTEORDER = 'little'

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
        Decode a one-hot tensor of length 'seq_len'.
        '''
        raise NotImplementedError

def _recursive_map(func, tensor, axis=0):
    '''
    Recursively map a function over a tensor.
    '''
    ndim = len(tensor.shape)
    if ndim == axis + 1:
        return tf.map_fn(func, tensor)
    return tf.map_fn(
        lambda x: _recursive_map(func, x, axis),
        tensor
    )

def _collect_indices(tensor):
    '''
    Collect hot indices from one-hot encoded vectors.
    '''
    return _recursive_map(
        lambda x: tf.cast(tf.argmax(x), tf.float32),
        tensor,
        axis=1
    )

class AsciiCodec(Codec):
    '''
    Encode ASCII as one-hot, or decode one-hot into ASCII.
    '''
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def encode(self, seq):
        '''
        One-hot encodes the contents of an ASCII string.
        :param s: The ASCII string.
        :param checked: Set to True if the input has already been checked for validity.
        :returns: A tensor with one row per character in the input string, and TARGET_SIZE (128) elements per row. One
        element per row will have the value 1 and the others will be zero.
        '''
        if not isinstance(seq, str):
            raise TypeError('Expected str, not {}'.format(type(seq).__name__))
        indices = [ord(c) for c in seq]
        for i, x in enumerate(indices):
            if x < 0 or x > ASCII_MAX:
                raise ValueError('{}: Bad ASCII value: Expected [0,{}], got {}.'.format(
                    i,
                    ASCII_MAX,
                    x
                ))
        # Pad with empty OH vectors until TARGET_LEN.
        while len(indices) < self.seq_len:
            indices.append(-1)
        start = time.time()
        t = tf.one_hot(indices, depth=TARGET_SIZE)
        elapsed = time.time() - start
        log.info('Encoded ASCII vector in {} seconds'.format(elapsed))
        return t

    def decode(self, tensor):
        '''
        Decode a one-hot encoded tensor into an ASCII string.
        :param tensor: The one-hot encoded tensor.
        '''
        # Check parameters.
        ndim = len(tensor.shape)
        if not isinstance(tensor, tf.Tensor):
            raise TypeError('Expected Tensor, not {}'.format(type(tensor).__name__))
        if not isinstance(tensor, tf.Tensor):
            raise TypeError('Expected Tensor, not {}'.format(type(tensor).__name__))
        if ndim not in (2,3):
            raise ValueError('Expected 2D or 3D tensor, not {}D'.format(len(tensor.shape)))
        if tensor.shape[-1] != TARGET_SIZE:
            raise ValueError('Expected size of dim {} to be {}, not {}'.format(
                ndim - 1,
                TARGET_SIZE,
                tensor.shape[1]
            ))
        indices = _collect_indices(tensor)
        string = tf.as_string(
            _recursive_map(
                lambda c: tf.cast(c, tf.string),
                indices,
                axis=0
            )
        )
        return string

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

    def encode(self, bs):
        '''
        Encode a bytes object as a one-hot tensor.
        '''
        if not isinstance(bs, bytes):
            raise TypeError('Expected bytes, not {}'.format(type(bs).__name__))
        bslen = len(bs)
        if bslen > self.seq_len:
            raise ValueError('Length of bytes ({}) is larger than sequence length ({})'.format(bslen, self.seq_len))
        indices = list(bs)
        for i, x in enumerate(indices):
            if x < 0 or x > BYTE_MAX:
                raise ValueError('{}: Bad byte value: Expected [0,{}], got {}.'.format(
                    i,
                    BYTE_MAX,
                    x
                ))
        # Pad with empty OH vectors until we reach seq_len.
        while len(indices) < self.seq_len:
            indices.append(-1)
        start = time.time()
        t = tf.one_hot(indices, depth=INPUT_SIZE)
        elapsed = time.time() - start
        log.info('Encoded bytes vector in {} seconds'.format(elapsed))
        return t

    def decode(self, tensor):
        '''
        Decode a one-hot tensor into bytes. Not implemented.
        '''
        raise NotImplementedError

default_ascii_codec = AsciiCodec(SEQ_LEN)
default_bytes_codec = BytesCodec(SEQ_LEN)
