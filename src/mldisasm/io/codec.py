#!/usr/bin/env python3

'''
MLDisasm encoding and decoding.
'''

from abc import ABCMeta, abstractmethod

import numpy as np

import tensorflow       as tf
import tensorflow.keras as keras

from mldisasm.constants import ASCII_MAX, BYTE_MAX, BYTEORDER
from mldisasm.util      import log

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

START_SEQ = '\t'
STOP_SEQ  = '\n'

class AsciiCodec(Codec):
    '''
    Encode ASCII as one-hot vectors, or decode one-hot vectors into ASCII.
    '''
    def __init__(self, seq_len, mask_value):
        self._seq_len    = seq_len
        self._mask_value = mask_value

    def encode(self, seq, as_tensor=True):
        '''
        Encodes the contents of an ASCII string as a one-hot matrix.
        :param seq: The ASCII string.
        :param as_tensor: Whether to encode as a tensor or a list.
        :returns: A one-hot encoded matrix representing the ASCII string.
        '''
        # Create indices with the start and end tokens added.
        indices = list(map(ord, seq))
        indices.insert(0, ord(START_SEQ))
        indices.append(ord(STOP_SEQ))
        # Convert to onehot and pad to seq_len.
        onehot = list(keras.utils.to_categorical(indices, num_classes=ASCII_MAX + 1))
        while len(onehot) < self._seq_len:
            onehot.append([0]*len(onehot[0]))
        onehot = np.asarray(onehot, dtype=np.int32)
        # Convert to tensor and/or return.
        if as_tensor:
            return tf.convert_to_tensor(onehot, dtype=tf.int32)
        return onehot

    def decode(self, onehot):
        '''
        Decode a tensor of token indices into an ASCII string tensor.
        :param onehot: A list of one-hot vectors encoding ASCII characters.
        :returns: The decoded string if onehot is a 2D matrix, or a list of such strings if onehot is 3D.
        '''
        # Check type and convert into NumPy array if needed.
        if isinstance(onehot, tf.Tensor):
            onehot = onehot.eval()
        elif isinstance(onehot, list):
            onehot = np.asarray(onehot)
        elif not isinstance(onehot, np.ndarray):
            raise TypeError('Expected Tensor or ndarray, not {}'.format(type(onehot).__name__))
        # Fail if dimensionality is less than 2.
        if len(onehot.shape) < 2:
            raise ValueError('Expected at least two dimensions, got {}'.format(len(onehot.shape)))
        # Map if dimensionality is greater than 2.
        if len(onehot.shape) > 2:
            return list(map(self.decode, onehot))
        # Decode into a string, filtering out empty one-hot vectors.
        return ''.join(map(lambda oh: chr(np.argmax(oh)), onehot))

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
        Encode a bytes to one-hot.
        :param bs: Either a bytes object or a string of hex-encoded bytes.
        :param as_tensor: Whether to return a tensor (True) or a list (False).
        :returns: A matrix of the one-hot encoded bytes.
        '''
        if isinstance(bs, str):
            # Convert to bytes from hex string.
            bs = int(bs, 16).to_bytes(
                len(bs)//2, # Every two chars in hexadecimal is one byte.
                BYTEORDER
            )
        if not isinstance(bs, bytes):
            raise TypeError('Expected bytes, not {}'.format(type(bs).__name__))
        # Convert bytes into integers.
        indices = list(bs)
        if len(indices) > self._seq_len:
            log.warning('Expected {} elements or fewer, got {}'.format(self._seq_len, len(indices)))
        # Encode as one-hot and pad to seq_len.
        onehot = list(keras.utils.to_categorical(indices, num_classes=BYTE_MAX + 1))
        while len(onehot) < self._seq_len:
            onehot.append([0]*len(onehot[0]))
        onehot = np.asarray(onehot, dtype=np.int32)
        if as_tensor:
            return tf.convert_to_tensor(onehot, dtype=tf.int32)
        return onehot

    def decode(self, tensor):
        '''
        Decode a one-hot tensor into a bytes object.
        '''
        if not isinstance(tensor, tf.Tensor):
            raise TypeError('Expected Tensor, not {}'.format(type(tensor).__name__))
        xs = map(
            lambda x: int(x[0]*BYTE_MAX),
            filter(
                lambda x: x >= 0,
                tensor.eval()
            )
        )
        return bytes(xs)
