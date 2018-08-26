#!/usr/bin/env python3

'''
MLDisasm encoding and decoding.
'''

from abc import ABCMeta, abstractmethod

import numpy as np

import tensorflow       as tf
import tensorflow.keras as keras

import mldisasm.util.log as log

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
    Encode ASCII as one-hot vectors, or decode one-hot vectors into ASCII.
    '''
    def __init__(self, seq_len, mask_value, tokens):
        self._seq_len    = seq_len
        self._mask_value = mask_value
        self._tokens     = tokens

    def encode_lite(self, seq, as_tensor=True):
        '''
        Encode an ASCII string as a vector of token indices.
        :param seq: The ASCII string.
        :param as_tensor: Whether to encode to a tensor or return a list.
        :returns: A tensor with one index per token in the string.
        '''
        # Check parameter.
        if not isinstance(seq, str):
            raise TypeError('Expected str, not {}'.format(type(seq).__name__))
        if not seq:
            raise ValueError('Received empty string')
        # Tokenise the string and convert TokenList indices to reals between 0 and 1.
        tokens  = self._tokens.tokenize(seq)
        indices = [[self._tokens.index(t)] for t in tokens]
        if as_tensor:
            return tf.convert_to_tensor(indices)
        # Pad to seq_len.
        if len(indices) > self._seq_len:
            log.warning('Expected {} elements or fewer, got {}'.format(self._seq_len, len(indices)))
        while len(indices) < self._seq_len:
            indices.append([self._mask_value])
        return indices

    def onehotify(self, indices):
        '''
        Convert a vector of token indices to a one-hot matrix.
        :param indices: A list or ndarray containing the indices. Negative indices are interpreted as masked values and
        will produce vectors whose elements are all zeros.
        :returns: A one-hot encoded matrix (ndarray).
        '''
        shape = np.shape(indices)
        # For legacy reasons the indices shape is (batch_size,seq_len,1). To handle arrays with 1, 2 and 3 dimensions
        # transparently, we map over dim 0 and squeeze out dim 2, so that the indices we actually process has shape
        # (seq_len,).
        if len(shape) == 3:
            return np.asarray(list(map(self.onehotify, indices)))
        if len(shape) == 2:
            indices = np.squeeze(indices)
        onehot = keras.utils.to_categorical(indices, len(self._tokens))
        # Workaround: Keras interparamsets negative indices as offsets from the end of the vector, so that -1 in a
        # 4-class vector would produce [0,0,0,1]. We want to interpret negative values as being masked/invalid, so we
        # patch these values with 0. (tf.one_hot() produces zeroed vectors for negative indices, but we don't want to
        # return a tensor here.)
        for i, idx in enumerate(indices):
            if idx < 0:
                j = len(onehot[i]) + idx
                onehot[i,j] = 0
        return onehot

    def encode(self, seq, as_tensor=True):
        '''
        Encodes the contents of an ASCII string as a one-hot matrix.
        :param seq: The ASCII string.
        :param as_tensor: Whether to encode as a tensor or a list.
        :returns: A one-hot encoded matrix representing the ASCII string.
        '''
        indices = self.encode_lite(seq, as_tensor=False)
        onehot  = self.onehotify(indices)
        if as_tensor:
            onehot = tf.convert_to_tensor(onehot)
        return onehot

    def decode(self, tensor):
        '''
        Decode a tensor of token indices into an ASCII string tensor.
        :param tensor: A 3D tensor or NumPy array of shape (batch_size,seq_len,1) containing the token indices.
        :returns: A list of strings, one per sample in the tensor.
        '''
        # Check type and evaluate into NumPy array if needed.
        if isinstance(tensor, tf.Tensor):
            tensor = tensor.eval()
        elif not isinstance(tensor, np.ndarray):
            raise TypeError('Expected Tensor or ndarray, not {}'.format(type(tensor).__name__))
        # Check dimensions. Fail if we don't have two or more. Map over tensor if we have more than two.
        ndims = len(tensor.shape)
        if ndims > 2:
            return list(map(self.decode, tensor))
        if ndims < 2:
            raise ValueError('Expected at least two dimensions, got {}'.format(ndims))
        # Convert one-hot vectors into tokens. The vectors returned by the model will contain probabilities; we consider
        # the "hot" element to be the one with the largest value (argmax).
        tokens = [None]*len(tensor)
        for i in range(len(tensor)):
            idx       = np.argmax(tensor[i])
            tokens[i] = self._tokens[idx]
        # Join tokens into string and remove trailing whitespace.
        return ' '.join(tokens).rstrip()

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
        xs = map(
            lambda x: int(x[0]*BYTE_MAX),
            filter(
                lambda x: x >= 0,
                tensor.eval()
            )
        )
        return bytes(xs)
