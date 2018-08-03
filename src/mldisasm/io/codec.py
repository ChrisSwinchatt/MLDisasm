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

# Maximum value of an ASCII character.
ASCII_MAX = 0x7F

# Size of an input vector (one-hot).
INPUT_SIZE = BYTE_MAX + 1

# Size of a target vector (one-hot).
TARGET_SIZE = ASCII_MAX + 1

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

def _tokenize(string):
    '''
    Tokenise an assembly string. Tokens are: punctuation, digits (such that the number 100 has three tokens, 1, 0 and 0)
    and identifiers (one or more alphabetic characters bounded by a non-alphabetic character or end of the string on
    either side).
    :returns: An ordered list of the tokens found in the string.
    '''
    tokens = []
    ident  = ''     # Identifier buffer.
    prefix = False  # Numeric prefix flag.
    for i, char in enumerate(string):
        # Skip x in hexadecimal prefix.
        if prefix and char.lower() == 'x':
            continue
        # Add alphabetic characters to ident buffer.
        if char.isalpha() and not prefix:
            ident += char
            continue
        # Set prefix flag if current char is 0 and next char is x.
        if not prefix and char == '0' and i + 1 < len(string) and string[i + 1].lower() == 'x':
            prefix = True
            tokens.append('0x')
            continue
        if ident:
            tokens.append(ident)
            ident = ''
        tokens.append(char)
        if not char.isalnum():
            prefix = False
    if ident:
        tokens.append(ident)
    return tokens

class AsciiCodec(Codec):
    '''
    Encode ASCII as token indices, or decode token indices into ASCII.
    '''
    def __init__(self, seq_len, tokens):
        self.seq_len = seq_len
        self.tokens  = tokens

    def encode(self, seq):
        '''
        Encodes the contents of an ASCII string as a vector of token indices.
        :param s: The ASCII string.
        :param checked: Set to True if the input has already been checked for validity.
        :returns: A tensor with one element per token in the string.
        '''
        # Check parameter.
        if not isinstance(seq, str):
            raise TypeError('Expected str, not {}'.format(type(seq).__name__))
        if not seq:
            raise ValueError('Received empty string')
        # Tokenise the string and convert to TokenList indices.
        tokens  = _tokenize(seq)
        indices = [[self.tokens.index(t)] for t in tokens]
        # Pad to seq_len and convert to tensor.
        while len(indices) < self.seq_len:
            indices.append([-1])
        return tf.convert_to_tensor(indices)

    def decode(self, tensor):
        '''
        Decode a tensor of token indexes into an ASCII string tensor.
        :param tensor: The encoded tensor.
        :returns: An ASCII string tensor.
        '''
        # Check parameters.
        ndim = len(tensor.shape)
        if ndim > 1:
            # Recursively map over nD tensor until n=1.
            return tf.map_fn(self.decode, tensor)
        if not isinstance(tensor, tf.Tensor):
            raise TypeError('Expected Tensor, not {}'.format(type(tensor).__name__))
        t = tf.map_fn(
            lambda idx: tf.cond(
                idx >= 0,
                true_fn  = lambda: self.tokens.as_tensor[idx],
                false_fn = lambda: tf.convert_to_tensor('')
            ),
            tensor
        )
        return tf.reduce_join(tf.as_string(t))

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
        xs = [[float(byte)/BYTE_MAX] for byte in bs]
        while len(xs) < self.seq_len:
            xs.append([np.inf])
        return tf.convert_to_tensor(xs)

    def decode(self, tensor):
        '''
        Decode a one-hot tensor into bytes. Not implemented.
        '''
        raise NotImplementedError
