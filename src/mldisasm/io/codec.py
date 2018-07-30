#!/usr/bin/env python3

'''
MLDisasm encoding and decoding.
'''

import tensorflow as tf

# Maximum value of a byte.
BYTE_MAX  = 0xFF

# Maximum value of an ASCII character.
ASCII_MAX = 0x7F

# Size of an input vector (one-hot).
INPUT_SIZE = BYTE_MAX + 1

# Size of a target vector (one-hot).
TARGET_SIZE = ASCII_MAX + 1

def bytes_to_one_hot(bs):
    '''
    One-hot encodes the contents of a bytes object.
    :param bs: The bytes object.
    :returns: A tensor with one row per byte in the input object, and INPUT_SIZE (256) elements per row. One element
    per row will have the value 1 and the others will be zero.
    '''
    if not isinstance(bs, bytes):
        raise TypeError('Expected bytes, not {}'.format(type(bs).__name__))
    indices = list(bs)
    return tf.one_hot(indices, depth=INPUT_SIZE)

def ascii_to_one_hot(s):
    '''
    One-hot encodes the contents of an ASCII string.
    :param s: The ASCII string.
    :returns: A tensor with one row per character in the input string, and TARGET_SIZE (128) elements per row. One
    element per row will have the value 1 and the others will be zero.
    '''
    if not isinstance(s, str):
        raise TypeError('Expected str, not {}'.format(type(s).__name__))
    indices = [ord(c) for c in s]
    return tf.one_hot(indices, depth=TARGET_SIZE)

def one_hot_to_bytes(tensor):
    '''
    Decode a one-hot encoded tensor into a bytes object.
    '''
    if not isinstance(tensor, tf.Tensor):
        raise TypeError('Expected Tensor, not {}'.format(type(tensor).__name__))
    if len(tensor.shape) != 2:
        raise ValueError('Expected 2D tensor, not {}D'.format(len(tensor.shape)))
    if tensor.shape[1] != TARGET_SIZE:
        raise ValueError('Expected size of dim 2 to be {}, not {}'.format(INPUT_SIZE, tensor.shape[1]))
    return bytes([row.find(max(row)) for row in tensor])

def one_hot_to_ascii(tensor):
    '''
    Decode a one-hot encoded tensor into an ASCII string.
    :param tensor: The one-hot encoded tensor.
    '''
    if not isinstance(tensor, tf.Tensor):
        raise TypeError('Expected Tensor, not {}'.format(type(tensor).__name__))
    if len(tensor.shape) != 2:
        raise ValueError('Expected 2D tensor, not {}D'.format(len(tensor.shape)))
    if tensor.shape[1] != TARGET_SIZE:
        raise ValueError('Expected size of dim 2 to be {}, not {}'.format(TARGET_SIZE, tensor.shape[1]))
    # Choose the most confident value in each row of the tensor and convert its index into a char according to its ASCII
    # value.
    return ''.join([chr(row.find(max(row))) for row in tensor])
