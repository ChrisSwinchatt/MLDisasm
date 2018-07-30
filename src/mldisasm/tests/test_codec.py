#!/usr/bin/env python3

'''
Test mldisasm.io.codec.py
'''

import sys

import numpy as np

import pytest

import tensorflow as tf

from mldisasm.io.codec import *

TEST_ITERATIONS = 100
MAX_TENSOR_SIZE = 1000
DEVICE          = '/GPU:0'
BAD_VALUES      = (9001,3.14,'hello, world',[1,2,3])

def setup_module():
    '''
    Set up TensorFlow.
    '''
    global DEVICE
    print('Starting TensorFlow on device {}'.format(DEVICE))
    tf.enable_eager_execution()
    DEVICE = tf.device(DEVICE)

def _random_size():
    return np.random.randint(1, MAX_TENSOR_SIZE)

def _random_bytes():
    return bytes(np.random.randint(0, BYTE_MAX, _random_size()))

def _random_ascii():
    return ''.join(map(chr, np.random.randint(0, ASCII_MAX, _random_size())))

def test_bytes_to_one_hot():
    '''
    Test mldisasm.io.codec.bytes_to_one_hot.
    '''
    sys.stdout.write('test_bytes_to_one_hot: ')
    for _ in range(TEST_ITERATIONS):
        sys.stdout.write('.')
        sys.stdout.flush()
        # Encode random bytes.
        bs     = _random_bytes()
        tensor = bytes_to_one_hot(bs)
        # Check the tensor.
        assert len(tensor.shape) == 2
        assert tensor.shape[0] == len(bs)
        # Check each vector.
        for v in tensor:
            # Ensure the number of indices is correct.
            assert v.shape[0] == BYTE_MAX + 1
            # Ensure there are no values outside the range (0,BYTE_MAX).
            for x in v:
                assert x >= 0
                assert x <= BYTE_MAX
    # Check that bytes_to_one_hot rejects non-bytes argument.
    for x in BAD_VALUES:
        try:
            bytes_to_one_hot(x)
        except TypeError:
            pass
        else:
            pytest.fail('Accepted bad input')
    sys.stdout.write('\n')

def test_ascii_to_one_hot():
    '''
    Test mldisasm.io.codec.acii_to_one_hot.
    '''
    sys.stdout.write('test_ascii_to_one_hot: ')
    for _ in range(TEST_ITERATIONS):
        sys.stdout.write('.')
        sys.stdout.flush()
        # Encode random string.
        s      = _random_ascii()
        tensor = ascii_to_one_hot(s)
        # Check the tensor.
        assert len(tensor.shape) == 2
        assert tensor.shape[0] == len(s)
        # Check each vector.
        for v in tensor:
            # Ensure the number of indices is correct.
            assert v.shape[0] == BYTE_MAX + 1
            # Ensure there are no values outside the range (0,BYTE_MAX).
            for x in v:
                assert x >= 0
                assert x <= BYTE_MAX
    # Check that bytes_to_one_hot rejects non-str argument.
    for x in BAD_VALUES:
        try:
            ascii_to_one_hot(x)
        except TypeError:
            pass
        else:
            pytest.fail('Accepted bad input')
    sys.stdout.write('\n')

def test_one_hot_to_bytes():
    '''
    Test mldisasm.io.codec.one_hot_to_bytes.
    '''
    sys.stdout.write('test_one_hot_to_bytes: ')
    for _ in range(TEST_ITERATIONS):
        sys.stdout.write('.')
        sys.stdout.flush()
        # Generate random bytes.
        bs   = _random_bytes()
        # Get the encoded tensor.
        tensor = bytes_to_one_hot(bs)
        # Decode it back.
        decoded = one_hot_to_bytes(tensor)
        # Check against original.
        assert decoded == bs
    # Check that one_hot_to_bytes rejects non-bytes argument.
    for x in BAD_VALUES:
        try:
            one_hot_to_bytes(x)
        except TypeError:
            pass
        else:
            pytest.fail('Accepted bad input')
    sys.stdout.write('\n')

def test_one_hot_to_ascii():
    '''
    Test mldisasm.io.codec.one_hot_to_ascii.
    '''
    sys.stdout.write('test_one_hot_to_ascii: ')
    for _ in range(TEST_ITERATIONS):
        sys.stdout.write('.')
        sys.stdout.flush()
        # Encode random bytes.
        bs     = _random_bytes()
        tensor = bytes_to_one_hot(bs)
        # Decode it back.
        decoded = one_hot_to_bytes(tensor)
        # Check against original.
        assert decoded == bs
    # Check that one_hot_to_bytes rejects non-bytes argument.
    for x in BAD_VALUES:
        try:
            one_hot_to_bytes(x)
        except TypeError:
            pass
        else:
            pytest.fail('Accepted bad input')
    sys.stdout.write('\n')

def test_one_hot_to_bytes():
    '''
    Test mldisasm.io.codec.one_hot_to_bytes.
    '''
    sys.stdout.write('test_one_hot_to_bytes: ')
    for _ in range(TEST_ITERATIONS):
        sys.stdout.write('.')
        sys.stdout.flush()
        # Encode random bytes.
        size   = _random_size()
        s      = _random_ascii(size)
        tensor = ascii_to_one_hot(s)
        # Decode it back.
        decoded = one_hot_to_ascii(tensor)
        # Check against original.
        assert decoded == bs
    # Check that one_hot_to_bytes rejects non-bytes argument.
    for x in BAD_VALUES:
        try:
            one_hot_to_ascii(x)
        except TypeError:
            pass
        else:
            pytest.fail('Accepted bad input')
    sys.stdout.write('\n')
