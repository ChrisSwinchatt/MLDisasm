#!/usr/bin/env python3

'''
Test mldisasm.io.codec
'''

import sys

import pytest

import tensorflow as tf

from mldisasm.io.codec     import *
from mldisasm.tests.common import *

BAD_VALUES = (9001,3.14,'hello, world',[1,2,3])

def setup_module():
    '''
    Set up TensorFlow.
    '''
    global TF_DEVICE
    print('(Starting TensorFlow on device {})'.format(TF_DEVICE), file=sys.stderr)
    tf.enable_eager_execution()
    TF_DEVICE = tf.device(TF_DEVICE)

def test_bytes_to_one_hot():
    '''
    Test mldisasm.io.codec.bytes_to_one_hot.
    '''
    enter_test(test_bytes_to_one_hot)
    for _ in range(TEST_ITERATIONS):
        enter_test_iter()
        # Encode random bytes.
        bs     = random_bytes()
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
    leave_test()

def test_ascii_to_one_hot():
    '''
    Test mldisasm.io.codec.acii_to_one_hot.
    '''
    enter_test(test_ascii_to_one_hot)
    for _ in range(TEST_ITERATIONS):
        enter_test_iter()
        # Encode random string.
        string = random_string()
        tensor = ascii_to_one_hot(tensor)
        assert len(tensor.shape) == 2
        assert tensor.shape[0] == len(string)
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
    leave_test()

def test_one_hot_to_bytes():
    '''
    Test mldisasm.io.codec.one_hot_to_bytes.
    '''
    enter_test(test_one_hot_to_bytes)
    for _ in range(TEST_ITERATIONS):
        enter_test_iter()
        # Generate random bytes.
        bs   = random_bytes()
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
    leave_test()

def test_one_hot_to_ascii():
    '''
    Test mldisasm.io.codec.one_hot_to_ascii.
    '''
    enter_test(test_one_hot_to_ascii)
    for _ in range(TEST_ITERATIONS):
        enter_test_iter()
        # Encode random bytes.
        bs     = random_bytes()
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
    leave_test()

def test_one_hot_to_bytes():
    '''
    Test mldisasm.io.codec.one_hot_to_bytes.
    '''
    enter_test(test_one_hot_to_bytes)
    for _ in range(TEST_ITERATIONS):
        enter_test_iter()
        # Encode random bytes.
        size   = _random_size()
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
    leave_test()
