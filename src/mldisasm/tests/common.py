#!/usr/bin/env python3

'''
MLDisasm testing framework.
'''

import sys

import numpy as np

TEST_ITERATIONS   = 100    # How many test iterations to run.
MIN_STRING_LENGTH = 2      # Minimum string length.
MAX_STRING_LENGTH = 512    # Maximum string length.
BYTE_MAX          = 0xFF   # Maximum value of a byte.
ASCII_MAX         = 0x7F   # Maximum value of an ASCII character.
TF_DEVICE       = '/device:SYCL:0' # TensorFlow device name.

def enter_test(func):
    '''
    Function called when a test begins.
    '''
    sys.stdout.write('\n{}: '.format(func.__name__))
    sys.stdout.flush()

def enter_test_iter():
    '''
    Function called when a test begins an iteration.
    '''
    sys.stdout.write('.')
    sys.stdout.flush()

def leave_test():
    '''
    Function called when a test finishes.
    '''
    sys.stdout.write('\n')
    sys.stdout.flush()

def random_size():
    '''
    Generate a random integer from MIN_STRING_LENGTH to MAX_STRING_LENGTH.
    '''
    return np.random.randint(MIN_STRING_LENGTH, MAX_STRING_LENGTH)

def random_bytes():
    '''
    Generate a random bytes object up to MAX_STRING_LENGTH bytes long.
    '''
    return bytes(np.random.randint(0, BYTE_MAX, random_size()))


def random_string():
    '''
    Generate a random string up to MAX_STRING_LENGTH chars long.
    '''
    size = random_size()
    return ''.join(map(chr, np.random.randint(ASCII_MAX, size=size)))
