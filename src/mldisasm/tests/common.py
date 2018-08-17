#!/usr/bin/env python3

'''
MLDisasm testing framework.
'''

import sys

import numpy as np

TEST_ITERATIONS   = 100      # How many test iterations to run.
MIN_STRING_LENGTH = 2        # Minimum string length.
MAX_STRING_LENGTH = 512      # Maximum string length.
BYTE_MAX          = 0xFF     # Maximum value of a byte.
ASCII_MAX         = 0x7F     # Maximum value of an ASCII character.
TF_DEVICE         = '/gpu:0' # TensorFlow device name.
TF_SESSION        = None     #

def enter_test(func):
    '''
    Function called when a test begins.
    '''
    sys.stdout.write('\n{}: '.format(func.__name__))
    sys.stdout.flush()

def leave_test_iter():
    '''
    Function called when a test iteration ends.
    '''
    sys.stdout.write('.')
    sys.stdout.flush()

def leave_test():
    '''
    Function called when a test finishes.
    '''
    sys.stdout.write('\n')
    sys.stdout.flush()

def random_size(max_size=MAX_STRING_LENGTH, min_size=MIN_STRING_LENGTH):
    '''
    Generate a random integer from MIN_STRING_LENGTH to MAX_STRING_LENGTH.
    '''
    return np.random.randint(min_size, max_size)

def random_bytes(max_size=MAX_STRING_LENGTH, min_size=MIN_STRING_LENGTH):
    '''
    Generate a random bytes object up to MAX_STRING_LENGTH bytes long.
    '''
    return b''.join(map(
        lambda x: int(x).to_bytes(1, 'little'), # Byteorder doesn't matter here as each value is a single byte.
        np.random.randint(0, BYTE_MAX, random_size(max_size, min_size))
    ))

def random_string(max_size=MAX_STRING_LENGTH, min_size=MIN_STRING_LENGTH):
    '''
    Generate a random string up to MAX_STRING_LENGTH chars long.
    '''
    size = random_size(max_size, min_size)
    return ''.join(map(chr, np.random.randint(ASCII_MAX, size=size)))
