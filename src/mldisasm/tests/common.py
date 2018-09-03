#!/usr/bin/env python3

'''
MLDisasm testing framework.
'''

from   abc import ABCMeta
import sys
import os

# Filter out debug messages from TF C++ library.
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '1'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'

import tensorflow as tf

# Filter out debug messages from TF Python API.
tf.logging.set_verbosity(tf.logging.INFO)

import numpy as np

from mldisasm.constants       import ASCII_MAX, BYTE_MAX, BYTEORDER
from mldisasm.io.file_manager import FileManager

TEST_ITERATIONS   = 100      # How many test iterations to run.
MIN_STRING_LENGTH = 2        # Minimum string length.
MAX_STRING_LENGTH = 512      # Maximum string length.
TF_DEVICE         = '/cpu:0' # TensorFlow device name. We use the CPU for testing.
MODEL_NAME        = 'att'    # Model name to test.
SEQ_LEN           = 50       # Sequence length.

class GenericTestCase:
    '''
    Base class for tests.
    '''
    __metaclass__ = ABCMeta

    def setup(self):
        '''
        Setup.
        '''
        # Disable checking for "attribute ... defined outside __init__" because PyTest test cases can't have __init__.
        # pylint: disable=W0201
        # Set device and session.
        self.device   = tf.device(TF_DEVICE)
        self.session  = tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) # Force TF to use the CPU.
        # Load FileManager and config.
        self.file_mgr = FileManager()
        self.config   = self.file_mgr.load_config(MODEL_NAME)
        # Set device and session as default.
        self.device.__enter__()
        self.session.__enter__()

    def teardown(self):
        '''
        Teardown.
        '''
        self.device.__exit__(None,  None, None)
        self.session.__exit__(None, None, None)
        self.session.close()

    def itertest(self, test, func, args=None):
        '''
        Run a test over TEST_ITERATIONS iterations.
        '''
        if args is None:
            args = tuple()
        enter_test(test)
        for _ in range(TEST_ITERATIONS):
            enter_test_iter()
            func(*args)
            leave_test_iter()
        leave_test()

def enter_test(func):
    '''
    Function called when a test begins.
    '''
    sys.stdout.write('\n{}: '.format(func.__name__))
    sys.stdout.flush()

def enter_test_iter():
    '''
    Function called when a test iteration begins.
    '''
    sys.stdout.write('-')
    sys.stdout.flush()

def leave_test_iter():
    '''
    Function called when a test iteration ends.
    '''
    sys.stdout.write('\b+')
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
        lambda x: int(x).to_bytes(1, BYTEORDER), # Byteorder doesn't matter here as each value is a single byte.
        np.random.randint(0, BYTE_MAX, random_size(max_size, min_size))
    ))

def random_string(max_size=MAX_STRING_LENGTH, min_size=MIN_STRING_LENGTH):
    '''
    Generate a random string up to MAX_STRING_LENGTH chars long.
    '''
    size = random_size(max_size, min_size)
    return ''.join(map(chr, np.random.randint(ASCII_MAX, size=size)))

def random_dict(max_size=SEQ_LEN, min_size=MIN_STRING_LENGTH + 1):
    '''
    Generate a dictionary with random keys and values which are lists.
    '''
    size = random_size(max_size, min_size)
    return dict([(random_string(size),np.random.random(size).tolist()) for _ in range(size)])
