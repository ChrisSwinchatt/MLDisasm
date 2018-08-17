#!/usr/bin/env python3

'''
Test mldisasm.io.codec
'''

import numpy as np

import tensorflow as tf

from mldisasm.io.codec        import AsciiCodec, BytesCodec
from mldisasm.io.file_manager import FileManager
from mldisasm.tests.common    import *

# Model name to test.
MODEL_NAME = 'att'

# Sequence length.
SEQ_LEN = 100

# File manager.
FILE_MGR = None

# Token list.
TOKENS = None

def setup_module():
    '''Setup.'''
    global TF_DEVICE, TF_SESSION, FILE_MGR, TOKENS
    # Set device and session.
    TF_DEVICE  = tf.device(TF_DEVICE).__enter__()
    TF_SESSION = tf.Session().__enter__()
    FILE_MGR   = FileManager()
    TOKENS     = FILE_MGR.load_tokens(MODEL_NAME)

def teardown_module():
    '''Teardown.'''
    TF_SESSION.close()

def random_tokens():
    '''Generate random tokens'''
    count = np.random.randint(1, SEQ_LEN)
    return ' '.join(np.random.choice(TOKENS, count))

def test_ascii_codec():
    ''' Test AsciiCodec. '''
    enter_test(test_ascii_codec)
    codec = AsciiCodec(SEQ_LEN, TOKENS)
    for _ in range(TEST_ITERATIONS):
        string  = random_tokens()
        encoded = codec.encode(string)
        decoded = codec.decode(encoded).eval().decode()
        assert string == decoded
        leave_test_iter()
    leave_test()

def test_bytes_codec():
    '''Test BytesCodec'''
    enter_test(test_bytes_codec)
    codec = BytesCodec(SEQ_LEN)
    for _ in range(TEST_ITERATIONS):
        bs      = random_bytes(SEQ_LEN)
        encoded = codec.encode(bs)
        decoded = codec.decode(encoded)
        assert bs == decoded
        leave_test_iter()
    leave_test()
