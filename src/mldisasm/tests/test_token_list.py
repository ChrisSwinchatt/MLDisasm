#!/usr/bin/env python3

'''
Test TokenList.
'''

import numpy as np

from mldisasm.io.file_manager import FileManager
from mldisasm.tests.common    import *

SEQ_LEN = 50

class TestTokenList:
    '''
    Test cases for TokenList.
    '''
    def setup(self):
        '''
        Set up tests.
        '''
        self.file_mgr = FileManager()
        self.tokens   = self.file_mgr.load_tokens()

    def test_tokenize(self):
        '''
        Test TokenList.tokenize.
        '''
        enter_test(self.test_tokenize)
        for _ in range(TEST_ITERATIONS):
            size    = random_size(max_size=SEQ_LEN)
            tokens1 = np.random.choice(self.tokens, size)
            string1 = ' '.join(tokens1).replace('  ', ' ')
            tokens2 = self.tokens.tokenize(string1)
            string2 = ' '.join(tokens2)
            assert string1 == string2
            leave_test_iter()
        leave_test()
