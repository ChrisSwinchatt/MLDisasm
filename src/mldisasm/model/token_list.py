#!/usr/bin/env python3

'''
MLDisasm token list.
'''

import tensorflow as tf

from   mldisasm.benchmarks.profiling import prof
import mldisasm.io.log               as     log

class TokenList:
    '''
    Token list for embedding.
    '''
    def __init__(self, file, *args, **kwargs):
        '''
        Initialise TokenList.
        :param file: A filename or a handle to a readable file.
        '''
        p = prof('Loaded tokens')
        if isinstance(file, str):
            file = open(file, 'r')
        if not hasattr(file, 'readable') or not file.readable():
            raise TypeError('Expected string or readable file for arg 0, not {}'.format(type(file).__name__))
        # Read the tokens.
        tokens = set()
        for line in file:
            tokens.add(line[:-1]) # Cut off the newline.
        log.debug('Read {} unique tokens'.format(len(tokens)))
        self._tokens = sorted(tokens)
        # Convert to tensor. We store the tensor separately which wastes a little memory, but not much.
        self.as_tensor = tf.convert_to_tensor(self._tokens)

    def __len__(self):
        '''
        Get the number of tokens in the list.
        '''
        return len(self._tokens)

    def index(self, token):
        '''
        Get the index of a token.
        :param token: The token to find.
        :returns: The index of the token.
        :raises ValueError: If the token is not found.
        '''
        try:
            return self._tokens.index(token)
        except ValueError as e:
            log.debug(self._tokens)
            raise e from None

    def __getitem__(self, index):
        '''
        Retrieve an item by index.
        '''
        return self._tokens[index]
