#!/usr/bin/env python3

'''
MLDisasm token list.
'''

import tensorflow as tf

from mldisasm.util import prof, log

class TokenList:
    '''
    Token list for embedding.
    '''
    def __init__(self, file, *args, **kwargs):
        '''
        Initialise TokenList.
        :param file: A filename or a handle to a readable file.
        '''
        with prof('Read {} unique tokens', lambda: len(tokens), resources=['time','memory']):
            tokens = set()
            if isinstance(file, str):
                file = open(file, 'r')
            if not hasattr(file, 'readable') or not file.readable():
                raise TypeError('Expected string or readable file for arg 0, not {}'.format(type(file).__name__))
            # Read the tokens.
            for line in file:
                tokens.add(line[:-1]) # Remove trailing newline.
            # Convert to sorted list and append empty token for masked values.
            self._tokens = sorted(tokens)
            self._tokens.append('')
            self._tensor = None

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

    def tokenize(self, string):
        '''
        Tokenise a string.
        :param string: The string to tokenise.
        :returns: A list of the tokens in order of appearance.
        '''
        tokens = []
        # Find tokens by longest match. This breaks when valid tokens pasted together form another valid token, e.g.
        # "add" and "subps" form "addsubps", all of which are legal tokens. This would only be a problem if the network
        # generates two instructions in a row, which should be taken care of during training. This also doesn't handle
        # the space token correctly.
        i = len(string)
        while i >= 0:
            j = 0
            while j < i:
                if string[j:i] in self._tokens:
                    tokens.append(string[j:i])
                    i = j
                    break
                j += 1
            i -= 1
        return reversed(tokens)

    def to_list(self):
        '''
        Return a list of the tokens. The returned list is newly allocated.
        '''
        return list(self._tokens)

    @property
    def as_tensor(self):
        '''
        Get the tokens list as a tensor.
        '''
        # We only create the tensor the first time it's demanded because preprocessor needs token_list to be
        # serialisable for multiprocessing, but we don't want to recreate the tensor every time we use it.
        if self._tensor is None:
            self._tensor = tf.convert_to_tensor(self._tokens, dtype=tf.string)
        return self._tensor

    def __len__(self):
        '''
        Get the number of tokens in the list.
        '''
        return len(self._tokens)

    def __iter__(self):
        return TokenList._Iterator(self)

    def __getitem__(self, index):
        '''
        Retrieve an item by index.
        '''
        return self._tokens[index]

    class _Iterator:
        def __init__(self, tokens):
            self._tokens = tokens
            self._index  = 0

        def __len__(self):
            return len(self._tokens)

        def __iter__(self):
            return self

        def __next__(self):
            if self._index >= len(self._tokens):
                raise StopIteration
            t = self._tokens[self._index]
            self._index += 1
            return t
