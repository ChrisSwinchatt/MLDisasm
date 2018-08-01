#!/usr/bin/env python3

'''
Test mldisasm.training.loss
'''

import numpy as np

from mldisasm.tests.common  import *
from mldisasm.training.loss import _levenshtein

TEST_ITERATIONS   = 300
ASCII_MAX         = 0x7F

def random_index(max_idx, not_in=None):
    '''
    Generate a random index.
    :param max_idx: The maximum index.
    :param not_in: None, or a list of indexes that should not be generated.
    '''
    if not_in is None:
        not_in = []
    while True:
        idx = np.random.randint(max_idx)
        if not not_in or idx not in not_in:
            return idx

def random_char(not_in=None):
    '''
    Generate a random ASCII character as an int.
    :param not_in: None, or a list of characters not to generate.
    '''
    while True:
        c = np.random.randint(ASCII_MAX)
        if not not_in or c not in not_in:
            return c

def random_subs(s1):
    '''
    Generate a random permutation of a string by substitution.
    :returns: A tuple of the permutated string and the number of substitutions made (Hamming distance).
    '''
    s1_len      = len(s1)
    num_changes = np.random.randint(1, int(s1_len/2))
    s2          = list(map(ord, s1))
    subs        = []
    for _ in range(num_changes):
        idx   = random_index(s1_len, not_in=subs)
        # Don't generate characters that are the same as adjacent chars, otherwise LD detects them as transpositions.
        # (Which is good, but will return a different result from num_changes.)
        chars = [s2[idx]]
        if idx > 0:
            chars.append(s2[idx - 1])
        if idx + 1 < s1_len:
            chars.append(s2[idx + 1])
        s2[idx] = random_char(not_in=chars)
        subs.append(idx)
    return ''.join(map(chr, s2)), num_changes

def random_adds(s1):
    '''
    Generate a random permutation of a string by addition.
    :returns: A tuple of the permutated string and the number of additions made.
    '''
    s1_len = len(s1)
    s2_len = np.random.randint(s1_len + 1, MAX_STRING_LENGTH + 1)
    delta  = s2_len - s1_len
    s2     = list(map(ord, s1))
    for _ in range(s1_len, s2_len):
        idx = np.random.randint(0, s2_len)
        if idx >= s1_len:
            s2.append(random_char())
        else:
            s2.insert(idx, random_char())
    return ''.join(map(chr, s2)), delta

def random_delete(s1):
    '''
    Generate a random permutation of a string by deletion.
    :returns: A tuple of the permutated string and the number of deletions made.
    '''
    s1_len = len(s1)
    s2_len = np.random.randint(1, s1_len)
    delta  = s1_len - s2_len
    s2     = list(map(ord, s1))
    while len(s2) > s2_len:
        idx = np.random.randint(0, len(s2))
        del s2[idx]
    return ''.join(map(chr, s2)), delta

def test_levenshtein():
    '''
    Test the Levenshtein distance implementation.
    '''
    s1 = random_string()
    enter_test(test_levenshtein)
    for _ in range(TEST_ITERATIONS):
        enter_test_iter()
        # Test substitution.
        s2, delta1_2 = random_subs(s1)
        assert delta1_2 == _levenshtein(s1, s2)
        # Test addition.
        s3, delta2_3 = random_adds(s2)
        assert delta2_3 == _levenshtein(s2, s3)
        # Test deletion.
        s4, delta3_4 = random_delete(s3)
        assert delta3_4 == _levenshtein(s3, s4)
    leave_test()
