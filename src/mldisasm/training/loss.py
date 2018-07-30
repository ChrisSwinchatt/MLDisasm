#!/usr/bin/env python3

'''
MLDisasm loss functions.
'''

from mldisasm.io.codec import one_hot_to_ascii

def _levenshtein(s1, s2):
    '''
    Compute the Levenshstein distance between two strings.
    '''
    s1_len = len(s1)
    s2_len = len(s2)
    if s1_len == 0:
        return s2_len
    if s2_len == 0:
        return s1_len
    return min([
        _levenshtein(s1[:-1], s2)      + 1,
        _levenshtein(s1,      s2[:-1]) + 1,
        _levenshtein(s1[:-1], s2[:-1]) + (0 if s2[-1] == s1[-1] else 1)
    ])

def levenshtein_loss(pred, target):
    '''
    Compute the Levenshtein distance between a predicted, one-hot encoded string
    and a target string.
    :param pred: The one-hot encoded predicted string.
    :param target: The one-hot encoded target string.
    '''
    return _levenshtein(one_hot_to_ascii(pred), one_hot_to_ascii(target))
