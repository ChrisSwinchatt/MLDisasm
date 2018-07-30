#!/usr/bin/env python3

'''
MLDisasm loss functions.
'''

from mldisasm.io.codec import one_hot_to_ascii

def _levenshtein(s1, s2):
    '''
    Compute 
    '''

def levenshtein_loss(pred, target):
    '''
    Compute the Levenshtein distance between a predicted, one-hot encoded string
    and a target string.
    :param pred: The one-hot encoded predicted string.
    :param target: The target string.
    '''
    return _levenshtein(one_hot_to_ascii(pred), target)
