#!/usr/bin/env python3

'''
MLDisasm loss functions.
'''

import numpy as np

from mldisasm.io.codec import one_hot_to_ascii

def _levenshtein(s1, s2):
    '''
    Compute the Levenshstein distance between two strings.
    '''
    # Check for empty strings. If either string is empty, the LD is the length of the other string.
    s1_len = len(s1)
    s2_len = len(s2)
    if s1_len == 0:
        return s2_len
    if s2_len == 0:
        return s1_len
    if s1_len > s2_len:
        s3     = s2
        s3_len = s2_len
        s2     = s1
        s2_len = s1_len
        s1     = s3
        s1_len = s3_len
    # Build matrix.
    matrix = np.zeros((s1_len + 1, s2_len + 1), dtype=int)
    for i in range(s1_len + 1):
        matrix[i, 0] = i
        if i == 0:
            continue
        for j in range(s2_len + 1):
            matrix[0, j] = j
            if j == 0:
                continue
            if s1[i - 1] == s2[j - 1]:
                matrix[i, j] = matrix[i - 1, j - 1]
            else:
                matrix[i, j] = min(
                    matrix[i,     j - 1] + 1,
                    matrix[i - 1, j]     + 1,
                    matrix[i - 1, j - 1] + 1
                )
    # Last values in the matrix are the compute distance.
    return matrix[-1, -1]

def levenshtein_loss(pred, target):
    '''
    Compute the Levenshtein distance between a predicted, one-hot encoded string
    and a target string.
    :param pred: The one-hot encoded predicted string.
    :param target: The one-hot encoded target string.
    '''
    return _levenshtein(one_hot_to_ascii(pred), one_hot_to_ascii(target))
