#!/usr/bin/env python3

'''
MLDisasm loss functions.
'''

import numpy as np

import tensorflow as tf

import mldisasm.io.log   as     log
from   mldisasm.io.codec import default_ascii_codec

def _levenshtein(s1, s2):
    '''
    Compute the Levenshstein distance between two string tensors.
    '''
    # Check for empty strings. If either string is empty, the LD is the length of the other string.
    s1_len = s1.shape[0]
    s2_len = s2.shape[0]
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

def levenshtein_loss(target, pred):
    '''
    Compute the Levenshtein distance between a predicted, one-hot encoded string
    and a target string.
    :param target: The one-hot encoded target string(s).
    :param pred: The one-hot encoded predicted string(s).
    :returns: The Levenshtein Distance between the two strings.
    '''
    shape = target.shape
    ndim  = len(shape)
    pred  = tf.reshape(pred, shape)
    if ndim < 2:
        raise ValueError('Expected two or more dimensions, got {}'.format(ndim))
    if ndim > 2:
        return tf.map_fn(
            lambda x: _levenshtein(default_ascii_codec.decode(x[0]), default_ascii_codec.decode(x[0])),
            tf.stack([pred, target], axis=1)
        )
    return _levenshtein(default_ascii_codec.decode(pred), default_ascii_codec.decode(target))

# Known loss functions, indexed by name.
LOSS_FUNCTIONS = {
    'levenshtein': levenshtein_loss
}
