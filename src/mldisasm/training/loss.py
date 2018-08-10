#!/usr/bin/env python3

'''
MLDisasm loss functions.
'''

import tensorflow             as     tf
import tensorflow.keras       as     keras
from   tensorflow.keras.utils import to_categorical

from   mldisasm.benchmarks.profiling import prof
import mldisasm.io.log as log

def levenshtein_loss(decoder, target, pred):
    '''
    Compute the Levenshtein distance between a predicted, one-hot encoded string
    and a target string.
    :param decoder: The decoder.
    :param target: The encoded target string(s).
    :param pred: The encoded predicted string(s).
    :returns: The Levenshtein Distance between the two strings.
    '''
    assert decoder is not None
    shape = target.shape
    pred  = tf.cast(tf.round(tf.reshape(pred, shape)), tf.int32)
    if shape.ndims < 2:
        raise ValueError('Expected two or more dimensions, got {}'.format(shape.ndims))
    # Decode into dense string tensors.
    pred_dense   = decoder.decode(pred)
    target_dense = decoder.decode(target)
    # Decompose into sparse tensors and compute LD.
    pred_sparse   = tf.string_split(pred_dense,   '')
    target_sparse = tf.string_split(target_dense, '')
    t = tf.edit_distance(pred_sparse, target_sparse)
    t.set_shape(shape[0])
    return t


# Known loss functions, indexed by name.
LOSS_FUNCTIONS = {
    'levenshtein':  levenshtein_loss
}
