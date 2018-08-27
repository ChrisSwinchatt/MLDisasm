#!/usr/bin/env python3

'''
MLDisasm utilities.
'''

import gc
import os
import tempfile

import tensorflow.keras         as keras
import tensorflow.keras.backend as K

from .     import log
from .prof import *

def refresh_graph(*args, model=None, build_fn=None, **kwargs):
    '''
    Refresh the TensorFlow graph. This is useful when fitting many successive models or training many epochs or batches.
    Each batch adds nodes to the graph, and TensorFlow evaluates all of these when executing the graph. This causes
    training to take increasing amounts of time, at a rate of about 3.5% (compounding) per batch: if the first batch
    takes 5 seconds, the 100th will take 150 seconds, and the 1,000th approximately 138 million years. To avoid this, we
    clear the graph at the end of each batch or epoch.
    :param model: None or a Keras model to save and reload. This is necessary when clearing the graph and continuing
    training, because the model's layers are stored within the graph and it isn't possible to delete nodes selectively.
    The model's weights will be saved to a temporary file and restored to a *new* model after the graph is cleared.
    :param build_fn: A function or constructor which returns a new model.
    :param args: Positional arguments to build_fn.
    :param kwargs: Keyword arguments to build_fn.
    :returns: If model is None, None is returned. Otherwise, new model with the same weights as model is returned.
    :raises ValueError: If model is None but build_fn is not, or vice versa.
    :raises TypeError: If model is not an instance of `keras.Model`; if build_fn is not callable.
    '''
    if [model,build_fn].count(None) == 1:
        raise ValueError('Either both \'model\' and \'build_fn\' should be None or neither of them should (got {} and {})'.format(
            None if model    is None else type(model).__name__,
            None if build_fn is None else type(build_fn).__name__
        ))
    if model is not None and not isinstance(model, keras.Model):
        raise TypeError('Expected instance of keras.Model for parameter \'model\', got {} instead'.format(
            type(model).__name__
        ))
    if build_fn is not None and not hasattr(build_fn, '__call__'):
        raise TypeError('Expected callable for parameter \'build_fn\', got {} instead'.format(
            type(build_fn).__name__
        ))
    with prof('Refreshed graph'):
        # Save the weights if model is given.
        tmp_path = None
        tmp_file = None
        if model is not None:
            tmp_file, tmp_path = tempfile.mkstemp()
            model.save_weights(tmp_path)
            del model
            model = None
            del tmp_file
        # Clear the graph and collect freed memory.
        K.clear_session()
        gc.collect()
        # Rebuild model and restore its weights.
        if tmp_path is not None:
            model = build_fn(*args, **kwargs)
            model.load_weights(tmp_path)
            try:
                os.remove(tmp_path)
            except PermissionError as e:
                log.warning('Failed to delete temporary file because "{}"'.format(
                    str(e)
                ))
        return model
