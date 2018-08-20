#!/usr/bin/env python3

'''Usage: {0} <model>
'''

import gc
import os
import sys
import traceback as tb
import warnings

import numpy as np

# Filter out debug messages from TF.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Ignore warnings generated by using a different NumPy version.
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

if __name__ == '__main__':
    print('*** Starting up...')

import tensorflow               as tf
import tensorflow.keras         as keras
import tensorflow.keras.backend as K

from   mldisasm.benchmarks.profiling import prof
import mldisasm.io.log               as     log
from   mldisasm.io.file_manager      import FileManager
from   mldisasm.model                import make_disassembler

def parameter_grid(params):
    '''
    Generate a list of parameter sets from a set of parameters whose values are lists.
    '''
    keys, values = zip(*sorted(params.items()))
    sizes        = [len(v) for v in values]
    size         = np.product(sizes)
    grid         = [None]*size
    for i in range(size):
        grid[i] = dict()
        for k, vs, size in zip(keys, values, sizes):
            grid[i][k] = vs[i % size]
    return grid

def cv_split(X, y):
    '''
    Split a training set in half for cross-validation.
    '''
    X_train, X_test = tf.split(X, 2)
    y_train, y_test = tf.split(y, 2)
    return X_train, y_train, X_test, y_test

def fit_model(params, X, y):
    '''
    Fit a model to a set of parameters and return the loss during cross-validation.
    '''
    # Seed PRNG with a fixed value so each model gets the same sequence of numbers.
    np.random.seed(1)
    # Clear graph and collect memory from any previous session. Each model we fit adds thousands of nodes to the graph,
    # and TensorFlow executes the entire graph whenever tf.Session.run() is called, which results in memory allocation
    # problems and increasingly slow training when we fit successive models. This gives us a clean graph for each model.
    K.clear_session()
    gc.collect()
    # Create training set split.
    X = tf.Variable(tf.stack(X))
    y = tf.Variable(tf.stack(y))
    X_train, y_train, X_test, y_test = cv_split(X, y)
    # Append training callbacks.
    callbacks = []
    if params.get('stop_early', False):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params.get('patience', 0)
        ))
    # Train the model.
    model   = make_disassembler(**params)
    history = model.fit(
        X_train,
        y_train,
        steps_per_epoch  = 1,
        epochs           = params['epochs'],
        validation_data  = (X_test,y_test),
        validation_steps = 1,
        callbacks        = callbacks
    )
    # Return the average validation loss.
    return np.mean(history.history['val_loss'])

def select_params(config, X, y):
    '''
    Select hyperparameters by gridsearch with cross-validation.
    '''
    log.info('Selecting hyperparameters')
    grid        = parameter_grid(config['grid'])
    if len(grid) == 0:
        log.warning('No parameters to tune. Stopping.')
        exit(0)
    fit_num     = 1
    num_fits    = len(grid)
    best_params = None
    best_loss   = np.inf
    loss = 0
    for grid_params in grid:
        log.info('Fitting grid {} of {} with parameters {}'.format(fit_num, num_fits, grid_params))
        with prof('Trained grid {} with loss={}', fit_num, lambda: loss, log_level='info'):
            params = dict(config['model'])
            params.update(grid_params)
            loss = fit_model(params, X, y)
            if loss < best_loss:
                best_loss   = loss
                best_params = params
            fit_num += 1
    assert best_params is not None
    log.info('Best loss was {} with parameters {}'.format(best_loss, best_params))
    return best_params

def read_command_line():
    '''
    Get the model name from the command-line.
    '''
    if len(sys.argv) != 2:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        exit(1)
    return sys.argv[1]

if __name__ == '__main__':
    # Read command-line args.
    model_name = read_command_line()
    # Start file manager & logging.
    tf.logging.set_verbosity(tf.logging.INFO)
    file_mgr = FileManager()
    log.init(file_mgr.open_log())
    # Train a model.
    try:
        # Load configuration and set TF device.
        config = file_mgr.load_config()
        # Find and save hyperparameters.
        K.set_learning_phase(1)
        X, y   = file_mgr.load_training(model_name, max_records=config['gs_record_count'])
        params = select_params(config, X, y)
        config['model'] = params
        file_mgr.save_config(config)
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error('Uncaught exception \'{}\': {}'.format(type(e).__name__, str(e).split('\n')[0]))
        log.error('See the log at {} for details.'.format(file_mgr.log_file_path))
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        exit(1)
