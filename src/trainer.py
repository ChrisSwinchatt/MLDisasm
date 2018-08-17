#!/usr/bin/env python3

'''Usage: {0} <model>
'''

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

import mldisasm.benchmarks.profiling as     profiling
from   mldisasm.benchmarks.profiling import prof
import mldisasm.io.log               as     log
from   mldisasm.io.file_manager      import FileManager
from   mldisasm.model                import make_disassembler

def train_model(params, tset):
    '''
    Train a model.
    '''
    log.info('Training model with parameters {}'.format(params))
    model     = make_disassembler(**params)
    loss      = 0
    callbacks = []
    if params.get('stop_early', False):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=params.get('patience', 0)
        ))
    for epoch in range(params['epochs']):
        log.info('Epoch {}/{}'.format(epoch, params['epochs']))
        for X, y in tset:
            history = model.fit(
                X,
                y,
                steps_per_epoch=1,
                epochs=1,
                verbose=0
            )
            loss = min(history.history['loss'])
    return model, loss

def start_training(model_name, file_mgr):
    '''
    Start training process.
    '''
    # Load configuration and set TF device.
    config = file_mgr.load_config()
    # Load datasets, train & save model.
    K.set_learning_phase(1)
    # Train model on the whole set.
    tset     = file_mgr.open_training(model_name, batch_size=config['batch_size'], seq_len=config['seq_len'])
    model, _ = train_model(config['model'], tset)
    file_mgr.save_model(model, model_name)

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
        start_training(model_name, file_mgr)
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error('Uncaught exception \'{}\': {}'.format(type(e).__name__, str(e).split('\n')[0]))
        log.error('See the log at {} for details.'.format(file_mgr.log_file_path))
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        exit(1)
