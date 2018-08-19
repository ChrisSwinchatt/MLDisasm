#!/usr/bin/env python3

'''Usage: {0} <model>
'''

import os
import sys
import traceback as tb

# Filter out debug messages from TF.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == '__main__':
    print('*** Starting up...')

import tensorflow               as tf
import tensorflow.keras         as keras
import tensorflow.keras.backend as K

from   mldisasm.benchmarks.profiling import prof
import mldisasm.io.log               as     log
from   mldisasm.io.file_manager      import FileManager
from   mldisasm.model                import make_disassembler

def train_model(config, name, file_mgr):
    '''
    Train a model.
    '''
    params = config['model']
    log.info('Training model with parameters {}'.format(params))
    K.set_learning_phase(1)
    model     = make_disassembler(**params)
    loss      = 0
    callbacks = []
    if params.get('stop_early', False):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor  = 'loss',
            patience = params.get('patience', 0)
        ))
    batch_num = 0
    for epoch in range(params['epochs']):
        log.info('Epoch {}/{}'.format(epoch + 1, params['epochs']))
        batch_num = 0
        for X, y in file_mgr.yield_training(name, config['batch_size']):
            batch_num += 1
            with prof('Trained batch {}', lambda: batch_num, log_level='info'):
                with tf.device('/cpu:0'):
                    X = tf.Variable(tf.stack(X))
                    y  = tf.Variable(tf.stack(y))
                history = model.fit(
                    X,
                    y,
                    steps_per_epoch = 1,
                    epochs          = 1,
                    verbose         = 0
                )
                loss = min(history.history['loss'])
    return model, loss

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
    try:
        # Load configuration,
        config = file_mgr.load_config()
        # Train model on whole dataset.
        model, _ = train_model(config, model_name, file_mgr)
        file_mgr.save_model(model, model_name)
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error('Uncaught exception \'{}\': {}'.format(type(e).__name__, str(e).split('\n')[0]))
        log.error('See the log at {} for details.'.format(file_mgr.log_file_path))
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        exit(1)
