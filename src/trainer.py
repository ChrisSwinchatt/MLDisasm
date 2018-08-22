#!/usr/bin/env python3

'''Usage: {0} <model>
'''

import gc
import os
import sys
import traceback as tb

if __name__ == '__main__':
    print('*** Starting up...')
    # Filter out debug messages from TF.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow               as tf
import tensorflow.keras.backend as K

from   mldisasm.benchmarks.profiling import prof
import mldisasm.io.log               as     log
from   mldisasm.io.file_manager      import FileManager
from   mldisasm.model                import make_disassembler

def train_batch(model, X, y, epoch, num_epochs, batch_num, max_batches):
    '''
    Train a single batch.
    '''
    loss = 0
    with prof(
        'Epoch {}/{}: Trained batch {}/{} with loss={}',
        epoch,
        num_epochs,
        batch_num,
        max_batches,
        lambda: loss,
        log_level='info'
    ):
        with tf.device('/cpu:0'):
            X = tf.stack(X)
            y = tf.stack(y)
        history = model.fit(
            X,
            y,
            steps_per_epoch = 1,
            epochs          = 1,
            verbose         = 0
        )
        loss = history.history['loss'][-1]
    # Exit the context before returning loss so prof can print the loss.
    return loss

def train_epoch(file_mgr, config, model, name, epoch):
    '''
    Train a single epoch.
    '''
    num_epochs = config['model']['epochs']
    loss       = 0
    with prof(
        'Trained epoch {} with final loss={}', lambda: epoch, lambda: loss,
        log_level='info'
    ):
        max_batches = config['max_records']//config['batch_size']
        batch_num   = 1
        for X, y in file_mgr.yield_training(name, config['batch_size']):
            loss = train_batch(model, X, y, epoch, num_epochs, batch_num, max_batches)
            batch_num += 1
            del X, y
            gc.collect()
    return loss

def train_model(config, name, file_mgr):
    '''
    Train a model.
    '''
    params = config['model']
    log.info('Training model with parameters {}'.format(params))
    K.set_learning_phase(1)
    model      = make_disassembler(**params)
    num_epochs = 1#params['epochs']
    # NB: Loss doesn't decrease significantly after the first epoch.
    for epoch in range(1, num_epochs + 1):
        train_epoch(file_mgr, config, model, name, epoch)
        # At the end of the epoch, save the model to disk, clear the graph, and then load the model back. This fixes a
        # performance problem caused by the execution graph growing in each batch and the fact that TensorFlow evaluates
        # the entire graph when tf.Session.run() is called, resulting in execution becoming exponentially slower.
        if epoch < num_epochs:
            with prof('Reloaded model'):
                import tempfile
                filepath = tempfile.mkstemp()
                model.save_weights(filepath)
                del model
                K.clear_session()
                gc.collect()
                model = make_disassembler(**params)
                model.load_weights(filepath)
    return model

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
        model = train_model(config, model_name, file_mgr)
        file_mgr.save_model(model, model_name)
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error('Uncaught exception \'{}\': {}'.format(type(e).__name__, str(e).split('\n')[0]))
        log.error('See the log at {} for details.'.format(file_mgr.log_file_path))
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        exit(1)
