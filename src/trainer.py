#!/usr/bin/env python3

'''
Usage: {0} <model>
'''

import os
import sys
import traceback as tb
import warnings

# Filter out debug messages from TF.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Ignore warnings generated by using a different NumPy version.
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

if __name__ == '__main__':
    print('*** Starting up...')

import tensorflow as tf

import mldisasm.benchmarks.profiling as     profiling
from   mldisasm.benchmarks.profiling import prof
from   mldisasm.io.codec             import AsciiCodec
import mldisasm.io.log               as     log
from   mldisasm.io.file_manager      import FileManager
from   mldisasm.model.disassembler   import Disassembler

def train_model(config, tset, y_codec, session=None):
    '''
    Train a model.
    '''
    # Create a model.
    model = Disassembler(
        **config['model'],
        decoder    = y_codec,
        batch_size = config['batch_size'],
        seq_len    = config['seq_len']
    )
    n_epochs = config['epochs']
    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        batch_num  = 1
        log.info('Epoch {} of {}'.format(epoch,n_epochs))
        profiler = prof('Epoch {} finished with loss={}', epoch, total_loss)
        for X, y in tset:
            log.info('`- Batch {}'.format(batch_num))
            total_loss = model.train(X, y)
            if session:
                session.run(total_loss)
            batch_num += 1
            if 'max_batches' in config and batch_num >= config['max_batches']:
                break
        profiler.end()

def load_datasets(model_name, config, file_mgr):
    '''
    Load training and token sets.
    '''
    tokens = file_mgr.load_tokens(**config)
    tset   = file_mgr.open_training(
        model_name,
        batch_size=config['batch_size'],
        seq_len=config['seq_len']
    )
    return tset, AsciiCodec(config['seq_len'], tokens)

def select_device(config):
    '''
    Select a TensorFlow device according to configuration.
    '''
    log.info('Checking TensorFlow device configuration (this can take some time)')
    preferred = config['preferred_device']
    fallback  = config['fallback_device']
    if 'gpu' in preferred.lower() and not tf.test.is_gpu_available():
        if fallback is None:
            log.error('Preferred device \'{}\' is not available and no fallback device was specified, stopping.')
            exit(1)
        log.warning('Preferred device \'{}\' is not available, falling back to \'{}\''.format(
            preferred,
            fallback
        ))
        return fallback
    log.info('Preferred TensorFlow device \'{}\' is available'.format(preferred))
    return preferred

def start_training(model_name, file_mgr):
    '''
    Train a model within a TF session.
    '''
    # Load configuration and set TF device.
    config = file_mgr.load_config()
    select_device(config)
    # Initialise profiler.
    profiling.init(config['prof_time'], config['prof_mem'])
    # Create a default session.
    with tf.Session() as session:
        session.as_default()
        # Load datasets and start training.
        tset, codec = load_datasets(model_name, config, file_mgr)
        train_model(config, tset, codec, session)

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
