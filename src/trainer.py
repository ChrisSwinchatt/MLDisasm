#!/usr/bin/env python3

'''Usage: {0} <model>
'''

import os
import sys
import traceback as tb

if __name__ == '__main__':
    print('*** Starting up...')
    # Filter out debug messages from TF.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow               as tf
import tensorflow.keras.backend as K

from mldisasm.fixes           import fix_output_size
from mldisasm.io.codec        import AsciiCodec
from mldisasm.io.file_manager import FileManager
from mldisasm.model           import Disassembler
from mldisasm.util            import log, prof, refresh_graph

def train_batch(model, X, y, epoch, num_epochs, batch_num, max_batches):
    '''
    Train a single batch.
    '''
    loss = 0
    acc  = 0
    with prof(
        'Epoch {}/{}: Trained batch {}/{} with {}% accuracy, loss={}',
        epoch,
        num_epochs,
        batch_num,
        max_batches,
        lambda: acc,
        lambda: loss,
        log_level='info'
    ):
        history = model.fit(
            X,
            y,
            steps_per_epoch = 1,
            epochs          = 1,
            verbose         = 0
        )
        loss = history.history['loss'][-1]
        acc  = history.history['acc'][-1]*100
    # Exit the context before returning loss so prof can print the loss.
    return loss, acc

def train_epoch(file_mgr, config, codec, model, name, epoch):
    '''
    Train a single epoch.
    '''
    num_epochs = config['model']['epochs']
    loss       = 0
    acc        = 0
    with prof(
        'Trained epoch {} with {}% accuracy, final loss={}', lambda: epoch, lambda: acc, lambda: loss,
        log_level='info'
    ):
        max_batches = config['max_records']//config['model']['batch_size']
        batch_num   = 1
        for X, y in file_mgr.yield_training(name, codec, config['model']['batch_size']):
            loss, acc = train_batch(model, X, y, epoch, num_epochs, batch_num, max_batches)
            batch_num += 1
            # Refresh the graph each ten batches to prevent TF slowdown.
            if batch_num % 10 == 0:
                model = refresh_graph(model=model, build_fn=Disassembler, **(config['model']))
    print('Stopping')
    return loss

def train_model(file_mgr, config, codec, name):
    '''
    Train a model.
    '''
    params = config['model']
    log.info('Training model with parameters {}'.format(params))
    K.set_learning_phase(1)
    model      = Disassembler(**params)
    num_epochs = params['epochs']
    for epoch in range(1, num_epochs + 1):
        # Allow user to stop training with ^C.
        try:
            train_epoch(file_mgr, config, codec, model, name, epoch)
        except KeyboardInterrupt:
            log.warning(
                'Training interrupted at epoch {}/{}, remaining {} epochs will be skipped'\
                '(press ^C again to quit)'.format(
                    epoch,
                    num_epochs,
                    num_epochs - epoch
                )
            )
            break
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
        tokens = file_mgr.load_tokens()
        codec  = AsciiCodec(config['seq_len'], config['mask_value'], tokens)
        # Apply output_size workaround.
        fix_output_size(config, tokens)
        # Train model on whole dataset.
        model = train_model(file_mgr, config, codec, model_name)
        file_mgr.save_model(model, model_name)
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error('Uncaught exception \'{}\': {}'.format(type(e).__name__, str(e).split('\n')[0]))
        log.error('See the log at {} for details.'.format(file_mgr.log_file_path))
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        exit(1)
