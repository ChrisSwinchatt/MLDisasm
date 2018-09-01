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

from mldisasm.io.codec        import AsciiCodec, BytesCodec
from mldisasm.io.file_manager import FileManager
from mldisasm.model           import trainable_disassembler
from mldisasm.training        import train_epoch
from mldisasm.util            import log

def train_model(file_mgr, config, codecs, name):
    '''
    Train a model.
    '''
    params = config['model']
    log.info('Training model with parameters {}'.format(params))
    K.set_learning_phase(1)
    model = trainable_disassembler(**params)
    num_epochs = params['epochs']
    for epoch in range(1, num_epochs + 1):
        _, _, model = train_epoch(
            model,
            file_mgr.yield_training(name, codecs, config['batch_size'], max_records=config['max_records']),
            epoch,
            num_epochs,
            params,
            config['max_records']//config['batch_size']
        )
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
        config  = file_mgr.load_config(model_name)
        x_codec = BytesCodec(config['model']['x_seq_len'], config['model']['mask_value'])
        y_codec = AsciiCodec(config['model']['y_seq_len'], config['model']['mask_value'])
        # Train model on whole dataset.
        model = train_model(file_mgr, config, (x_codec,y_codec), model_name)
        model.save_weights(file_mgr.qualify_model(model_name))
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error('Uncaught exception \'{}\': {}'.format(type(e).__name__, str(e).split('\n')[0]))
        log.error('See the log at {} for details.'.format(file_mgr.log_file_path))
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        exit(1)
