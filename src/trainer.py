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
from mldisasm.training        import train_model
from mldisasm.util            import log
from mldisasm.util.force_cpu  import force_cpu

force_cpu()

def batch_train(config, params, file_mgr, model_name, codecs):
    '''
    Train a model in batches.
    '''
    model = None
    for X, y in file_mgr.yield_training(model_name, codecs, batch_size=config['max_records']//10, max_records=config['max_records']):
        model, _, _ = train_model(X, y, params, retrain=model)
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
    K.set_learning_phase(1)
    # Read command-line args.
    model_name = read_command_line()
    # Start file manager & logging.
    tf.logging.set_verbosity(tf.logging.INFO)
    file_mgr = FileManager()
    log.init(file_mgr.open_log())
    try:
        # Load configuration,
        config  = file_mgr.load_config(model_name)
        params  = config['model']
        x_codec = BytesCodec(params['x_seq_len'], params['mask_value'])
        y_codec = AsciiCodec(params['y_seq_len'], params['mask_value'])
        # Train model on whole dataset and save weights.
        model = batch_train(config, params, file_mgr, model_name, (x_codec,y_codec))
        model.save_weights(file_mgr.qualify_model(model_name))
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error('Uncaught exception \'{}\': {}'.format(type(e).__name__, str(e).split('\n')[0]))
        log.error('See the log at {} for details.'.format(file_mgr.log_file_path))
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        exit(1)
