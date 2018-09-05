#!/usr/bin/env python3

'''Usage: {0} <model>
'''

import multiprocessing as mp
import os
import sys
import traceback as tb

import numpy as np

# Filter out debug messages from TF.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if __name__ == '__main__':
    print('*** Starting up...')

import tensorflow               as tf
import tensorflow.keras.backend as K

from mldisasm.io.codec        import AsciiCodec, BytesCodec
from mldisasm.io.file_manager import FileManager
from mldisasm.util            import log, prof, force_cpu
from mldisasm.training        import parameter_grid, kfolds_train

# Force TF to use the CPU.
force_cpu()

def tune_model(queue, config, params, file_mgr, model_name, codecs):
    '''
    Train a model with a set of parameters and return the average accuracy and loss during cross-validation.
    '''
    X, y = file_mgr.load_training(model_name, codecs, max_records=config['gs_records'])
    acc, loss = kfolds_train(X, y, params)
    queue.put((acc,loss))

def select_params(config, file_mgr, model_name, codecs):
    '''
    Select hyperparameters by gridsearch with cross-validation.
    '''
    log.info('Selecting hyperparameters')
    grid = parameter_grid(config['grid'])
    if len(grid) == 0:
        log.error('No parameters to tune. Stopping.')
        exit(0)
    fit_num     = 1
    num_fits    = len(grid)
    best_params = None
    best_acc    = -np.inf
    acc         = 0
    loss        = 0
    for grid_params in grid:
        with prof(
            'Grid {}/{}: acc={}%, loss={}', fit_num, num_fits, lambda: 100*acc, lambda: round(loss, 4),
            log_level='info',
            start_msg='Grid {}/{}: {}'.format(fit_num, num_fits, grid_params)
        ):
            params = dict(config['model'])
            params.update(grid_params)
            try:
                queue   = mp.Queue()
                process = mp.Process(target=tune_model, args=(queue,config,params,file_mgr,model_name,codecs))
                process.start()
                acc, loss = queue.get()
            finally:
                process.join()
            # Select model by accuracy.
            if acc > best_acc:
                best_acc = acc
                best_params = params
            fit_num += 1
    assert best_params is not None
    log.info('Best accuracy was {}% with parameters {}'.format(round(best_acc*100, 2), best_params))
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
    K.set_learning_phase(1)
    # Read command-line args.
    model_name = read_command_line()
    # Start file manager & logging.
    tf.logging.set_verbosity(tf.logging.INFO)
    file_mgr = FileManager()
    log.init(file_mgr.open_log())
    # Train a model.
    try:
        # Load configuration and set TF device.
        config  = file_mgr.load_config(model_name)
        x_codec = BytesCodec(config['model']['x_seq_len'], config['model']['mask_value'])
        y_codec = AsciiCodec(config['model']['y_seq_len'], config['model']['mask_value'])
        # Find and save hyperparameters.
        config['model'] = select_params(config, file_mgr, model_name, (x_codec,y_codec))
        del config['grid']
        file_mgr.save_config(model_name, config)
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error('Uncaught exception \'{}\': {}'.format(type(e).__name__, str(e).split('\n')[0]))
        log.error('See the log at {} for details.'.format(file_mgr.log_file_path))
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        exit(1)
