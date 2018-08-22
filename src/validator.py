#!/usr/bin/env python3

'''Validate a disassembly model.

Usage: {0} <model>
'''

import os
import sys

import numpy as np

if __name__ == '__main__':
    print('*** Starting up...')
    # Filter out debug messages from TF.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow               as tf
import tensorflow.keras         as keras
import tensorflow.keras.backend as K

from   mldisasm.io.codec        import AsciiCodec, BytesCodec
from   mldisasm.io.file_manager import FileManager
import mldisasm.io.log          as     log
from   mldisasm.model           import make_disassembler

if __name__ == '__main__':
    # Read the command line.
    if len(sys.argv) != 2:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        exit(1)
    model_name = sys.argv[1]
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load files and create codecs.
    file_mgr   = FileManager()
    config     = file_mgr.load_config()
    tokens     = file_mgr.load_tokens()
    seq_len    = config['seq_len']
    mask_value = config['mask_value']
    x_codec    = BytesCodec(seq_len, mask_value)
    y_codec    = AsciiCodec(seq_len, mask_value, tokens)
    # NB: Creating a new model and loading the weights into it works around a bug in keras.models.load_model(). This
    # will fail if the model configuration (number of units or layers) changes between saving and loading the model.
    model = make_disassembler(**config['model'])
    model.load_weights(file_mgr._qualify_model(model_name))
    # Perform validation.
    sample  = 1
    losses = []
    session = K.get_session()
    with session.as_default():
        for X, y_true in file_mgr.yield_validation(model_name):
            # Convert to NumPy arrays.
            X       = np.asarray([X])
            y_true  = np.asarray([y_true])
            # Compute predictions and loss.
            y_pred  = model.predict(X)
            loss    = model.test_on_batch(X, y_true)[-1]
            # Decode and print results.
            y_true  = y_codec.decode(y_true)
            y_pred  = y_codec.decode(y_pred)
            losses.append(loss)
            print('Sample {}: loss={}, avg_loss={}, y_pred=\'{}\', y_true=\'{}\''.format(sample, loss, np.mean(losses), y_pred, y_true))
            sample += 1
