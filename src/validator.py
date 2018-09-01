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

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from mldisasm.io.codec        import AsciiCodec, BytesCodec
from mldisasm.io.file_manager import FileManager
from mldisasm.model           import trainable_disassembler

if __name__ == '__main__':
    # Read the command line.
    if len(sys.argv) != 2:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        exit(1)
    model_name = sys.argv[1]
    # Load files and create codecs.
    file_mgr   = FileManager()
    config     = file_mgr.load_config(model_name)
    x_codec = BytesCodec(config['model']['x_seq_len'], config['model']['mask_value'])
    y_codec = AsciiCodec(config['model']['x_seq_len'], config['model']['mask_value'])
    # NB: Creating a new model and loading the weights into it works around a bug in keras.models.load_model(). This
    # will fail if the model configuration (number of units or layers) changes between saving and loading the model.
    model = trainable_disassembler(**config['model'])
    model.load_weights(file_mgr.qualify_model(model_name))
    # Perform validation line-by-line.
    sample = 0
    losses = [None]*config['max_records']
    accs   = [None]*config['max_records']
    for X, y_true in file_mgr.yield_validation(model_name, y_codec):
        X      = np.asarray([X])
        y_true = np.asarray([y_true])
        # Compute predictions and loss.
        y_pred  = model.predict(X)
        metrics = model.test_on_batch(X, y_true)
        if model.metrics_names == ['acc','loss']:
            acc, loss = metrics
        elif model.metrics_names == ['loss','acc']:
            loss, acc = metrics
        else:
            raise ValueError('Unrecognised metrics names: {}'.format(','.join(model.metrics_names)))
        accs[sample]   = acc
        losses[sample] = loss
        sample += 1
        # Decode and print results.
        #print('y_true =', y_true, '; y_pred =', y_pred)
        y_true = y_codec.decode(y_true)[0]
        y_pred = y_codec.decode(y_pred)[0]
        print('Sample {}: loss={}, avg_loss={}, acc={}%, avg_acc={}%, y_pred="{}", y_true="{}"'.format(
            sample,
            loss,
            np.mean(losses[:sample]),
            acc,
            np.mean(accs[:sample]),
            y_pred,
            y_true
        ))
        if sample >= config['max_records']:
            break
    print('Validated {} samples')
    print('          MIN\tMEAN\tMAX')
    print('Accuracy: {}%\t{}%\t{}%'.format(min(accs), np.mean(accs),   max(accs)))
    print('Loss:     {}\t{}\t{}'.format(min(losses),  np.mean(losses), max(losses)))
