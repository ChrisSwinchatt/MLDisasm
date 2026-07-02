#!/usr/bin/env python3

# Copyright (C) 2018,2021 Chris Swinchatt
#
# MLDisasm is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# MLDisasm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with MLDisasm.  If not, see <http://www.gnu.org/licenses/>.

"""Usage: {0} <model>
"""

import os
import sys
import traceback as tb

if __name__ == '__main__':
    print('*** Starting up...')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from mldisasm.io.codec        import AsciiCodec, BytesCodec
from mldisasm.io.file_manager import FileManager
from mldisasm.training        import train_model
from mldisasm.util            import log, prof


def batch_train(config, params, file_mgr, model_name, codecs):
    """Train a model in batches."""
    model = None
    for X, y in file_mgr.yield_training(
        model_name, codecs,
        batch_size=config['max_records'] // 10,
        max_records=config['max_records']
    ):
        with prof('Trained batch'):
            model, _, _ = train_model(X, y, params, retrain=model)
    return model


def read_command_line():
    """Get the model name from the command-line."""
    if len(sys.argv) != 2:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
    return sys.argv[1]


if __name__ == '__main__':
    model_name = read_command_line()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    file_mgr = FileManager()
    log.init(file_mgr.open_log())
    try:
        config  = file_mgr.load_config(model_name)
        params  = config['model']
        x_codec = BytesCodec(params['x_seq_len'], params['mask_value'])
        y_codec = AsciiCodec(params['y_seq_len'], params['mask_value'])
        model = batch_train(config, params, file_mgr, model_name, (x_codec, y_codec))
        model.save_weights(file_mgr.qualify_model(model_name))
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error(f"Uncaught exception '{type(e).__name__}': {str(e).split(chr(10))[0]}")
        log.error(f'See the log at {file_mgr.log_file_path} for details.')
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        sys.exit(1)
