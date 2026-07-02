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

"""CLI entry points for MLDisasm."""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def disassemble():
    """Disassemble a binary file."""
    from mldisasm.io.codec import AsciiCodec, BytesCodec
    from mldisasm.io.file_manager import FileManager
    from mldisasm.model import Disassembler

    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <model> <binary>', file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]
    input_path = sys.argv[2]

    file_mgr = FileManager()
    config = file_mgr.load_config(model_name)
    x_codec = BytesCodec(config['model']['x_seq_len'], config['model']['mask_value'])
    y_codec = AsciiCodec(config['model']['y_seq_len'], config['model']['mask_value'])

    model = Disassembler(**config['model'])
    model.load_weights(file_mgr.qualify_model(model_name))

    with open(input_path, 'rb') as f:
        offset = 0
        while True:
            buf = f.read(config['model']['x_seq_len'])
            if not buf:
                break
            encoded = x_codec.encode(buf)
            pred = model.predict(tf.convert_to_tensor([encoded]))
            decoded = y_codec.decode(pred)
            print(f'0x{offset:08x}:\t{buf.hex()}\t{decoded}')
            offset += len(buf)


def train():
    """Train a model."""
    import tensorflow as tf
    from mldisasm.io.codec import AsciiCodec, BytesCodec
    from mldisasm.io.file_manager import FileManager
    from mldisasm.model import Disassembler
    from mldisasm.training import train_model
    from mldisasm.util import log, prof

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <model>', file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]

    file_mgr = FileManager()
    log.init(file_mgr.open_log())

    try:
        config = file_mgr.load_config(model_name)
        params = config['model']
        x_codec = BytesCodec(params['x_seq_len'], params['mask_value'])
        y_codec = AsciiCodec(params['y_seq_len'], params['mask_value'])

        model = None
        for X, y in file_mgr.yield_training(
            model_name, (x_codec, y_codec),
            batch_size=config['max_records'] // 10,
            max_records=config['max_records']
        ):
            with prof('Trained batch'):
                model, _, _ = train_model(X, y, params, retrain=model)

        model.save_weights(file_mgr.qualify_model(model_name))
    except Exception as e:
        import traceback as tb
        log.debug('=' * 20 + '[ UNCAUGHT EXCEPTION ]' + '=' * 20)
        log.error(f"Uncaught exception '{type(e).__name__}': {str(e).split(chr(10))[0]}")
        log.error(f'See the log at {file_mgr.log_file_path} for details.')
        log.debug('Exception Traceback:\n' + ''.join(tb.format_tb(e.__traceback__)))
        sys.exit(1)


def tune():
    """Select hyperparameters."""
    import multiprocessing as mp
    import traceback as tb

    import numpy as np

    from mldisasm.io.codec import AsciiCodec, BytesCodec
    from mldisasm.io.file_manager import FileManager
    from mldisasm.training import parameter_grid, kfolds_train
    from mldisasm.util import log, prof

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <model>', file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]

    file_mgr = FileManager()
    log.init(file_mgr.open_log())

    try:
        config = file_mgr.load_config(model_name)
        x_codec = BytesCodec(config['model']['x_seq_len'], config['model']['mask_value'])
        y_codec = AsciiCodec(config['model']['y_seq_len'], config['model']['mask_value'])

        log.info('Selecting hyperparameters')
        grid = parameter_grid(config['grid'])
        if len(grid) == 0:
            log.error('No parameters to tune. Stopping.')
            sys.exit(0)

        fit_num = 1
        num_fits = len(grid)
        best_params = None
        best_acc = -np.inf
        acc = 0
        loss = 0

        for grid_params in grid:
            with prof(
                'Grid {}/{}: acc={}%, loss={}', fit_num, num_fits,
                lambda: 100 * acc, lambda: round(loss, 4),
                log_level='info',
                start_msg=f'Grid {fit_num}/{num_fits}: {grid_params}'
            ):
                params = dict(config['model'])
                params.update(grid_params)
                try:
                    queue = mp.Queue()
                    process = mp.Process(
                        target=_tune_subprocess,
                        args=(queue, config, params, file_mgr, model_name, (x_codec, y_codec))
                    )
                    process.start()
                    acc, loss = queue.get()
                finally:
                    process.join()
                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                fit_num += 1

        assert best_params is not None
        log.info(f'Best accuracy was {round(best_acc * 100, 2)}% with parameters {best_params}')
        config['model'] = best_params
        del config['grid']
        file_mgr.save_config(model_name, config)
    except Exception as e:
        log.debug('=' * 20 + '[ UNCAUGHT EXCEPTION ]' + '=' * 20)
        log.error(f"Uncaught exception '{type(e).__name__}': {str(e).split(chr(10))[0]}")
        log.error(f'See the log at {file_mgr.log_file_path} for details.')
        log.debug('Exception Traceback:\n' + ''.join(tb.format_tb(e.__traceback__)))
        sys.exit(1)


def _tune_subprocess(queue, config, params, file_mgr, model_name, codecs):
    """Train a model with a set of parameters (runs in subprocess)."""
    from mldisasm.training import kfolds_train
    X, y = file_mgr.load_training(model_name, codecs, max_records=config['gs_records'])
    acc, loss = kfolds_train(X, y, params)
    queue.put((acc, loss))


def validate():
    """Validate a disassembly model."""
    import numpy as np
    import tensorflow as tf

    from mldisasm.io.codec import AsciiCodec, BytesCodec
    from mldisasm.io.file_manager import FileManager
    from mldisasm.model import Disassembler

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <model>', file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]

    file_mgr = FileManager()
    config = file_mgr.load_config(model_name)
    x_codec = BytesCodec(config['model']['x_seq_len'], config['model']['mask_value'])
    y_codec = AsciiCodec(config['model']['y_seq_len'], config['model']['mask_value'])

    model = Disassembler(**config['model'])
    model.load_weights(file_mgr.qualify_model(model_name))

    sample = 0
    losses = [None] * config['max_records']
    accs = [None] * config['max_records']
    for X, y_true in file_mgr.yield_validation(model_name, (x_codec, y_codec)):
        X = tf.expand_dims(X, 0)
        y_true_str = y_true
        y_pred = model.infer(X)
        sample += 1
        print(f'Sample {sample}: y_pred="{y_pred}", y_true="{y_true_str}"')
        if sample >= config['max_records']:
            break
    print(f'Validated {sample} samples')
