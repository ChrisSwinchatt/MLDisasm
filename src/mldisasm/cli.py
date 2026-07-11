#!/usr/bin/env python3

# Copyright (C) 2018,2021,2026 Chris Swinchatt
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

import multiprocessing as mp
import os
import sys
import traceback as tb

import numpy as np
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def _load_config(model_name):
    """Load config and create codecs."""
    from mldisasm.io.file_manager import FileManager
    from mldisasm.io.codec import AssemblyTokenizer, ByteIndexer
    file_mgr = FileManager()
    config = file_mgr.load_config(model_name)
    x_codec = ByteIndexer(config['model']['x_seq_len'])
    y_codec = AssemblyTokenizer(config['model']['y_seq_len'])
    return file_mgr, config, x_codec, y_codec


def disassemble():
    """Disassemble a binary file."""
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <model> <binary>', file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]
    input_path = sys.argv[2]

    from mldisasm.model import Disassembler
    file_mgr, config, x_codec, y_codec = _load_config(model_name)

    model = Disassembler(**config['model'])
    file_mgr.load_model(model_name, model)

    with open(input_path, 'rb') as f:
        offset = 0
        while True:
            buf = f.read(config['model']['x_seq_len'])
            if not buf:
                break
            encoded = x_codec.encode(buf)
            decoded = model.infer(torch.tensor([encoded]))
            print(f'0x{offset:08x}:\t{buf.hex()}\t{decoded}')
            offset += len(buf)


def train():
    """Train a model."""
    from mldisasm.training import train_model
    from mldisasm.util import log

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <model>', file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]

    file_mgr, config, x_codec, y_codec = _load_config(model_name)
    log.init(file_mgr.open_log())

    try:
        params = config['model']
        X, y = file_mgr.load_training(
            model_name, (x_codec, y_codec), max_records=config['max_records']
        )
        model, acc, loss = train_model(X, y, params)
        log.info(f'Final accuracy: {acc * 100:.2f}%, loss: {loss:.4f}')
        file_mgr.save_model(model, model_name)
    except Exception as e:
        log.debug('=' * 20 + '[ UNCAUGHT EXCEPTION ]' + '=' * 20)
        log.error(f"Uncaught exception '{type(e).__name__}': {str(e).split(chr(10))[0]}")
        log.error(f'See the log at {file_mgr.log_file_path} for details.')
        log.debug('Exception Traceback:\n' + ''.join(tb.format_tb(e.__traceback__)))
        sys.exit(1)


def tune():
    """Select hyperparameters."""
    from mldisasm.training import parameter_grid
    from mldisasm.util import log, prof

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <model>', file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]

    file_mgr, config, x_codec, y_codec = _load_config(model_name)
    log.init(file_mgr.open_log())

    try:
        log.info('Selecting hyperparameters')
        grid = parameter_grid(config['grid'])
        if len(grid) == 0:
            log.error('No parameters to tune. Stopping.')
            sys.exit(0)

        fit_num = 0
        num_fits = len(grid)
        best_params = None
        best_acc = -np.inf

        for grid_params in grid:
            fit_num += 1
            params = dict(config['model'])
            params.update(grid_params)
            acc = 0.0
            loss = 0.0
            with prof(
                'Grid {}/{}: acc={}%, loss={}', fit_num, num_fits,
                lambda: 100 * acc, lambda: round(loss, 4),
                log_level='info',
                start_msg=f'Grid {fit_num}/{num_fits}: {grid_params}'
            ):
                queue = mp.Queue()
                process = mp.Process(
                    target=_tune_subprocess,
                    args=(queue, config, params, file_mgr, model_name, (x_codec, y_codec))
                )
                try:
                    process.start()
                    acc, loss = queue.get()
                finally:
                    process.join()
                if acc > best_acc:
                    best_acc = acc
                    best_params = params

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
    """Train model with parameters in a subprocess."""
    from mldisasm.training import kfolds_train
    X, y = file_mgr.load_training(model_name, codecs, max_records=config['gs_records'])
    acc, loss = kfolds_train(X, y, params)
    queue.put((acc, loss))


def validate():
    """Validate a disassembly model."""
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <model>', file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]

    from mldisasm.model import Disassembler
    file_mgr, config, x_codec, y_codec = _load_config(model_name)

    model = Disassembler(**config['model'])
    file_mgr.load_model(model_name, model)

    sample = 0
    for X, y_true in file_mgr.yield_validation(model_name, (x_codec, y_codec)):
        y_pred = model.infer(X)
        sample += 1
        print(f'Sample {sample}: y_pred="{y_pred}", y_true="{y_true}"')
        if sample >= config['max_records']:
            break
    print(f'Validated {sample} samples')
