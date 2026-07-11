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

"""Usage: {0} <model>
"""

import multiprocessing as mp
import os
import sys
import traceback as tb

import numpy as np

if __name__ == '__main__':
    print('*** Starting up...')

import torch

from mldisasm.io.codec        import AssemblyTokenizer, ByteIndexer
from mldisasm.io.file_manager import FileManager
from mldisasm.training        import parameter_grid, kfolds_train
from mldisasm.util            import log, prof


def tune_subprocess(queue, config, params, file_mgr, model_name, codecs):
    """Train a model with a set of parameters (runs in subprocess)."""
    X, y = file_mgr.load_training(model_name, codecs, max_records=config['gs_records'])
    acc, loss = kfolds_train(X, y, params)
    queue.put((acc, loss))


def select_params(config, file_mgr, model_name, codecs):
    """Select hyperparameters by gridsearch with cross-validation."""
    log.info('Selecting hyperparameters')
    grid = parameter_grid(config['grid'])
    if len(grid) == 0:
        log.error('No parameters to tune. Stopping.')
        sys.exit(0)
    fit_num     = 1
    num_fits    = len(grid)
    best_params = None
    best_acc    = -np.inf
    acc         = 0.0
    loss        = 0.0
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
                queue   = mp.Queue()
                process = mp.Process(
                    target=tune_subprocess,
                    args=(queue, config, params, file_mgr, model_name, codecs)
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
    return best_params


def read_command_line():
    """Get the model name from the command-line."""
    if len(sys.argv) != 2:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
    return sys.argv[1]


if __name__ == '__main__':
    model_name = read_command_line()
    file_mgr = FileManager()
    log.init(file_mgr.open_log())
    try:
        config  = file_mgr.load_config(model_name)
        x_codec = ByteIndexer(config['model']['x_seq_len'])
        y_codec = AssemblyTokenizer(config['model']['y_seq_len'])
        config['model'] = select_params(config, file_mgr, model_name, (x_codec, y_codec))
        del config['grid']
        file_mgr.save_config(model_name, config)
    except Exception as e:
        log.debug('====================[ UNCAUGHT EXCEPTION ]====================')
        log.error(f"Uncaught exception '{type(e).__name__}': {str(e).split(chr(10))[0]}")
        log.error(f'See the log at {file_mgr.log_file_path} for details.')
        log.debug('Exception Traceback:\n{}'.format(''.join(tb.format_tb(e.__traceback__))))
        sys.exit(1)
