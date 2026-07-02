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

"""Validate a disassembly model.

Usage: {0} <model>
"""

import os
import sys

if __name__ == '__main__':
    print('*** Starting up...')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

from mldisasm.io.codec        import AssemblyTokenizer, ByteIndexer
from mldisasm.io.file_manager import FileManager
from mldisasm.model           import Disassembler

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]

    file_mgr = FileManager()
    config   = file_mgr.load_config(model_name)
    x_codec  = ByteIndexer(config['model']['x_seq_len'])
    y_codec  = AssemblyTokenizer(config['model']['y_seq_len'])

    model = Disassembler(**config['model'])
    model.load_weights(file_mgr.qualify_model(model_name))

    sample = 0
    for X, y_true in file_mgr.yield_validation(model_name, (x_codec, y_codec)):
        y_pred = model.infer(X)
        sample += 1
        print(f'Sample {sample}: y_pred="{y_pred}", y_true="{y_true}"')
        if sample >= config['max_records']:
            break
    print(f'Validated {sample} samples')
