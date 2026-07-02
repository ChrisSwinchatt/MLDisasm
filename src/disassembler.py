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

"""Disassemble a binary file using one of the supported syntax models.

Usage: {0} <model> <binary>

 * <model> is the name of the assembly syntax model to use, such as "intel" or "att" (AT&T).
 * <binary> names a file containing machine code. The file will be interpreted as raw binary.

Disassembly is written to standard output.
"""

import os
import sys

if __name__ == '__main__':
    print('*** Starting up...')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from mldisasm.io.codec        import AssemblyTokenizer, ByteIndexer
from mldisasm.io.file_manager import FileManager
from mldisasm.util            import log
from mldisasm.model           import Disassembler

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
    model_name = sys.argv[1]
    input_path = sys.argv[2]
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    file_mgr = FileManager()
    config   = file_mgr.load_config(model_name)
    x_codec  = ByteIndexer(config['model']['x_seq_len'])
    y_codec  = AssemblyTokenizer(config['model']['y_seq_len'])

    model = Disassembler(**config['model'])
    model.load_weights(file_mgr.qualify_model(model_name))

    with open(input_path, 'rb') as f:
        offset = 0
        while True:
            buf = f.read(config['model']['x_seq_len'])
            if not buf:
                break
            encoded = x_codec.encode(buf)
            decoded = model.infer(tf.convert_to_tensor([encoded]))
            print(f'0x{offset:08x}:\t{buf.hex()}\t{decoded}')
            offset += len(buf)
