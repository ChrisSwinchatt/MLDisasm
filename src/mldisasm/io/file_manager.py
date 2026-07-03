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

'''
MLDisasm file manager.
'''

import os

import numpy as np

import tensorflow as tf
from tensorflow import keras

try:
    import ujson as json
except ImportError:
    import json

from mldisasm.util import prof, log

class FileManager:
    '''
    Manage files within a directory.
    '''
    # Default data directory path.
    default_data_dir = os.path.join(os.getcwd(), 'data')

    def __init__(self, data_dir=None):
        '''
        Initialise FileManager.
        :param data_dir: The path to the data directory. If None, defaults to FileManager.default_data_dir.
        '''
        if data_dir is None:
            data_dir = FileManager.default_data_dir
        self.chdir(data_dir)

    def chdir(self, path):
        '''
        Change the managed directory.
        :note: This does *not* invalidate open handles.
        '''
        if path is None:
            raise ValueError('Can\'t change directory to {}'.format(path))
        self._data_dir = path

    def qualify(self, *args):
        '''
        Qualify a path.
        '''
        return os.path.join(self._data_dir, *args)

    ############################################################################
    # CONFIG                                                                   #
    ############################################################################
    def qualify_config(self, name):
        '''
        Get the qualified filename of a configuration file.
        '''
        return self.qualify(name, FileManager._config_name)

    def open_config(self, name, *args, **kwargs):
        '''
        Open configuration file.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        return open(self.qualify_config(name), *args, **kwargs)

    def load_config(self, name, *args, **kwargs):
        '''
        Load configuration data.
        :param name: The configuration name.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        with self.open_config(name, *args, **kwargs) as file:
            return json.load(file)

    def save_config(self, name, config):
        '''
        Save a configuration to json.
        :param config: A configuration.
        '''
        with self.open_config(name, 'w', newline='\n') as file:
            json.dump(config, file, indent=4, )
            file.write('\n')

    ############################################################################
    # LOG                                                                      #
    ############################################################################
    def qualify_log(self):
        '''
        Get the qualified filename of the log.
        '''
        return self.qualify(FileManager._log_name)

    @property
    def log_file_path(self):
        '''
        Get the path to the log file.
        '''
        return self.qualify_log()

    def open_log(self, *args, **kwargs):
        '''
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        return open(self.qualify_log(), 'w', *args, **kwargs)

    ############################################################################
    # MODEL                                                                    #
    ############################################################################
    def qualify_model(self, name):
        '''
        Get the qualified filename of a model.
        '''
        return self.qualify(name, FileManager._model_name)

    def load_model(self, name):
        '''
        Load a model.
        :param name: The model name.
        :returns: The loaded model.
        '''
        return keras.models.load_model(self.qualify_model(name))

    def save_model(self, model, name):
        '''
        Save a model.
        :param model: The model to save.
        '''
        model.save(self.qualify_model(name), overwrite=True)

    ############################################################################
    # TRAINING                                                                 #
    ############################################################################
    def qualify_training(self, name):
        '''
        Get the qualified filename of a training set.
        '''
        return self.qualify(name, FileManager._training_name)

    def open_training(self, name, *args, **kwargs):
        '''
        Open training set file.
        :param name: The name of the training set.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open(). Note: Any 'newline' key will be overridden with the value of '\n'.
        :returns: An open handle to the training set file.
        '''
        kwargs['newline'] = '\n'
        return open(self.qualify_training(name), 'r', *args, **kwargs)

    def load_training(self, name, codecs, block_size=65536, max_records=np.inf):
        '''
        Load (up to) an entire training set into memory at once.
        :param name: The model name.
        :param codecs: A tuple of (ByteIndexer, AssemblyTokenizer).
        :param block_size: The amount of data to read at once. Affects I/O performance but probably isn't critical.
        Default is 64K.
        :param max_records: The maximum number of records to load. Default: infinity, which means load everything.
        :returns: A tuple of (X, y) where X is (byte_indices, token_indices) and y is token_indices.
        '''
        with self.open_training(name) as file:
            return _do_load_training(file, codecs, block_size, max_records)

    def yield_training(self, name, codecs, batch_size, block_size=65535, max_records=np.inf, loop_mode=False):
        """
        Yield training samples in batches.

        :param name: Training set name.
        :param codecs: Tuple of (ByteIndexer, AssemblyTokenizer).
        :param batch_size: Records per batch.
        :param block_size: Bytes to read at once from file.
        :param max_records: Maximum records to yield (inf = unlimited).
        :param loop_mode: If True, loop indefinitely.
        :yields: Tuple ((byte_indices, token_indices), token_indices).
        """
        with self.open_training(name) as file:
            batch_num = 0
            num_records = max_records
            while True:
                count = min(batch_size, num_records)
                batch_num += 1
                Xy = None
                if count:
                    Xy = _do_load_training(file, codecs, block_size, count)
                if not Xy:
                    if not loop_mode:
                        break
                    log.debug('Restarting training file generator')
                    file.seek(0)
                    count = batch_size
                    batch_num = 0
                    num_records = max_records
                    continue
                X, y = Xy
                x_bytes, x_tokens = X
                assert x_bytes.shape[0] == x_tokens.shape[0] == y.shape[0]
                num_records -= int(x_bytes.shape[0])
                if num_records < 0:
                    num_records = 0
                yield X, y

    ############################################################################
    # VALIDATION                                                               #
    ############################################################################
    def qualify_validation(self, name):
        '''
        Get the qualified filename of a validation set.
        '''
        return self.qualify(name, FileManager._validation_name)

    def open_validation(self, name):
        '''
        Open validation file.
        '''
        return open(self.qualify_validation(name), 'r')

    def load_validation(self, name, codecs, block_size=65536, max_records=np.inf):
        '''
        Load (up to) an entire validation set into memory at once.
        :param name: The model name.
        :param codecs: A tuple of (ByteIndexer, AssemblyTokenizer).
        :param block_size: The amount of data to read at once.
        Default is 64K.
        :param max_records: The maximum number of records to load.
        :returns: A tuple of ((byte_indices, token_indices), token_indices).
        '''
        with self.open_validation(name) as file:
            return _do_load_training(file, codecs, block_size, max_records)

    def yield_validation(self, name, codecs):
        '''
        Yield validation samples one at a time.
        :param name: The model name.
        :param codecs: A tuple of (ByteIndexer, AssemblyTokenizer).
        :yields: A tuple of (byte_indices, ground_truth_string).
        '''
        with self.open_validation(name) as file:
            x_codec, y_codec = codecs
            for line in file:
                opcode, disasm = line.split('|')
                yield x_codec.encode(opcode), disasm

    ############################################################################
    # CONSTANTS                                                                #
    ############################################################################
    _log_name          = 'mldisasm.log'     # Log filename.
    _config_name       = 'config.json'      # Config filename.
    _model_name        = 'model.h5'         # Model filename.
    _training_name     = 'training.csv'     # Preprocessed training set filename.
    _validation_name   = 'validation.csv'   # Validation training set filename.


def _do_load_training(file, codecs, block_size, max_records):
    """Load grouped instruction blocks for autoregressive training.

    Groups consecutive instructions from the CSV into blocks of
    approximately x_codec.seq_len bytes each, so each training sample
    fills the model's context window with real instruction data.

    :returns: A tuple of ((byte_indices, token_indices), token_indices) or None if EOF.
    """
    x_codec, y_codec = codecs
    chunk_bytes = x_codec.seq_len
    records = []

    data = ''
    while True:
        block = file.read(block_size)
        if not block:
            break
        data += block
    if not data:
        return None

    lines = data.split('\n')
    acc_opcode = ''
    acc_disasm = ''

    for line in lines:
        if not line or '|' not in line:
            continue
        opcode, disasm = line.split('|', 1)
        acc_opcode += opcode
        acc_disasm += disasm

        if len(acc_opcode) // 2 >= chunk_bytes:
            records.append((
                x_codec.encode(acc_opcode, as_tensor=False),
                y_codec.encode(acc_disasm, as_tensor=False),
            ))
            acc_opcode = ''
            acc_disasm = ''
            if max_records is not None and len(records) >= max_records:
                break

    # Emit final partial block if it has at least one instruction.
    if acc_opcode and (max_records is None or len(records) < max_records):
        records.append((
            x_codec.encode(acc_opcode, as_tensor=False),
            y_codec.encode(acc_disasm, as_tensor=False),
        ))

    if not records:
        return None

    with prof('Loaded batch ({} blocks)', lambda: len(records)):
        X_bytes, X_tokens = zip(*records)
        X = (tf.convert_to_tensor(list(X_bytes)), tf.convert_to_tensor(list(X_tokens)))
        return X, tf.convert_to_tensor(list(X_tokens))
