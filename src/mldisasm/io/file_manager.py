#!/usr/bin/env python3

'''
MLDisasm file manager.
'''

import json
import os

import numpy as np

import tensorflow.keras as keras

from   mldisasm.benchmarks.profiling import prof
import mldisasm.io.log               as     log
from   mldisasm.io.token_list        import TokenList

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

    def _qualify(self, *args):
        '''
        Qualify a path.
        '''
        return os.path.join(self._data_dir, *args)

    ############################################################################
    # CONFIG                                                                   #
    ############################################################################

    def _qualify_config(self):
        '''
        Get the qualified filename of a configuration file.
        '''
        return self._qualify(FileManager._config_name)

    def _open_config(self, *args, **kwargs):
        '''
        Open configuration file.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        return open(self._qualify_config(), *args, **kwargs)

    def load_config(self, *args, **kwargs):
        '''
        Load configuration data.
        :param name: The configuration name.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        with self._open_config(*args, **kwargs) as file:
            return json.load(file)

    def save_config(self, config):
        '''
        Save a configuration to JSON.
        :param config: A configuration.
        '''
        with self._open_config('w', newline='\n') as file:
            json.dump(config, file, indent=4, )
            file.write('\n')

    ############################################################################
    # LOG                                                                      #
    ############################################################################

    def _qualify_log(self):
        '''
        Get the qualified filename of the log.
        '''
        return self._qualify(FileManager._log_name)

    @property
    def log_file_path(self):
        '''
        Get the path to the log file.
        '''
        return self._qualify_log()

    def open_log(self, *args, **kwargs):
        '''
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        return open(self._qualify_log(), 'w', *args, **kwargs)

    ############################################################################
    # MODEL                                                                    #
    ############################################################################

    def _qualify_model(self, name):
        '''
        Get the qualified filename of a model.
        '''
        return self._qualify(name, FileManager._model_name)

    def load_model(self, name):
        '''
        Load a model.
        :param name: The model name.
        :returns: The loaded model.
        '''
        return keras.models.load_model(self._qualify_model(name))

    def save_model(self, model, name):
        '''
        Save a model.
        :param model: The model to save.
        '''
        model.save(self._qualify_model(name))

    ############################################################################
    # TOKENS                                                                   #
    ############################################################################

    def _qualify_tokens(self):
        '''
        Get the qualified name of a token list.
        '''
        return self._qualify(FileManager._tokens_name)

    def load_tokens(self, *args, **kwargs):
        '''
        Load tokens list.
        :param args: Extra arguments for TokenList().
        :param kwargs: Keyword arguments for TokenList().
        '''
        return TokenList(self._qualify_tokens(), *args, **kwargs)

    ############################################################################
    # TRAINING                                                                 #
    ############################################################################

    def _qualify_training(self, name):
        '''
        Get the qualified filename of a training set.
        '''
        return self._qualify(name, FileManager._training_name)

    def _qualify_training_raw(self, name):
        '''
        Get the qualified filename of a training set.
        '''
        return self._qualify(name, FileManager._training_raw_name)

    def _qualify_validation(self, name):
        '''
        Get the qualified filename of a validation set.
        '''
        return self._qualify(name, FileManager._validation_name)

    def load_training(self, name, max_records=np.inf, block_size=65536):
        '''
        Load (up to) an entire JSON training set into memory at once.
        :param name: The model name.
        :param max_records: The maximum number of records to load. Default: infinity, which means load everything.
        :param block_size: The amount of data to read at once.
        :returns: A tuple of the training inputs and targets.
        '''
        X = []
        y = []
        i = 0
        with open(self._qualify_training(name), 'r') as file, prof(
            'Loaded training set ({} records)',
            lambda: i,
            resources=['time','memory']
        ):
            # Read the file in blocks until we find up to max_records lines.
            data      = ''
            num_lines = 0
            while True:
                block = file.read(block_size)
                if not block:
                    break
                data += block
                if data.count('\n') > max_records:
                    break
            lines     = data.split('\n')
            num_lines = min(len(lines), max_records)
            if num_lines % 2 != 0:
                log.warning(
                    'An even number of training examples is required but {} were loaded, '
                    'the last example will not be used'.format(num_lines)
                )
                num_lines -= 1
            # Preallocate buffers.
            X = [None]*num_lines
            y = [None]*num_lines
            # Fill buffers.
            for line in lines:
                record = json.loads(line)
                X[i] = record[0]
                y[i] = record[1]
                i += 1
                if i >= num_lines:
                    break
        return X[:i], y[:i]

    def yield_training(self, name, batch_size, block_size=65536):
        '''
        Yield training samples in batches.
        '''
        with self.open_training(name) as file:
            while True:
                with prof('Loaded batch'):
                    # Load blocks until we have more than batch_size lines. This ensures that the block doesn't end
                    # partway through the last record in the batch. The last batch might contain fewer than batch_size
                    # records; in this case, file.read() will return an empty block and we will process however many
                    # records we have, all of which will be complete.
                    file_pos = file.tell()
                    data = ''
                    while data.count('\n') <= batch_size:
                        block = file.read(block_size)
                        if not block:
                            break
                        data += block
                    if not data:
                        break
                    # Take up to `batch_size` lines and drop any remaining.
                    lines     = data.split('\n')[:batch_size]
                    num_lines = len(lines)
                    # Each line contains a pair of JSON arrays. Decode, decompose and save in X and y.
                    X = [None]*num_lines
                    y = [None]*num_lines
                    for i in range(num_lines):
                        try:
                            record = json.loads(lines[i])
                        except json.JSONDecodeError as e:
                            print('\'{}\'\n\'{}\''.format(record, lines[i]))
                            raise e from None
                        assert len(record) == 2
                        X[i], y[i] = record
                    # Rewind the file to the end of the last record that we actually used.
                    file_pos += sum(map(lambda x: len(x) + 1, lines))
                    file.seek(file_pos)
                    # Yield the batch.
                    yield X, y

    def open_training(self, name, *args, **kwargs):
        '''
        Open training set file.
        :param name: The name of the training set.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open(). Note: Any 'newline' key will be overridden with the value of '\n'.
        :returns: An open handle to the training set file.
        '''
        kwargs['newline'] = '\n'
        return open(self._qualify_training(name), *args, **kwargs)

    def open_training_raw(self, name, *args, **kwargs):
        '''
        Open training set file (raw file handle).
        :param name: The name of the training set.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        :returns: An open handle to the training set file.
        '''
        return open(self._qualify_training_raw(name), 'r', *args, **kwargs)

    def open_training_pp(self, name, *args, **kwargs):
        '''
        Open training set file (preprocessed file handle).
        :param name: The name of the training set.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        :returns: An open handle to the training set file.
        '''
        return open(self._qualify_training(name), 'w', *args, **kwargs)

    _log_name          = 'mldisasm.log'    # Log filename.
    _config_name       = 'config.json'     # Config filename.
    _model_name        = 'model.hdf5'      # Model filename.
    _training_name     = 'training.json'   # Preprocessed training set filename.
    _training_raw_name = 'rawtraining.csv' # Raw training set filename.
    _validation_name   = 'validation.json' # Validation training set filename.
    _tokens_name       = 'tokens.list'     # Token list filename.
