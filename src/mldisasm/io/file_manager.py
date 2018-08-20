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

class TrainingSetError(Exception):
    '''
    Exception raised when there is an error loading the training set.
    '''
    pass

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
        model.save(self._qualify_model(name), overwrite=True)

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

    def _do_load_training(self, file, block_size, max_records):
        '''
        Load up to `max_records` from `file` using blocks of `block_size` bytes.
        :returns: A tuple of the training inputs and labels.
        '''
        num_lines = 0
        with prof('Loaded batch ({} records)', lambda: num_lines, resources=['time','memory']):
            # Blocks are likely to end partway through a record, so we read more data than we need and discard any
            # excess records. We save the file position so we can calculate the amount of data actually used and seek to
            # the beginning of the discarded record(s) for the next read.
            file_pos  = file.tell()
            data      = ''
            num_lines = 0
            while data.count('\n') <= max_records:
                block = file.read(block_size)
                if not block:
                    break
                data += block
            # Split on newline and discard records above the maximum.
            lines     = data.split('\n')
            num_lines = min(len(lines), max_records)
            if num_lines % 2 != 0:
                log.warning(
                    'An even number of training examples is required but {} were loaded, '
                    'the last example will not be used'.format(num_lines)
                )
                num_lines -= 1
            # Process the records and rewind to account for any extra records read.
            X = [None]*num_lines
            y = [None]*num_lines
            len_lines = 0
            for i in range(num_lines):
                len_lines += len(lines[i]) + 1 # +1 to account for newline stripped by str.split().
                try:
                    X[i], y[i] = json.loads(lines[i])
                except (TypeError,ValueError,json.JSONDecodeError) as e:
                    # Three exceptions can be raised when decoding training samples:
                    #  * TypeError: if json.loads doesn't return an iterable.
                    #  * ValueError: if json.loads returns an object with too many or too few values to
                    #    unpack.
                    #  * JSONDecodeError: if the line doesn't contain a valid JSON object.
                    log.debug(lines[i])
                    raise TrainingSetError('Training set error: {}'.format(' '.join(e.args))) from e
            file.seek(file_pos + len_lines)
            return X, y

    def load_training(self, name, block_size=65536, max_records=np.inf):
        '''
        Load (up to) an entire JSON training set into memory at once.
        :param name: The model name.
        :param max_records: The maximum number of records to load. Default: infinity, which means load everything.
        :param block_size: The amount of data to read at once.
        :returns: A tuple of the training inputs and targets.
        '''
        with self.open_training(name) as file:
            return self._do_load_training(file, block_size, max_records)

    def yield_training(self, name, batch_size, block_size=65536, max_records=np.inf):
        '''
        Yield training samples in batches.
        :param name: The name of the training set.
        :param batch_size: The number of records in each batch. The actual size of a batch may be smaller than
        batch_size if there are fewer records in the file, or max_records is smaller than batch_size.
        :param block_size: How many bytes to load from the file at once. This can effect performance, but not the number
        of records returned - more than one block will be read if necessary.
        :param max_records: The maximum number of records to load. Overrides batch_size if max_records is smaller.
        Default value is infinity, which means load up to batch_size or the entire file, whichever is smaller.
        :returns: A tuple of the training inputs and targets.
        '''
        with self.open_training(name) as file:
            while True:
                batch_size = min(batch_size, max_records)
                X, y       = self._do_load_training(file, block_size, batch_size)
                assert len(X) == len(y)
                if not X:
                    break
                max_records -= len(X)
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
