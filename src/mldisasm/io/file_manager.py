#!/usr/bin/env python3

'''
MLDisasm file manager.
'''

import json
import pickle
import os

from mldisasm.io.training_set import TrainingSet

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

    def load_config(self, name, *args, **kwargs):
        '''
        Load configuration data.
        :param name: The configuration name.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        with self._open_config(name, *args, **kwargs) as file:
            return json.load(file)

    def open_log(self, *args, **kwargs):
        '''
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        return open(self._qualify_log(), *args, **kwargs)

    def open_model(self, name, *args, **kwargs):
        '''
        Open model.
        :param name: The model name.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        :returns: An open handle to the model file.
        '''
        return open(self._qualify_model(name), FileManager._model_mode, *args, **kwargs)

    def open_training(self, name, *args, **kwargs):
        '''
        Open training set file.
        :param name: The name of the training set.
        :param args: Extra arguments for TrainingSet.__init__().
        :param kwargs: Keyword arguments for TrainingSet.__init__().
        :returns: An open handle to the training set file.
        '''
        return TrainingSet(self._qualify_training(name), *args, **kwargs)

    def save_model(self, model, name, *args, **kwargs):
        '''
        Save a model.
        :param model: The model to save.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        with open(self._qualify_model(name), FileManager._model_mode, *args, **kwargs) as file:
            pickle.dump(model, file)

    def _open_config(self, name, *args, **kwargs):
        '''
        Open configuration file.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open().
        '''
        return open(self._qualify_config(name), *args, **kwargs)

    def _qualify_training(self, name):
        '''
        Get the qualified filename of a training set.
        '''
        return self._qualify(name, FileManager._training_name)

    def _qualify_model(self, name):
        '''
        Get the qualified filename of a model.
        '''
        return self._qualify(name, FileManager._model_name)

    def _qualify_config(self, name):
        '''
        Get the qualified filename of a configuration file.
        '''
        return self._qualify(name, FileManager._config_name)

    def _qualify_log(self):
        '''
        Get the qualified filename of the log.
        '''
        return self._qualify(FileManager._log_name)

    def _qualify(self, *args):
        '''
        Qualify a path.
        '''
        return os.path.join(self._data_dir, *args)

    _model_mode      = 'rb'             # Model open() mode.
    _log_name        = 'mldisasm.log'   # Log filename.
    _config_name     = 'config.json'    # Config filename.
    _model_name      = 'model.pkl'      # Model filename.
    _training_name   = 'training.csv'   # Training set filename.
