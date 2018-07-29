#!/usr/bin/env python3

'''
MLDisasm file manager.
'''

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

    def open_model(self, name, *args, **kwargs):
        '''
        Open model.
        :param name: The model name.
        :returns: An open handle to the model file.
        '''
        return open(self._qualify_model(name), 'rb', *args, **kwargs)

    def open_training(self, name, *args, **kwargs):
        '''
        Open training set file.
        :param name: The name of the training set.
        :returns: An open handle to the training set file.
        '''
        return TrainingSet(self._qualify_training(name), *args, **kwargs)

    def save_model(self, model, name):
        '''
        Save a model.
        :param model: The model to save.
        '''
        with open(self._qualify_model(name), 'rb') as file:
            pickle.dump(model, file)

    def _qualify_training(self, name):
        '''
        Qualify a training set filename.
        '''
        return os.path.join(self._data_dir, name, 'training.csv')

    def _qualify_model(self, name):
        '''
        Qualify a model filename.
        '''
        return os.path.join(self._data_dir, name, 'model.pkl')
