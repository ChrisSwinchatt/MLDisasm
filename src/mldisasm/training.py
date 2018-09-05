#!/usr/bin/env python3

'''
MLDisasm basic training stuff.
'''

import random

import numpy as np

import tensorflow               as tf
import tensorflow.keras.backend as K

from mldisasm.model import Disassembler
from mldisasm.util  import log, prof

def grids(params):
    '''
    Generate parameter sets from a parameter grid.
    :param params: The parameter grid. A dictionary whose values are lists.
    :yields: One parameter set for each possible combination of values in the grid.
    '''
    keys, values = zip(*sorted(params.items()))
    sizes        = [len(v) for v in values]
    total_size   = np.product(sizes)
    for i in range(total_size):
        grid = dict()
        for k, vs, size in zip(keys, values, sizes):
            i, off  = divmod(i, size)
            grid[k] = vs[off]
        yield grid

def parameter_grid(params):
    '''
    Return a list of parameter sets from a parameter grid.
    :param params: The parameter grid. A dictionary whose values are lists.
    :returns: A list of every possible combination of values in the grid.
    '''
    return list(grids(params))

def kfolds(n, k=3, shuffle=True):
    '''
    Generate indices to fold a training set into k subsets and yield the subsets.
    :param n: The number of samples in the tensors.
    :param k: The number of folds to generate. Must be greater than 1.
    :param shuffle: Whether to shuffle the order of each fold.
    :yields: A tuple of the training and testing indices for each of k different folds, each training on (k - 1)/k of
    the samples and testing on 1/k samples.
    '''
    m = n//k
    for i in range(k):
        # Generate the indices for the fold. The test fold is everything from [i*m, (i + 1)*m); the training fold is
        # everything else.
        start = i*m
        end   = (i + 1)*m
        test  = np.array(list(range(start, end)),                         dtype=np.int32)
        train = np.array(list(range(0,     start)) + list(range(end, n)), dtype=np.int32)
        if shuffle:
            np.random.shuffle(train)
            np.random.shuffle(test)
        yield train, test

def cv_split(X, y, train, test):
    '''
    Split tensors for cross-validation.
    :param X: The input tensor.
    :param y: The target tensor.
    :param train: The training indices.
    :param test: The testing indices.
    :returns: A tuple of (X_train,y_train,X_test,y_test).
    '''
    return tf.gather(X, train), tf.gather(y, train), tf.gather(X, test), tf.gather(y, test)

def train_model(X, y, params, train=None, test=None, retrain=None):
    '''
    Train a model.
    :param X: The training inputs.
    :param y: The training targets.
    :param train: If given, a list of indices to be used for training.
    :param test: If given, a list of indices to be used for cross validating.
    :param retrain: A model to retrain or None to train a new model..
    :returns: A tuple of (model,acc,loss), where acc and loss are the *final* training accuracy and loss (if train and
    test are None) or the *final* cross-validation accuracy and loss (if neither train nor test is None).
    :raises ValueError: If one of train and test is None, but not both of them.
    '''
    # Get validation data if we're cross-validating.
    validation_data = None
    if train is not None and test is not None:
        X, y, X_test, y_test = cv_split(X, y, train, test)
        validation_data      = [[X_test,y_test],tf.manip.roll(y_test, 1, 1)]
    elif train is None != test is None:
        raise ValueError('Either none or both of train and test should be None')
    # Retrain existing model or create a new one.
    model = retrain
    if model is None:
        model = Disassembler(**params)
    # Train the model.
    log.info('Training model with parameters {}'.format(params))
    K.set_learning_phase(1)
    num_epochs = params['epochs']
    history    = model.fit(
        [X, y],
        tf.manip.roll(y, 1, 1),
        epochs           = num_epochs,
        steps_per_epoch  = 1,
        validation_data  = validation_data,
        validation_steps = 0 if validation_data is None else 1
        #callbacks        = [keras.callbacks.TensorBoard(log_dir=os.path.join(FileManager.default_data_dir, 'tb'))]
    )
    # Extract metrics.
    acc, loss = 'acc', 'loss'
    if validation_data is not None:
        acc  = 'val_acc'
        loss = 'val_loss'
    return model, history.history[acc][-1], history.history[loss][-1]

def _train_fold(X, y, params, train, test, fold_num):
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)
    acc = -np.inf
    loss = np.inf
    with prof(
        'Fold {}/{}: acc={}%, loss={}', fold_num, params['kfolds'], lambda: acc*100, lambda: loss,
        log_level='info',
        start_msg='Fold {}/{}'.format(fold_num, params['kfolds'])
    ):
        # Train and cross-validate.
        _, acc, loss = train_model(X, y, params, train, test)
        return acc, loss

def _running_average(value, sample, count):
    return value*(count - 1)/count + sample/count

def kfolds_train(X, y, params):
    '''
    Perform training with k-folds cross-validation.
    :param X: The training inputs.
    :param y: The training targets.
    :param params: The model parameters for creating a model.
    :param num_batches: The maximum number of batches or None.
    :returns: A tuple of the average accuracy and loss over all folds.
    '''
    avg_acc  = 0
    avg_loss = 0
    fold_num = 1
    for train, test in kfolds(int(X.shape[0]), params['kfolds'], params['shuffle']):
        acc, loss = _train_fold(X, y, params, train, test, fold_num)
        avg_acc   = _running_average(avg_acc,  acc,  fold_num)
        avg_loss  = _running_average(avg_loss, loss, fold_num)
        fold_num += 1
    return avg_acc, avg_loss
