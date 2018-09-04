#!/usr/bin/env python3

'''
MLDisasm basic training stuff.
'''

import numpy as np

import tensorflow as tf

from mldisasm.model import Disassembler
from mldisasm.util  import prof

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

def _extract_metrics(model, metrics):
    if model.metrics_names == ['acc','loss']:
        acc, loss = metrics
    elif model.metrics_names == ['loss','acc']:
        loss, acc = metrics
    else:
        raise ValueError('Unrecognised metrics names: {}'.format(','.join(model.metrics_names)))
    return acc, loss

def _running_average(value, sample, count):
    return value*(count - 1)/count + sample/count

def _train_fold(X, y, params, train, test, fold_num):
    acc = -np.inf
    loss = np.inf
    with prof(
        'Fold {}/{}: acc={}%, loss={}', fold_num, params['kfolds'], lambda: acc*100, lambda: loss,
        log_level='info',
        start_msg='Fold {}/{}'.format(fold_num, params['kfolds'])
    ):
        # Train and cross-validate.
        X_train, y_train, X_test, y_test = cv_split(X, y, train, test)
        model   = Disassembler(**params)
        history = model.fit(
            [X_train,y_train],
            tf.manip.roll(y_train, 1, 1),
            validation_data  = [[X_test,y_test], tf.manip.roll(y_test, 1, 1)],
            steps_per_epoch  = 1,
            validation_steps = 1,
            epochs           = params['epochs']
        )
        acc  = history.history['val_acc'][-1]
        loss = history.history['val_loss'][-1]
        return acc, loss

def kfolds_train(X, y, params):
    '''
    Perform training with k-folds cross-validation.
    :param X: The training inputs.
    :param y: The training outputs.
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
