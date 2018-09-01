#!/usr/bin/env python3

'''
MLDisasm basic training stuff.
'''

import numpy as np

import tensorflow as tf

from mldisasm.model import trainable_disassembler
from mldisasm.util  import prof, refresh_graph

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
        idx  = i
        for k, vs, size in zip(keys, values, sizes):
            idx, off   = divmod(idx, size)
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
        train = list(range(i*m)) + list(range((i + 1)*m, k*m))
        test  = list(range(i*m, (i + 1)*m))
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

def train_batch(model, X, y, batch_num, params=None, refresh_step=50):
    '''
    Train a model on one batch of samples.
    :param model: The model.
    :param X: The input tensor.
    :param y: The target tensor.
    :param batch_num: The one-based batch number.
    :param params: Parameters to rebuild model when refreshing the graph, or None to disable graph refreshing and
    rebuilding model.
    :param refresh_step: How many batches to wait before refreshing the graph. Ignored if params is None.
    :returns: A tuple of the accuracy, loss and model (always returned but only useful if the graph was refreshed).
    '''
    acc, loss = -np.inf, np.inf
    with prof('Batch {}: acc={}%, loss={}', batch_num, lambda: acc*100, lambda: loss, log_level='info'):
        metrics = model.train_on_batch([X, y], tf.manip.roll(y, 1, 1))
        acc, loss = _extract_metrics(model, metrics)
        if params is not None and batch_num >= refresh_step and batch_num%refresh_step == 0:
            model = refresh_graph(model, build_fn=trainable_disassembler, **params)
    return acc, loss, model

def test_batch(model, X, y):
    '''
    Test a model on a batch.
    :param model: The model.
    :param X: The input tensor.
    :param y: The target tensor.
    :returns: A tuple of the accuracy and loss.
    '''
    metrics = model.test_on_batch([X, y], tf.manip.roll(y, 1, 1))
    return _extract_metrics(model, metrics)

def _running_average(value, sample, count):
    return value*(count - 1)/count + sample/count

def train_epoch(model, batches, epoch, num_epochs, params=None, num_batches=0):
    '''
    Perform training for one epoch of samples.
    :param model: The model to train.
    :param batches: An iterable (e.g. a generator) of batches of training samples.
    :param epoch: The current epoch being trained.
    :param num_epochs: The number of epochs being trained.
    :param params: Parameters to rebuild model when refreshing the graph, or None to disable graph refreshing and
    rebuilding model.
    :returns: A tuple of the average accuracy and loss during training and the model, in case it has been rebuilt.
    '''
    avg_acc  = 0
    avg_loss = 0
    with prof(
        'Epoch {}/{}: acc={}%, loss={}', epoch, num_epochs, lambda: acc*100, lambda: loss,
        log_level='info',
        start_msg='Epoch {}/{}'.format(epoch, num_epochs)
    ):
        batch_num = 1
        for X, y in batches:
            acc, loss, model = train_batch(model, X, y, batch_num, params)
            avg_acc  = _running_average(avg_acc,  acc,  batch_num)
            avg_loss = _running_average(avg_loss, loss, batch_num)
            del X, y
            batch_num += 1
            if num_batches > 0 and batch_num >= num_batches:
                break
    return avg_acc, avg_loss, model

def _train_fold(batches, params, train, test, fold_num, num_batches):
    with prof(
        'Fold {}/{}: acc={}%, loss={}', fold_num, params['kfolds'], lambda: avg_acc*100, lambda: avg_loss,
        log_level='info',
        start_msg='Fold {}/{}'.format(fold_num, params['kfolds'])
    ):
        avg_acc = 0
        avg_loss = 0
        # Clear the graph.
        refresh_graph()
        # Create and train a model over the full training set.
        model     = trainable_disassembler(**params)
        batch_num = 1
        for X, y in batches:
            # Train on the batch and extract metrics.
            X_train, y_train, X_test, y_test = cv_split(X, y, train, test)
            _, _, model = train_batch(model, X_train, y_train, batch_num)
            acc, loss   = test_batch(model,  X_test,  y_test)
            # Recompute averages.
            avg_acc     = _running_average(avg_acc,  acc,  batch_num)
            avg_loss    = _running_average(avg_loss, loss, batch_num)
            if num_batches > 0 and batch_num >= num_batches:
                break
            batch_num += 1
        return avg_acc, avg_loss

def kfolds_train(batches, params, num_batches=-1):
    '''
    Perform training with k-folds cross-validation.
    :param batches: An iterable (e.g. a generator) of batches of training samples. If this is a generator, it should be
    able to reproduce its dataset repeatedly because the generator will be iterated more than once.
    :param params: The model parameters for creating a trainable_model().
    :returns: A tuple of the average accuracy and loss over all batches and folds.
    '''
    avg_acc  = 0
    avg_loss = 0
    fold_num = 1
    for train, test in kfolds(params['batch_size'], params['kfolds'], params['shuffle']):
        acc, loss = _train_fold(batches, params, train, test, fold_num, num_batches)
        avg_acc   = _running_average(avg_acc,  acc,  fold_num)
        avg_loss  = _running_average(avg_loss, loss, fold_num)
        fold_num += 1
    return avg_acc, avg_loss
