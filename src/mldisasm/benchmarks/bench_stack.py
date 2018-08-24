#!/usr/bin/env python3

'''
Compares tf.stack, tf.convert_to_tensor and tf.concat.
'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

import ujson

from mldisasm.benchmarks.common import *
from mldisasm.io.file_manager   import FileManager
from mldisasm.util.prof         import prof

BATCH_SIZE = 100

def run():
    '''
    Run benchmark.
    '''
    tf.enable_eager_execution()
    file_mgr = FileManager()
    results  = []
    with tf.device('/cpu:0'), file_mgr.open_training(MODEL_NAME) as file:
        # Load and process batch.
        X         = [None]*BATCH_SIZE
        y         = [None]*BATCH_SIZE
        file_iter = iter(file)
        for i in range(BATCH_SIZE):
            X[i], y[i] = ujson.loads(next(file_iter))
        assert len(X) == BATCH_SIZE
        assert len(y) == BATCH_SIZE
        assert list(filter(lambda x: x is None, X)) == []
        assert list(filter(lambda x: x is None, y)) == []
        # Convert each pair of lists to tensors.
        stack_times   = []
        concat_times  = []
        convert_times = []
        # Hack: the first call (stack/concat/convert) runs significantly slower, resulting in imbalanced results. We try
        # to fix this by running an unprofiled stack(). This does seem to make the subsequent calls perform more evenly.
        # Most likely, the first call causes TF to initialise the GPU, so by calling stack() we are avoiding this.
        tf.stack(X)
        tf.stack(y)
        for _ in range(BENCH_ITER):
            # Profile tf.stack().
            with prof(None, resources=['time'], log_level=None) as p:
                tf.stack(X)
                tf.stack(y)
                time = p.profilers['time'].compute_delta()
                time = round(time*1000)
                stack_times.append(time)
            # Profile tf.concat().
            with prof(None, resources=['time'], log_level=None) as p:
                tf.concat(X, axis=0)
                tf.concat(y, axis=0)
                time = p.profilers['time'].compute_delta()
                time = round(time*1000)
                concat_times.append(time)
            # Profile tf.convert_to_tensor().
            with prof(None, resources=['time'], log_level=None) as p:
                tf.convert_to_tensor(X)
                tf.convert_to_tensor(y)
                time = p.profilers['time'].compute_delta()
                time = round(time*1000)
                convert_times.append(time)
        results.append(BenchmarkResult('stack',   stack_times))
        results.append(BenchmarkResult('concat',  concat_times))
        results.append(BenchmarkResult('convert', convert_times))
    return results
