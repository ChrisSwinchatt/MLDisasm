#!/usr/bin/env python3

'''
Test TrainingSet.
'''

from   io import StringIO
import json

import numpy as np

import pytest

import tensorflow as tf

from mldisasm.io.training_set import TrainingSet
from mldisasm.tests.common    import *

BATCH_SIZE = 100
SEQ_LEN    = 100

def random_float_array():
    '''Generate an array of random floats.'''
    return np.random.random((SEQ_LEN,1)).tolist()

def random_int_array():
    '''Generate an array of random integers.'''
    return np.random.randint(0, 1000, (SEQ_LEN,1)).tolist()

class TestTrainingSet:
    '''
    TrainingSet test case.
    '''
    def setup(self):
        '''
        Initialise test case.
        '''
        # Generate TEST_ITERATIONS*BATCH_SIZE random pairs of lists, convert to JSON and stream to TrainingSet.
        # It would be faster to let NumPy generate BATCH_SIZE sized arrays (in native code) but TrainingSet expects one
        # pair of JSON arrays per line.
        # We also save the generated lists as records and tensors so we can compare the decoded JSON/tensors with the
        # originals later.
        self.stream = StringIO()
        self.records = []
        self.tensors = []
        for _ in range(TEST_ITERATIONS):
            examples = []
            targets  = []
            for __ in range(BATCH_SIZE):
                example = random_float_array()
                target  = random_int_array()
                self.stream.write(json.dumps([example, target]))
                self.stream.write('\n')
                self.records.append([example, target])
                examples.append(example)
                targets.append(target)
            self.tensors.append((
                tf.reshape(tf.convert_to_tensor(examples), (BATCH_SIZE, SEQ_LEN, 1)),
                tf.reshape(tf.convert_to_tensor(targets),  (BATCH_SIZE, SEQ_LEN, 1))
            ))
        self.tset = TrainingSet(self.stream, BATCH_SIZE, SEQ_LEN)

    def test_worker_next(self):
        '''
        Test TrainingSet.Worker.__next__().
        '''
        enter_test(self.test_worker_next)
        records_iter = iter(self.records)
        for _ in range(TEST_ITERATIONS):
            # Load a batch and verify that it contains the same data that we input during setup.
            try:
                batch = next(self.tset._worker)
            except StopIteration:
                pytest.fail('StopIteration raised before end of test')
            else:
                assert len(batch) == BATCH_SIZE
                for decoded in batch:
                    try:
                        original = next(records_iter)
                    except StopIteration:
                        pytest.fail('StopIteration raised before end of test')
                    else:
                        assert decoded == original
                leave_test_iter()
        leave_test()

    def test_next(self):
        '''
        Test TrainingSet.__next__().
        '''
        enter_test(self.test_next)
        with tf.Session():
            tensors_iter = iter(self.tensors)
            for _ in range(TEST_ITERATIONS):
                # Load a pair of tensors and verify that they contains the same data that we input during setup.
                try:
                    X_decoded,  y_decoded  = next(self.tset)
                    X_original, y_original = next(tensors_iter)
                except StopIteration:
                    pytest.fail('StopIteration raised before end of test')
                else:
                    X_decoded  = X_decoded.eval().tolist()
                    y_decoded  = y_decoded.eval().tolist()
                    X_original = X_original.eval().tolist()
                    y_original = y_original.eval().tolist()
                    assert X_decoded == X_original
                    assert y_decoded == y_original
                    leave_test_iter()
        leave_test()
