#!/usr/bin/env python3

'''
MLDisasm training set.
'''

import threading
import time

import tensorflow as tf


from   mldisasm.benchmarks.profiling import prof
from   mldisasm.io.codec             import BYTEORDER
import mldisasm.io.log as log

# Training set delimiter.
DELIMITER = '|'

# Training set encoding.
ENCODING = 'ascii'

# Delimiter decoded to bytes.
DELIMITER_BYTES = bytes(DELIMITER, ENCODING)

# Every two hex chars represent one byte.
CHARS_PER_BYTE = 2

class TrainingSet:
    '''
    Allows iterating over training set data.
    '''

    def __init__(self, file, batch_size, x_encoder, y_encoder):
        '''
        Initialise TrainingSet.
        :param file: A path or handle to the file containing the training set.
        :param batch_size: Size of a batch of training examples. If this is not a clean divisor of the total training
        set size, the last batch will be smaller than the others.
        :param x_encoder: A callable which encodes the input bytes into a tensor. Default is to use one-hot encoding.
        :param y_encoder: A callable which encodes the target string into a tensor. Default is to use one-hot encoding.
        '''
        if batch_size < 1:
            batch_size = 1
        if isinstance(file, str):
            file = open(file, 'r')
        p = prof('Opened training set')
        self._file        = file
        self._num_records = len([_ for _ in self._file])
        self._file.seek(0)
        self._batch_size  = batch_size
        self._x_encoder   = x_encoder
        self._y_encoder   = y_encoder
        self._record_num  = 1
        self._worker      = TrainingSet.Worker(self._file, self._batch_size)
        self._worker.start()

    def __del__(self):
        '''
        Stop worker thread before destroying object.
        '''
        self._worker.join()

    def __len__(self):
        '''
        Get the number of records in the training set.
        '''
        return self._num_records

    @property
    def num_batches(self):
        '''
        Get the number of batches in the training set.
        '''
        return int(self._num_records/self._batch_size)

    def __iter__(self):
        '''
        Get an iterator to the training set.
        '''
        self._worker.restart()
        return self

    def __next__(self):
        '''
        Get the next batch of records. Blocks until the batch is available.
        :returns: A tuple of (example,targets)
        '''
        p         = prof('Processed batch')
        batch     = next(self._worker)
        batch_len = len(batch)
        examples  = [None]*batch_len
        targets   = [None]*batch_len
        for i in range(batch_len):
            record = batch[i][:-1] # Cut off the newline.
            elems  = record.split(DELIMITER)
            if len(elems) != 2:
                raise ValueError('training:{}: Bad training example: {}'.format(self._record_num, batch[i]))
            try:
                # Append encoded tensors.
                target       = elems[1]
                opcode       = int(elems[0], 16)
                opcode_len   = int(0.5 + len(elems[0])/CHARS_PER_BYTE)
                opcode_bytes = opcode.to_bytes(opcode_len, BYTEORDER)
                examples[i]  = self._x_encoder.encode(opcode_bytes)
                targets[i]   = self._y_encoder.encode(target)
                self._record_num += 1
            except ValueError as e:
                # Re-raise the exception with a better message. 'raise ... from None' tells Python not to produce output
                # like 'while handling ValueError, another exception occurred'.
                raise ValueError('training:{}: {}: {}'.format(
                    self._record_num,
                    record,
                    str(e)
                )) from None
        # Convert into tensors with fixed shape.
        return (
            tf.reshape(tf.stack(examples), (batch_len, examples[0].shape[0], 1)),
            tf.reshape(tf.stack(targets),  (batch_len, targets[0].shape[0],  1))
        )

    class Worker(threading.Thread):
        '''
        Load training data in a subprocess.
        '''
        def __init__(self, file, batch_size):
            '''
            Initialise Worker.
            :param file: An open file containing a training set.
            :param batch_size: The size of a batch.
            '''
            super().__init__()
            assert hasattr(file, 'readable')
            assert file.readable()
            self._file        = file
            self._batch_size  = batch_size
            self._active      = True
            self._lock        = threading.Lock()
            self._records     = [None]*self._batch_size
            self._index       = 0
            self._batch_ready = False

        def restart(self):
            '''
            Restart the thread and begin reading from the beginning of the file.
            '''
            with self._lock:
                self._file.seek(0)
                self._index  = 0
                self._active = True

        def run(self):
            '''
            Run the thread's main loop.
            '''
            while self._active:
                try:
                    self.load_batch()
                except StopIteration:
                    self._active = False

        def join(self):
            '''
            Tell the thread to stop and wait until it does.
            '''
            self._active = False
            super().join()

        def load_batch(self):
            '''
            Load a batch of records.
            :raises StopIteration: If there are no more examples to load. There may still be some saved in the batch.
            '''
            with self._lock:
                num_records = len(self._records)
                num_to_load = num_records - self._index
                if num_to_load > 0:
                    p = prof('Loaded {} records', num_to_load)
                    while self._index < len(self._records):
                        record = self._file.readline()
                        if not record:
                            raise StopIteration
                        self._records[self._index] = record
                        self._index += 1
                    self._batch_ready = True
                    del p
        def __iter__(self):
            '''
            Get an iterator to the worker.
            '''
            return self

        def __next__(self):
            '''
            Get the next batch of records. May block until a batch is ready.
            :raises StopIteration: If there are no more records in the file.
            '''
            # Raise StopIteration if we've run out of records and the batch is empty.
            if not self._active and not self._batch_ready:
                raise StopIteration
            # Block until a batch is loaded.
            while not self._batch_ready:
                time.sleep(0)
            records = []
            with self._lock:
                self._index = 0
                self._batch_ready = False
                records = list(self._records)
            return records
