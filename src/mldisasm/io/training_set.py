#!/usr/bin/env python3

'''
MLDisasm training set.
'''

import json
import threading
import time

import tensorflow as tf

from   mldisasm.benchmarks.profiling import prof
import mldisasm.io.log               as     log

class TrainingSet:
    '''
    Allows iterating over training set data.
    '''
    def __init__(self, file, batch_size,  seq_len):
        '''
        Initialise TrainingSet.
        :param file: A path or handle to the file containing the training set.
        :param batch_size: Size of a batch of training examples. If this is not a clean divisor of the total training
        set size, the last batch will be smaller than the others.
        :param seq_len: The sequence length.
        '''
        with prof('Opened training set'):
            if batch_size < 1:
                batch_size = 1
            if isinstance(file, str):
                file = open(file, 'r')
            self._file        = file
            self._file.seek(0)
            self._batch_size  = batch_size
            self._seq_len     = seq_len
            self._batch_num   = 1
            self._iter_flag   = False
            self._worker      = TrainingSet.Worker(self._file, self._batch_size)
            self._worker.start()

    def __del__(self):
        '''
        Stop worker thread before destroying object.
        '''
        self._worker.join()

    def __iter__(self):
        '''
        Get an iterator to the training set.
        '''
        # Don't restart the worker on the first call to __iter__. Usually a batch has already been loaded and calling
        # restart() discards it and loads it again. If we have called __iter__ before, then we do want to discard loaded
        # batches and start iterating from the beginning.
        if self._iter_flag:
            self._worker.restart()
        self._iter_flag = True
        self._batch_num = 1
        return self

    def __next__(self):
        '''
        Get the next batch of records. Blocks until the batch is available.
        :returns: A tuple of (examples,targets)
        '''
        log.info('Batch {}'.format(self._batch_num))
        with prof('Processed batch'):
            batch     = next(self._worker) # This can raise StopIteration; if so, we let the caller catch it.
            batch_len = len(batch)
            examples  = [None]*batch_len
            targets   = [None]*batch_len
            for i in range(batch_len):
                assert batch[i] is not None
                assert len(batch[i]) == 2
                examples[i] = batch[i][0]
                targets[i]  = batch[i][1]
        # Build tensors on CPU.
        with tf.device('/cpu:0'):
            return tf.stack(examples), tf.stack(targets)

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

        @property
        def active(self):
            '''
            Indicate whether the worker is active or has finished processing the file.
            '''
            return self._active

        def restart(self):
            '''
            Restart the thread and begin reading from the beginning of the file.
            '''
            with self._lock:
                self._file.seek(0)
                self._active      = True
                self._batch_ready = False
                self._index       = 0

        def run(self):
            '''
            Run the thread's main loop.
            '''
            while self._active:
                # Wait until the current batch is used.
                while self._active and self._batch_ready:
                    time.sleep(0)
                # Try to load a new batch.
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
            if self._index < self._batch_size:
                with self._lock, prof('Loaded {} records', lambda: self._index):
                    while self._index < self._batch_size:
                        record = self._file.readline()
                        if not record:
                            break
                        self._records[self._index] = json.loads(record)
                        self._index += 1
                    if self._index < self._batch_size:
                        # Don't iterate any more if we hit EOF before finishing the batch.
                        self._active = False
                    if self._index == 0:
                        # Stop iterating now if current batch empty.
                        raise StopIteration
                    self._batch_ready = True

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
            # Copy the batch and return it.
            with self._lock:
                assert self._index > 0
                records = list(self._records[:self._index])
                assert len(records) > 0
                self._index = 0
                self._batch_ready = False
                return records
