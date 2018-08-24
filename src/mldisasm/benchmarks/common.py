#!/usr/bin/env python3

'''
Common stuff for benchmarking.
'''

import numpy as np

# Model to use when benchmarking.
MODEL_NAME = 'att'

# Number of iterations to benchmark.
BENCH_ITER  = 100

class BenchmarkResult:
    '''
    Benchmark result.
    '''
    def __init__(self, name, samples, unit='ms'):
        '''
        Initialise BenchmarkResult.
        :param name: The name of the benchmark.
        :param samples: The time samples.
        :param unit: The unit. Default is 'ms'. If None, no unit is displayed.
        '''
        self.name    = name
        self.samples = sorted(samples)
        self.unit    = '' if unit is None else unit

    @property
    def best(self):
        '''
        Get the best sample value.
        '''
        return np.min(self.samples)

    @property
    def worst(self):
        '''
        Get the worst sample value.
        '''
        return np.max(self.samples)

    @property
    def mean(self):
        '''
        Get the mean sample value.
        '''
        return np.mean(self.samples)

    @property
    def median(self):
        '''
        Get the median sample value.
        '''
        return self.samples[len(self.samples)//2]

    @property
    def total(self):
        '''
        Get the total of all samples.
        '''
        return sum(self.samples)

    def __str__(self):
        return '{:13}{:12}{:12}{:12}{:12}{:12}{:12}'.format(
            self.name,
            str(len(self.samples)),
            '{} {}'.format(self.worst,  self.unit),
            '{} {}'.format(self.mean,   self.unit),
            '{} {}'.format(self.median, self.unit),
            '{} {}'.format(self.best,   self.unit),
            '{} {}'.format(self.total,  self.unit)
        )
