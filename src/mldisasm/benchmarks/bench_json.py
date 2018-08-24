#!/usr/bin/env python3

'''
Benchmark JSON libraries.
'''

import importlib
import sys

from mldisasm.benchmarks.common import *
from mldisasm.io.file_manager   import FileManager
from mldisasm.util.prof         import prof

MAX_RECORDS = 1000

MODULES = ['json','ujson','simplejson']

def bench_with_module(module, file):
    '''
    Benchmark ujson or json module.
    '''
    #file.seek(0)
    count = 0
    with prof(None, resources=['time','memory'], log_level=None) as p:
        for line in file:
            module.loads(line)
            if count >= MAX_RECORDS:
                break
            count += 1
    return p.profilers['time'].compute_delta()

def run():
    '''
    Run benchmark.
    '''
    file_mgr = FileManager()
    results  = []
    with file_mgr.open_training(MODEL_NAME) as file:
        for name in MODULES:
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError:
                print('NB: Not benchmarking {}: Module not found'.format(name), file=sys.stderr)
                continue
            times = []
            for _ in range(BENCH_ITER):
                time = bench_with_module(module, file)
                time = round(time*1000)
                times.append(time)
            assert len(times) == BENCH_ITER
            results.append(BenchmarkResult(name, times, unit='ms'))
    return results
