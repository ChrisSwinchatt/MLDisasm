#!/usr/bin/env python3

# Copyright (C) 2018,2021 Chris Swinchatt
# 
# MLDisasm is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
# 
# MLDisasm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with MLDisasm.  If not, see <http://www.gnu.org/licenses/>.

'''
Benchmark JSON libraries.
'''

import importlib
import sys

from mldisasm.benchmarks.common import *
from mldisasm.io.file_manager   import FileManager
from mldisasm.util              import prof

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
