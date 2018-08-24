#!/usr/bin/env python3

'''
Runs benchmarks.
'''

import pkgutil
import sys

import mldisasm.benchmarks as benchmarks

if __name__ == '__main__':
    for importer, name, is_package in pkgutil.iter_modules(benchmarks.__path__):
        if not name.startswith('bench_'):
            # Skip non-benchmarks.
            continue
        num_eq = max(0, 80 - len(name) - len('[  ]'))//2
        print('{0}[ {1} ]{0}'.format('='*num_eq, name), file=sys.stderr)
        module  = importer.find_module(name).load_module(name)
        results = module.run()
        print('NAME         SAMPLES     WORST       MEAN        MEDIAN      BEST        TOTAL', file=sys.stderr)
        for result in results:
            print(str(result), file=sys.stderr)
