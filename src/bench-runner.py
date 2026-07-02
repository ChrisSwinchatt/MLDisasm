#!/usr/bin/env python3

# Copyright (C) 2018,2021,2026 Chris Swinchatt
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
