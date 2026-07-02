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

"""Compute correlation between opcode and disassembly lengths in a training CSV.

Usage: {0} <training.csv>
"""

import sys

import matplotlib.pyplot as plt
import numpy as np


def compute_correlation(path):
    """Compute Pearson correlation between opcode byte count and disasm char count."""
    X_lens = []
    y_lens = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or '|' not in line:
                continue
            opcode, disasm = line.split('|', 1)
            X_lens.append(len(opcode) // 2)  # hex chars → byte count
            y_lens.append(len(disasm))
            if i % 25000 == 0 and i > 0:
                sys.stdout.write('.')
                sys.stdout.flush()
    sys.stdout.write('\n')
    pearson = np.corrcoef(X_lens, y_lens)[0, 1]
    return np.array(X_lens), np.array(y_lens), pearson


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    print(f'Loading {path}...')
    X, y, coeff = compute_correlation(path)
    print(f'Pearson correlation: {coeff:.4f}')
    print(f'Records: {len(X)}, X range: [{X.min()}, {X.max()}], y range: [{y.min()}, {y.max()}]')

    plt.clf()
    plt.xlabel('Opcodes (bytes)')
    plt.ylabel('Disassembly (chars)')
    plt.scatter(X, y, alpha=0.3, s=1)
    plt.savefig('training-correlation.png')
    print('Saved training-correlation.png')
