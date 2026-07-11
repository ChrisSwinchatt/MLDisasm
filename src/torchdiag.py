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
MLDisasm PyTorch diagnostics.
'''

if __name__ == '__main__':
    print('*** Starting PyTorch diagnostics')

    import torch

    # Print PyTorch version info.
    print(f'\n*** PyTorch version: {torch.__version__}')
    print(f'*** CUDA available: {"yes" if torch.cuda.is_available() else "no"}')

    if torch.cuda.is_available():
        print(f'*** CUDA version: {torch.version.cuda}')
        print(f'*** cuDNN version: {torch.backends.cudnn.version()}')
        print(f'*** GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'*** GPU {i}: {torch.cuda.get_device_name(i)}')

    # Run sample computation.
    print('\n*** Running sample code...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.zeros((10, 10, 10), device=device)
    u = torch.ones((10, 10, 10), device=device)
    v = t + u
    if torch.allclose(v, torch.ones_like(v)):
        print('*** [OK]')
    else:
        print('*** [Fail]')
