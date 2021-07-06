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
MLDisasm TensorFlow diagnostics.
'''

import importlib
import os

if __name__ == '__main__':
    print('*** Starting TensorFlow diagnostics')
    # Import TF with verbosity.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    tensorflow  = importlib.import_module('tensorflow')
    # Print detected devices.
    device_lib  = importlib.import_module('tensorflow.python.client.device_lib')
    print('\n*** GPU available: {}'.format('yes' if tensorflow.test.is_gpu_available() else 'no'))
    print('\n*** Devices found by TensorFlow:\n{}'.format(
        '\n * '.join(list(map(str, device_lib.list_local_devices())))
    ))
    gpu_device = tensorflow.test.gpu_device_name()
    if gpu_device:
        print('\n*** GPU device name: {}'.format(gpu_device))
    # Run some sample code.
    print('\n*** Running sample code...')
    with tensorflow.device(gpu_device), tensorflow.Session() as session:
        t = tensorflow.zeros((10,10,10))
        u = tensorflow.ones((10,10,10))
        v = session.run(t + u)
        if (v == session.run(u)).all():
            print('*** [OK]')
        else:
            print('*** [Fail]')
