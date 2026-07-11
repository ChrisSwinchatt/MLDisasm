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

"""
MLDisasm testing framework.
"""

from   abc import ABCMeta
import sys
import os

import numpy as np

from mldisasm.constants       import ASCII_MAX, BYTE_MAX, BYTEORDER
from mldisasm.io.file_manager import FileManager

TEST_ITERATIONS   = 100
MIN_STRING_LENGTH = 2
MAX_STRING_LENGTH = 512
MODEL_NAME        = 'att'
SEQ_LEN           = 50


class GenericTestCase:
    """Base class for tests."""
    __metaclass__ = ABCMeta

    def setup(self):
        """Setup."""
        self.file_mgr = FileManager()
        self.config   = self.file_mgr.load_config(MODEL_NAME)

    def teardown(self):
        """Teardown."""
        pass

    def itertest(self, test, func, args=None):
        """Run a test over TEST_ITERATIONS iterations."""
        if args is None:
            args = tuple()
        enter_test(test)
        for _ in range(TEST_ITERATIONS):
            enter_test_iter()
            func(*args)
            leave_test_iter()
        leave_test()


def enter_test(func):
    """Function called when a test begins."""
    sys.stdout.write(f'\n{func.__name__}: ')
    sys.stdout.flush()


def enter_test_iter():
    """Function called when a test iteration begins."""
    sys.stdout.write('-')
    sys.stdout.flush()


def leave_test_iter():
    """Function called when a test iteration ends."""
    sys.stdout.write('\b+')
    sys.stdout.flush()


def leave_test():
    """Function called when a test finishes."""
    sys.stdout.write('\n')
    sys.stdout.flush()


def random_size(max_size=MAX_STRING_LENGTH, min_size=MIN_STRING_LENGTH):
    """Generate a random integer from MIN_STRING_LENGTH to MAX_STRING_LENGTH."""
    return np.random.randint(min_size, max_size)


def random_bytes(max_size=MAX_STRING_LENGTH, min_size=MIN_STRING_LENGTH):
    """Generate a random bytes object up to MAX_STRING_LENGTH bytes long."""
    return b''.join(map(
        lambda x: int(x).to_bytes(1, BYTEORDER),
        np.random.randint(0, BYTE_MAX, random_size(max_size, min_size))
    ))


def random_string(max_size=MAX_STRING_LENGTH, min_size=MIN_STRING_LENGTH):
    """Generate a random string up to MAX_STRING_LENGTH chars long."""
    size = random_size(max_size, min_size)
    return ''.join(map(chr, np.random.randint(ASCII_MAX, size=size)))


def random_dict(max_size=SEQ_LEN, min_size=MIN_STRING_LENGTH + 1):
    """Generate a dictionary with random keys and values which are lists."""
    size = random_size(max_size, min_size)
    return dict([(random_string(size), np.random.random(size).tolist()) for _ in range(size)])
