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
MLDisasm Logging.
'''

import logging
import warnings
import sys

try:
    import colorama
    from   colorama import Fore
    _HAVE_COLORAMA = True
except ImportError:
    warnings.warn('Colorama is not available, output formatting disabled')
    _HAVE_COLORAMA = False

LOGGER = logging.root
LEVEL  = logging.DEBUG
MODE   = 'w'
STREAM = sys.stderr
FORMAT = '[%(asctime)s]: %(levelname)s: %(message)s'
COLORS = {}

if _HAVE_COLORAMA:
    COLORS = {
        'DEBUG':    Fore.BLUE,
        'INFO':     Fore.LIGHTBLACK_EX,
        'WARNING':  Fore.RED,
        'ERROR':    Fore.LIGHTRED_EX,
        'CRITICAL': Fore.LIGHTRED_EX
    }

class ColoredFormatter(logging.Formatter):
    '''
    Logging formatter which colours output according to logging-level.
    '''
    def __init__(self, fmt, *args, **kwargs):
        '''
        Construct formatter.
        '''
        super().__init__(fmt, *args, **kwargs)

    def format(self, record):
        level   = record.levelname
        message = super().format(record)
        if _HAVE_COLORAMA:
            color = COLORS[level]
            return str(color) + message + str(Fore.RESET)
        return message


class TeeHandler(logging.Handler):
    '''
    Logging handler which tees logging messages to a stream and a file.
    '''
    def __init__(self, file, stream, fmt, level=logging.NOTSET):
        '''
        Initialise handler.
        '''
        super().__init__(level)
        # Set up file handler. The log file contains all logging messages.
        if isinstance(file, str):
            file = open(str, 'w')
        self.file_handler = logging.StreamHandler(file)
        self.file_handler.setLevel(logging.NOTSET)
        self.file_handler.setFormatter(logging.Formatter(fmt))
        # Set up stream handler. Log stream only contains INFO level messages and above.
        self.stream_handler = logging.StreamHandler(stream)
        self.stream_handler.setLevel(logging.INFO)
        self.stream_handler.setFormatter(ColoredFormatter(fmt))

    def emit(self, record):
        '''
        Emit record.
        '''
        if record.levelno >= self.file_handler.level:
            self.file_handler.emit(record)
        if record.levelno >= self.stream_handler.level:
            self.stream_handler.emit(record)

def init(file, stream=STREAM, fmt=FORMAT, level=LEVEL):
    '''
    Initialise logging subsystem.
    '''
    global debug, info, warning, error, critical
    if _HAVE_COLORAMA:
        colorama.init()
    logger  = LOGGER
    logger.setLevel(logging.NOTSET)
    handler = TeeHandler(file, stream, fmt, level)
    logger.handlers = []
    logger.addHandler(handler)
    debug    = logging.debug
    info     = logging.info
    warning  = logging.warning
    error    = logging.error
    critical = logging.critical

# Callable which does nothing.
class _NullCallable:
    def __call__(self, *args, **kwargs):
        return

# Aliases for logging functions.
debug    = _NullCallable()
info     = _NullCallable()
warning  = _NullCallable()
error    = _NullCallable()
critical = _NullCallable()
