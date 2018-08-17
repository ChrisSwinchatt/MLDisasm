#!/usr/bin/env python3

'''
MLDisasm profiling.
'''

from   abc import ABCMeta, abstractmethod
import os
import sys
import time

import tensorflow as tf

import mldisasm.io.log as log

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError as e:
    HAVE_PSUTIL = False
    log.debug('Couldn\'t import psutil, memory profiling disabled')
    log.debug('ImportError: {}'.format(str(e)))

class GenericResourceProfiler:
    '''
    Abstract base class for resource profilers.
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        '''
        Initialise GenericResourceProfiler. Takes the initial resource measurement.
        '''
        self.initial = self.measure_resource()
        self.final   = None

    @abstractmethod
    def measure_resource(self):
        '''
        Measure the resource.
        :returns: The resource measurement or None if it can't be measured. This method shouldn't raise exceptions.
        '''
        raise NotImplementedError

    def compute_delta(self):
        '''
        Take the final resource measurement and compute the delta between the initial and final samples.
        :returns: The change in resource usage, e.g. time elapsed or memory allocated. If this can't be computed
        (because the resource can't be measured), None is returned.
        '''
        if self.initial is None:
            return None
        self.final = self.measure_resource()
        if self.final is None:
            return None
        return self.final - self.initial

    def __str__(self):
        '''
        Get a string describing the change in resource, e.g. "allocated 100 bytes". This should be overloaded.
        :returns: The string, or an empty string if the resource delta couldn't be computed.
        '''
        delta = self.compute_delta()
        if delta is None:
            return ''
        return 'Resource usage {}creased by {}'.format(
            'in' if delta >= 0 else 'de',
            delta
        )

class TimeProfiler(GenericResourceProfiler):
    '''
    Profile elapsed time.
    '''
    def measure_resource(self):
        '''
        Measure elapsed time.
        '''
        return time.time()

    def __str__(self):
        return 'in {} seconds'.format(self.compute_delta())

class MemoryProfiler(GenericResourceProfiler):
    '''
    Profile memory usage.
    '''
    def __init__(self):
        '''
        Initialise MemoryProfiler.
        '''
        if HAVE_PSUTIL:
            self.process = psutil.Process(os.getpid())
        super().__init__()

    def measure_resource(self):
        '''
        Measure memory usage.
        '''
        if not HAVE_PSUTIL:
            return None
        return self.process.memory_info().rss

    def compute_delta(self):
        '''
        Compute the change in memory usage. Returns None if psutil was not loaded.
        '''
        if HAVE_PSUTIL:
            return super().compute_delta()
        return None

    def __str__(self):
        delta = self.compute_delta()
        if delta is None:
            return ''
        return '{}d {} bytes'.format(
            'allocate' if delta >= 0 else 'free',
            delta
        )

class GraphProfiler(GenericResourceProfiler):
    '''
    Profile graph operations.
    '''
    def measure_resource(self):
        '''
        Measure graph nodes.
        '''
        return len(tf.get_default_graph().get_operations())

    def __str__(self):
        delta = self.compute_delta()
        return '{}ed {} graph nodes'.format(
            'add' if delta >= 0 else 'remov',
            delta
        )

_RESOURCE_PROFILERS = {
    'time':   TimeProfiler(),
    'memory': MemoryProfiler(),
    'graph':  GraphProfiler()
}

class Profiler:
    '''
    Reports time taken or memory used between its initialisation and its destruction (or when the end method is called,
    whichever comes first).
    '''
    def __init__(self, start_msg, end_msg, *args, log_level='debug', resources=['time','memory','graph']):
        '''
        Initialise Profiler.
        :param start_msg: A message to print upon starting. If None or empty, nothing is printed.
        :param end_msg: A message to print upon ending. If at least one of PROF_TIME and PROF_MEM is True, the amount of
        time elapsed and/or memory consumed is appended to end_msg before printing.
        :param args: Extra format arguments for end_msg. Any callable arguments will be called (with no arguments) and
        replaced by their return value. This can be used to print values that change in the time between the object's
        instantiation and its destruction/the call to Profiler.end().
        :param log_level: The log level to to output to. Possible values are None, 'debug', 'info', 'warning' and
        'error'. Default is 'debug'. None means output goes to stderr.
        :param resources: The resources to measure.
        :example:
            y = 0
            with Profiler('Snafucating', 'Result of snafucation: {}', lambda: y, log_level='info'):
                y = make_snafucated(x)
        '''
        # Save params and initialise attributes.
        if not end_msg:
            end_msg = 'Operation finished'
        self.args      = args
        self.end_msg   = end_msg
        self.log_level = log_level
        self.ended     = False
        self.profilers = []
        for resource in resources:
            if resource not in _RESOURCE_PROFILERS:
                raise ValueError('Can\'t profile unknown resource \'{}\'. Known resources: \'{}\''.format(
                    resource,
                    '\', \''.join(_RESOURCE_PROFILERS.keys())
                ))
            self.profilers.append(_RESOURCE_PROFILERS[resource])
        # Print start message if any.
        if start_msg:
            self._print(start_msg)

    def __del__(self):
        if not self.ended:
            self.end()

    def __enter__(self):
        '''
        Enter context.
        '''
        pass

    def __exit__(self, exc_type, exc_value, exc_tb):
        '''
        Leave context.
        '''
        if not self.ended:
            self.end()

    def end(self):
        '''
        Print the message and profiling info. The start time and memory usage are reset so that successive calls to end
        print updated information. However, after this function is called, leaving the context will not print the new
        message. To re-enable this behaviour, set the "ended" attribute to False.
        '''
        msg  = self.end_msg
        args = self.args
        # Call callable args.
        tmp = tuple()
        for arg in args:
            if hasattr(arg, '__call__'):
                arg = arg()
            tmp = (*tmp, arg)
        args = tmp
        # Append profiler info.
        msg = msg + ' ' + ', '.join(map(str, self.profilers))
        # Print the message and set the 'ended' flag.
        self._print(msg.format(*args))
        self.ended = True

    def _print(self, msg):
        if self.log_level is None:
            print(msg, file=sys.stderr)
        elif self.log_level == 'debug':
            log.debug(msg)
        elif self.log_level == 'info':
            log.info(msg)
        elif self.log_level == 'warning':
            log.warning(msg)
        elif self.log_level == 'error':
            log.error(msg)
        else:
            raise ValueError('Unknown log level \'{}\''.format(self.log_level))

def prof(end_msg, *args, start_msg=None, **kwargs):
    '''
    Create a profiler which reports when it goes out of scope, leaves its context, or you call its "end" method
    explicitly. See Profiler.__init__() for parameter info.
    The following examples are all equivalent.
    :example:
        with prof('Loaded file'), open(path, 'r') as file:
            data = file.read()
    :example:
        p = prof('Loaded file')
        with open(path, 'r') as file:
            data = file.read()
        p.end()
    :example:
        p = prof('Loaded file')
        with open(path, 'r') as file:
            return file.read()
    '''
    return Profiler(start_msg, end_msg, *args, **kwargs)