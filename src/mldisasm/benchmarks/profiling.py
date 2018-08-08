#!/usr/bin/env python3

'''
MLDisasm profiling.
'''

import os
import sys
import time

import psutil

import mldisasm.io.log as log

PROF_TIME = True
PROF_MEM  = True

class Profiler:
    '''
    Reports time taken or memory used between its initialisation and its destruction (or when the end method is called,
    whichever comes first).
    '''
    def __init__(self, start_msg, end_msg, *args, use_log=True):
        '''
        Initialise Profiler.
        :param start_msg: A message to print upon starting. If None or empty, nothing is printed.
        :param end_msg: A message to print upon ending. If None or empty but at least one of PROF_TIME and PROF_MEM is
        True, a default message containing the amount of time and/or memory used is printed.
        :param args: Extra format arguments for end_msg. Any callable arguments will be called (with no arguments) and
        replaced by their return value. This can be used to print values that change in the time between the object's
        instantiation and its destruction/the call to Profiler.end().
        '''
        # Save params and initialise attributes.
        if not end_msg:
            end_msg = 'Operation finished'
        self.args       = args
        self.end_msg    = end_msg
        self._use_log   = use_log
        self.start_time = time.time()
        self.end_time   = 0
        self.process    = psutil.Process(os.getpid())
        self.start_mem  = self.process.memory_info()
        self.end_mem    = None
        self.ended      = False
        # Print start message if any.
        if start_msg:
            self._print(start_msg)

    @property
    def elapsed(self):
        '''
        Compute the amount of time elapsed.
        '''
        return self.end_time - self.start_time

    @property
    def allocated(self):
        '''
        Compute the amount of memory allocated.
        '''
        assert self.end_mem is not None
        return self.end_mem.rss - self.start_mem.rss

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
        # Get the new end time & memory usage.
        self.end_time = time.time()
        self.end_mem  = self.process.memory_info()
        msg  = self.end_msg
        args = self.args
        # Call callable args.
        tmp = tuple()
        for arg in args:
            if hasattr(arg, '__call__'):
                arg = arg()
            tmp = (*tmp, arg)
        args = tmp
        # Append time/mem info.
        if PROF_TIME:
            msg += ' in {:f} seconds'
            args = (*args, self.elapsed)
        if PROF_MEM:
            alloc = self.allocated
            if alloc >= 0:
                msg += ' and allocated {} bytes'
            else:
                alloc = -alloc
                msg += ' and freed {} bytes'
            args = (*args, alloc)
        self._print(msg.format(*args))
        self.ended = True
        # Reset stats.
        self.start_time = self.end_time
        self.start_mem  = self.end_mem

    def _print(self, msg):
        if self._use_log:
            log.debug(msg)
        else:
            print(msg, file=sys.stderr)

def prof(end_msg, *args, start_msg=None, **kwargs):
    '''
    Create a profiler which reports when it goes out of scope or you call its "end" method.
    '''
    return Profiler(start_msg, end_msg, *args, **kwargs)

def init(prof_time=True, prof_mem=True):
    '''
    Initialise profiling.
    :param prof_time: Whether to profile time.
    :param prof_mem: Whether to profile memory.
    '''
    global PROF_TIME, PROF_MEM
    PROF_TIME = prof_time
    PROF_MEM  = prof_mem
