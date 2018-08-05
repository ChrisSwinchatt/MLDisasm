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
        :param args: Extra format arguments for end_msg.
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

    def __del__(self):
        '''
        Destroy the profiler and print the message if it hasn't been printed already.
        '''
        if not self.ended:
            self.end()

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

    def end(self):
        '''
        Print the message and stop profiling. Successive calls will print the same message, but calls to __del__ won't
        do anything after this is called.
        '''
        self.end_time = time.time()
        self.end_mem  = self.process.memory_info()
        msg   = self.end_msg
        args  = self.args
        alloc = self.allocated
        if PROF_TIME:
            msg += ' in {} seconds'
            args = (*args, self.elapsed)
        if PROF_MEM:
            if alloc >= 0:
                msg += ', allocated {} bytes'
            else:
                alloc = -alloc
                msg += ', freed {} bytes'
            args = (*args, self.allocated)
        self._print(msg.format(*args))
        self.ended = True

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
