#!/usr/bin/env python3

'''
MLDisasm training preprocessor. Encode raw training data as JSON.

Usage: {0} <model name>
'''

import json
import sys

from   mldisasm.benchmarks.profiling import prof
from   mldisasm.io.codec             import AsciiCodec, BytesCodec
from   mldisasm.io.file_manager      import FileManager
import mldisasm.io.log               as     log
from   mldisasm.io.training_set      import DELIMITER, CHARS_PER_BYTE

REPORT_STEP = 10000

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__.format(sys.argv[0]))
        exit(1)
    model_name = sys.argv[1]
    # Open files.
    file_mgr = FileManager()
    log.init(file_mgr.open_log())
    config   = file_mgr.load_config()
    tokens   = file_mgr.load_tokens()
    x_codec  = BytesCodec(config['seq_len'])
    y_codec  = AsciiCodec(config['seq_len'], tokens)
    tset_in  = file_mgr.open_training_raw(model_name)
    tset_out = file_mgr.open_training_pp(model_name)
    # Encode training set.
    record_num = 1
    log.info('Processing training file')
    profiler = prof('Processed {} records', REPORT_STEP, use_log=False)
    for record in tset_in:
        # Parse record into inputs and targets.
        record = record[:-1] # Cut off the newline.
        elems  = record.split(DELIMITER)
        if len(elems) != 2:
            raise ValueError('training:{}: Bad training example: {}'.format(record_num, record))
        target       = elems[1]
        opcode       = int(elems[0], 16)
        opcode_len   = int(0.5 + len(elems[0])/CHARS_PER_BYTE)
        opcode_bytes = opcode.to_bytes(opcode_len, 'little')
        inputs       = x_codec.encode(opcode_bytes, as_tensor=False)
        targets      = y_codec.encode(target, as_tensor=False)
        # Save to JSON file with a pair per line.
        json.dump((inputs,targets), tset_out)
        tset_out.write('\n')
        # Report progress.
        record_num += 1
        if record_num % REPORT_STEP == 0:
            profiler.end()
            profiler = prof('Processed {} records ({} total)', REPORT_STEP, record_num, use_log=False)
