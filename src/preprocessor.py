#!/usr/bin/env python3

'''
MLDisasm training preprocessor. Encode raw training data as JSON.

Usage: {0} <model name>
'''

import multiprocessing as mp
import json
import sys

from   mldisasm.io.codec        import AsciiCodec, BytesCodec
from   mldisasm.io.file_manager import FileManager
import mldisasm.io.log          as     log
from   mldisasm.io.training_set import DELIMITER, CHARS_PER_BYTE

REPORT_STEP = 10000

def pp_encode(record, tokens, x_codec, y_codec):
    '''
    Encode a single record.
    :param record: The record.
    :param tokens: The TokenList.
    :param x_codec: A BytesCodec.
    :param y_codec: An AsciiCodec.
    :returns: A JSON encoded tuple of the input vector and a one-hot encoded matrix.
    '''
    # Parse record into inputs and targets.
    elems  = record.split(DELIMITER)
    if len(elems) != 2:
        raise ValueError('training:{}: Bad training example: {}'.format(record_num, record))
    target       = elems[1]
    # Encode opcode (inputs).
    opcode       = int(elems[0], 16)
    opcode_len   = int(0.5 + len(elems[0])/CHARS_PER_BYTE)
    opcode_bytes = opcode.to_bytes(opcode_len, 'little')
    inputs       = x_codec.encode(opcode_bytes, as_tensor=False)
    # Encode target indices as one-hot vectors.
    targets  = y_codec.encode(target, as_tensor=False)
    return json.dumps([inputs,targets])

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
    # Encode training set. Our strategy here is to read the records into memory one at a time and hand them to one of N
    # workers (see pp_encode) where N is the number of CPU threads. mp.Pool handles allocating each line of input to a
    # worker. Each worker produces a single line of JSON data for each record it's given and the results are stored
    # asynchronously in a list. After queueing all the tasks (one per line of input) we write the results out in the
    # order that they become available.
    record_num = 0
    n_threads  = mp.cpu_count()
    with mp.Pool(processes=n_threads) as pool:
        # Submit tasks and collect asynchronous results.
        log.info('Processing training file (spawning {} workers)'.format(n_threads))
        results = []
        for record in tset_in:
            record = record[:-1] # Remove trailing newline.
            results.append(pool.apply_async(pp_encode, (record, tokens, x_codec, y_codec)))
        pool.close()
        # Write the results of each task out to the file as they become available.
        log.info('Waiting for results to become available')
        while results:
            for i in range(len(results)):
                if results[i].ready():
                    line = results[i].get()
                    tset_out.write(line)
                    tset_out.write('\n')
                    if record_num and record_num % REPORT_STEP == 0:
                        log.info('Processed {} records'.format(record_num))
                    record_num += 1
                    del results[i] # This invalidates the indices, so we need to loop again from the beginning.
                    break
        pool.join()
