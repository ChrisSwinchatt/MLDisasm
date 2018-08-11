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

def pp_encode(record, x_codec, y_codec):
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

def write_available_results(results, tset_out, record_num):
    '''
    Wait until at least one result is available and then write it/them out to disk.
    '''
    # Collect at least one ready result.
    avail = []
    while not avail:
        for i in range(len(results)):
            if results[i].ready():
                avail.append(i)
    count = len(avail)
    assert count > 0
    # Delete the results in reverse order so we don't invalidate indices.
    for i in reversed(avail):
        line = results[i].get()
        tset_out.write(line)
        tset_out.write('\n')
        del results[i]
    if (record_num + count) % REPORT_STEP == 0:
        log.info('Processed {} records'.format(record_num + count))
    return count

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
    x_codec  = BytesCodec(config['seq_len'], config['mask_value'])
    y_codec  = AsciiCodec(config['seq_len'], config['mask_value'], tokens)
    tset_in  = file_mgr.open_training_raw(model_name)
    tset_out = file_mgr.open_training_pp(model_name)
    # Encode training set. Our strategy is to read one record per worker thread (n_threads), then wait for at least one
    # of those workers to finish and write the results. This way we don't load the entire file into memory at once.
    record_num = 0
    n_threads  = mp.cpu_count()
    with mp.Pool(processes=n_threads) as pool:
        # Submit tasks and collect asynchronous results.
        log.info('Processing training file (spawning {} workers)'.format(n_threads))
        results    = []
        record_num = 0
        active     = 0
        for record in tset_in:
            record = record[:-1] # Remove trailing newline.
            results.append(pool.apply_async(pp_encode, (record, x_codec, y_codec)))
            active += 1
            while active >= n_threads:
                count = write_available_results(results, tset_out, record_num)
                record_num += count
                active     -= count
                # Don't let active underflow.
                if active < 0:
                    log.warning('`active` counter underflowed')
                    active = 0
        pool.close()
        # Wait for the rest of the results to become available.
        log.info('Waiting for remaining {} results'.format(len(results)))
        while results:
            record_num += write_available_results(results, tset_out, record_num)
