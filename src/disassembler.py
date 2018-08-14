#!/usr/bin/env python3

'''Disassemble a binary file using one of the supported syntax models.

Usage: {0} <model> <binary>

 * <model> is the name of the assembly syntax model to use, such as "intel" or "att" (AT&T).
 * <binary> names a file containing machine code. The file will be interpreted as raw binary.

Disassembly is written to standard output.
'''

import sys

import tensorflow               as tf
import tensorflow.keras.backend as K

from   mldisasm.io.codec        import AsciiCodec, BytesCodec
from   mldisasm.io.file_manager import FileManager
import mldisasm.io.log          as     log

if __name__ == '__main__':
    # Read the command line.
    if len(sys.argv) != 3:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        exit(1)
    model_name = sys.argv[1]
    input_path = sys.argv[2]
    # Load files and create codecs.
    file_mgr   = FileManager()
    config     = file_mgr.load_config()
    tokens     = file_mgr.load_tokens()
    seq_len    = config['seq_len']
    mask_value = config['mask_value']
    x_codec    = BytesCodec(seq_len, mask_value)
    y_codec    = AsciiCodec(seq_len, mask_value, tokens)
    model      = file_mgr.load_model(model_name)
    # Process the file in seq_len sized chunks. TODO: Implement sliding window. FIXME: How do we detect instruction
    # boundaries when a block of N bytes could contain anywhere from N/15 to N instructions?
    with open(input_path, 'rb') as file:
        while True:
            buffer  = file.read(seq_len)
            if not buffer:
                break
            encoded = x_codec.encode(buffer)
            pred    = model.predict(tf.stack([encoded]), steps=1)
            decoded = y_codec.decode(pred).eval()
            print(decoded[0].decode())
