#!/usr/bin/env python3

'''Disassemble a binary file using one of the supported syntax models.

Usage: {0} <model> <binary>

 * <model> is the name of the assembly syntax model to use, such as "intel" or "att" (AT&T).
 * <binary> names a file containing machine code. The file will be interpreted as raw binary.

Disassembly is written to standard output.
'''

import os
import sys

if __name__ == '__main__':
    print('*** Starting up...')
    # Filter out debug messages from TF.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow               as tf
import tensorflow.keras.backend as K

from mldisasm.io.codec        import AsciiCodec, BytesCodec
from mldisasm.io.file_manager import FileManager
from mldisasm.util            import log
from mldisasm.model           import Disassembler

if __name__ == '__main__':
    # Read the command line.
    if len(sys.argv) != 3:
        print(__doc__.format(sys.argv[0]), file=sys.stderr)
        exit(1)
    model_name = sys.argv[1]
    input_path = sys.argv[2]
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load files and create codecs.
    file_mgr = FileManager()
    config   = file_mgr.load_config(model_name)
    x_codec  = BytesCodec(config['model']['x_seq_len'], config['model']['mask_value'])
    y_codec  = AsciiCodec(config['model']['y_seq_len'], config['model']['mask_value'])
    # Create a new model and load pretrained weights into it.
    model = Disassembler(**config['model']) #file_mgr.load_model(model_name)
    model.load_weights(file_mgr.qualify_model(model_name))
    # Process the file in seq_len sized chunks. TODO: Implement sliding window. FIXME: How do we detect instruction
    # boundaries when a block of N bytes could contain anywhere from N/15 to N instructions?
    with tf.Session() as session, open(input_path, 'rb') as file:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        offset = 0
        while True:
            buffer = file.read(config['model']['x_seq_len'])
            if not buffer:
                break
            encoded = x_codec.encode(buffer)
            pred    = model.predict(tf.convert_to_tensor([encoded]), steps=1)
            decoded = ''.join(y_codec.decode(pred))
            print('0x{:08x}:\t{}\t{}'.format(
                offset,
                buffer.hex(),
                decoded
            ))
            offset += len(buffer)
