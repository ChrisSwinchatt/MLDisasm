#!/usr/bin/env python3

'''
Test mldisasm.io.codec
'''

from mldisasm.constants    import START_TOKEN, STOP_TOKEN
from mldisasm.io.codec     import AsciiCodec, BytesCodec
from mldisasm.tests.common import *

# Disable checking for "instance ... has no ... member" because PyTest test case instance attributes have to be defined
# outside of __init__.
# pylint: disable=E1101

class TestCodec(GenericTestCase):
    '''
    Test codecs.
    '''
    def _test_ascii_codec_iter(self, codec):
        string  = START_TOKEN + random_string(SEQ_LEN - 2) + STOP_TOKEN
        encoded = codec.encode(string)
        decoded = codec.decode(encoded)
        assert string == decoded

    def test_ascii_codec(self):
        '''
        Test AsciiCodec.
        '''
        codec = AsciiCodec(self.config['model']['y_seq_len'], self.config['model']['mask_value'])
        self.itertest(
            test=self.test_ascii_codec,
            func=self._test_ascii_codec_iter,
            args=(codec,)
        )

    def _test_bytes_codec_iter(self, codec):
        bs      = random_bytes(SEQ_LEN)
        encoded = codec.encode(bs)
        decoded = codec.decode(encoded)
        assert bs == decoded

    def test_bytes_codec(self):
        '''
        Test BytesCodec
        '''
        codec = BytesCodec(self.config['model']['x_seq_len'], self.config['model']['mask_value'])
        self.itertest(
            test=self.test_bytes_codec,
            func=self._test_bytes_codec_iter,
            args=(codec,)
        )
