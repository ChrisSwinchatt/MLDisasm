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
