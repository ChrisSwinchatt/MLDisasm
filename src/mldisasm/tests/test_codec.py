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

"""
Test mldisasm.io.codec
"""

from mldisasm.constants    import START_TOKEN, STOP_TOKEN
from mldisasm.io.codec     import ByteIndexer, AssemblyTokenizer
from mldisasm.tests.common import *


class TestCodec(GenericTestCase):
    """Test codecs."""

    def _test_assembly_tokenizer_iter(self, tokenizer):
        string = START_TOKEN + random_string(SEQ_LEN - 2) + STOP_TOKEN
        encoded = tokenizer.encode(string, as_tensor=False)
        decoded = tokenizer.decode(encoded)
        assert string == decoded

    def test_assembly_tokenizer(self):
        """Test AssemblyTokenizer."""
        tokenizer = AssemblyTokenizer(self.config['model']['y_seq_len'])
        self.itertest(
            test=self.test_assembly_tokenizer,
            func=self._test_assembly_tokenizer_iter,
            args=(tokenizer,)
        )

    def _test_byte_indexer_iter(self, indexer):
        bs = random_bytes(SEQ_LEN)
        encoded = indexer.encode(bs, as_tensor=False)
        decoded = indexer.decode(encoded)
        assert bs == decoded

    def test_byte_indexer(self):
        """Test ByteIndexer."""
        indexer = ByteIndexer(self.config['model']['x_seq_len'])
        self.itertest(
            test=self.test_byte_indexer,
            func=self._test_byte_indexer_iter,
            args=(indexer,)
        )
