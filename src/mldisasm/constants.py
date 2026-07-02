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
MLDisasm constants.
'''

# Maximum value of an ASCII character.
ASCII_MAX = 0x7F

# Maximum value of a byte.
BYTE_MAX  = 0xFF

# Byteorder.
BYTEORDER = 'little'

# First token in a decoded sequence.
START_TOKEN = '\t'

# Special token indices for the assembly vocabulary.
PAD_IDX   = 0
START_IDX = 1
END_IDX   = 2
UNK_IDX   = 3

# Assembly vocabulary size (printable ASCII + 4 special tokens).
ASSEMBLY_VOCAB_SIZE = ASCII_MAX + 4
