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
Force MLDisasm/TensorFlow to use the CPU.
'''

import tensorflow               as tf
import tensorflow.keras.backend as K

TF_SESSION  = None
FORCING_CPU = False

def force_cpu():
    '''
    Force TensorFlow to use the CPU.
    '''
    global TF_SESSION, FORCING_CPU
    graph      = tf.get_default_graph()
    TF_SESSION = tf.Session(graph=graph, config=tf.ConfigProto(device_count={'GPU':0}))
    K.set_session(TF_SESSION)
    FORCING_CPU = True
