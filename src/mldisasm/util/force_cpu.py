#!/usr/bin/env python3

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
