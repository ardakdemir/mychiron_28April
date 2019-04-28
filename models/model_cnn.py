"""
2018-11-1 model structures of different implementation of U-net
"""

from __future__ import division

from util import *
from keras import Input, models, layers, regularizers
from keras.optimizers import RMSprop,SGD, Adam
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical
from keras_contrib.layers import CRF


########################################
# Network structures
# 1. CNN
########################################

def CNN(signals, output_len, kernel_size, conv_window_len, lr, dropoutRate):
    
    # conv
    inner = signals
    inner = layers.Conv1D(kernel_size[0], conv_window_len, activation="relu")(inner)
    inner = layers.Conv1D(kernel_size[1], conv_window_len, activation="relu")(inner)
    inner = layers.MaxPooling1D(2)(inner)
    

    inner = layers.Conv1D(kernel_size[0], conv_window_len, activation="relu")(inner)
    inner = layers.Conv1D(kernel_size[1], conv_window_len, activation="relu")(inner)
    inner = layers.MaxPooling1D(2)(inner)

    # fc
    inner = layers.Flatten()(inner)
    #inner = layers.Dense(output_len, activation="relu")(inner)

    # output
    output =  layers.Concatenate(1)([ layers.Reshape((1,4))(layers.Dense(4, activation='softmax')(inner)) for i in range(output_len) ])
    

    model = models.Model(signals, output)

    return model


