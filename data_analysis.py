#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Mon Mar 27 14:04:57 2017
# from rnn import rnn_layers

# 20190116
# UNet for processing the same amout of data as chiron, revised by Yao-zhong@imsut

from __future__ import absolute_import
from __future__ import print_function
import argparse, sys, os, time, json

from distutils.dir_util import copy_tree

import tensorflow as tf
import chiron.chiron_model as model

from data_input import read_raw_data_sets
from chiron.cnn import getcnnfeature
from chiron.cnn import getcnnlogit
from six.moves import range

from keras import Input, models, layers, regularizers
from keras.optimizers import RMSprop, SGD, Adam
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras.layers import Lambda

from model_unet import *
from model_ctc import *

import h5py
import numpy as np

CB = [callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="auto", restore_best_weights=True)]


def loading_data(cacheFile="../data/cache/train_cache.h5"):

    if os.path.exists(cacheFile):

        with h5py.File(cacheFile, "r") as hf:

            X = hf["X_data"][:]
            seq_len = hf["seq_len"][:]
            label = [K.cast(hf["Y_ctc/index"][:], np.int64), K.cast(hf["Y_ctc/value"][:], np.int64), K.cast(hf["Y_ctc/shape"], np.int64)]
            label_vec = hf["Y_vec"][:]
            label_seg = hf["Y_seg"][:]
    else:

        print("Now caching the data ... ")
        
        ds = read_raw_data_sets(FLAGS.data_dir, FLAGS.train_cache,FLAGS.sequence_len, FLAGS.k_mer)
        X, seq_len, label, label_vec, label_seg = ds.next_batch(ds._reads_n)

        with h5py.File(cacheFile, "w") as hf:

            hf.create_dataset("X_data", data=X)
            hf.create_dataset("seq_len", data=seq_len)

            hf.create_dataset("Y_vec", data=label_vec)
            hf.create_dataset("Y_seg", data=label_seg)

            hf.create_dataset("Y_ctc/index", data=label[0])
            hf.create_dataset("Y_ctc/value", data=label[1])
            hf.create_dataset("Y_ctc/shape", data=label[2])

        print("Done!")

    return X, seq_len, label, label_vec, label_seg




# UNet train for the model, non-generator version
def kmer_signal_analysis(kmer):

    print("* Loading cached singal-label data ...")
    trainX, seqLen, label, label_vec, label_segs = loading_data("../data/cache/train_cache.h5")
    print("- Done!")

    print("** [Data scale]: Sample=%d, maxSignalLen=%d, maxBaseLen=%d" \
            %(trainX.shape[0], trainX.shape[1], np.max([len(x) for x in label_segs])))

    # generate kmer-dic
    nc_dic = ['A', 'C', 'G', 'T']
    basic="ACGT"
    tmp=product(basic, repeat=kmer)
    mers = [ ''.join(x) for x in tmp]
    for i in range(len(mers)):
        mers[i] = []

    # iterate over all samples
    """
    for sidx in range(trainX.shape[0]):
        # process the kmer    
        for kidx in range(trainX):

    """
    

    
#def kmer_shift_analysis

if __name__ == "__main__":
    kmer_signal_analysis(3)
