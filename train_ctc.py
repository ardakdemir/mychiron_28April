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

from keras import Input, models, layers, regularizers, metrics
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


def loading_data(cacheFile):

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
def train():

    # loading from cache data
    # seqLen records the real size containing the intergate event
    print("* Loading the data ...")
    trainX, seqLen, label, label_vec, label_segs = loading_data("../data/cache/test_cache.h5")
    print("Done!")
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
    
    # generate the sparse matrix
    trainY = K.eval(tf.SparseTensor(label[0], label[1], label[2]))
    # trainY = to_categorical(trainY) 
    
    print(trainX.shape)
    #print(trainY.shape) # length of y should be more less.


    ########################################
    # loading keras model and train
    #######################################
    kernel_size =[128, 256]
    maxpooling_len = [5, 5, 2, 2, 5, 5]
    conv_window_len = 7
    dropoutRate = 0.2
    lr = 0.001

    signals = Input(name='input', shape=[trainX.shape[1], 1], dtype='float32')

    model = CTC(signals, trainX.shape[1], trainY.shape[1], kernel_size,conv_window_len, lr, dropoutRate, True)
    model.compile(loss={'ctc':lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=[metrics.categorical_accuracy])
    model.summary()

    hisotry = model.fit([trainX, trainY, np.array(np.ones(trainX.shape[0])*100), np.array(np.ones(trainY.shape[0])*50)],\
            trainY, epochs=100, batch_size=64, verbose=1, callbacks=CB, validation_split=0.2)
 

# decoding results


def run(args):
    global FLAGS
    FLAGS = args
    FLAGS.data_dir = FLAGS.data_dir + os.path.sep
    FLAGS.log_dir = FLAGS.log_dir + os.path.sep
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')

    parser.add_argument('-g', '--gpu', default='0', help="Assigned GPU for running the code")

    parser.add_argument('-i', '--data_dir', default="/data/workspace/nanopore/data/chiron_data/eval/" ,required= False,
                        help="Directory that store the tfrecord files.")
    parser.add_argument('-o', '--log_dir', default="model/", required = False  ,
                        help="log directory that store the training model.")
    parser.add_argument('-m', '--model_name', default="devTest", required = False,
                        help='model_name')
    parser.add_argument('-v', '--validation', default = None, 
                        help="validation tfrecord file, default is None, which conduct no validation")
    parser.add_argument('-f', '--tfrecord', default="train.tfrecords",
                        help='tfrecord file')
    parser.add_argument('--train_cache', default=None, help="Cache file for training dataset.")
    parser.add_argument('--valid_cache', default=None, help="Cache file for validation dataset.")
    parser.add_argument('-s', '--sequence_len', type=int, default=300,
                        help='the length of sequence')
    parser.add_argument('-b', '--batch_size', type=int, default=400,
                        help='Batch size')
    parser.add_argument('-t', '--step_rate', type=float, default=1e-2,
                        help='Step rate')
    parser.add_argument('-x', '--max_steps', type=int, default=10000,
                        help='Maximum step')
    parser.add_argument('-n', '--segments_num', type = int, default = 20000,
                        help='Maximum number of segments read into the training queue, default(None) read all segments.')
    parser.add_argument('--configure', default = None,
                        help="Model structure configure json file.")
    parser.add_argument('-k', '--k_mer', default=1, help='Output k-mer size')
    parser.add_argument('--retrain', dest='retrain', action='store_true',
                        help='Set retrain to true')
    parser.add_argument('--read_cache',dest='read_cache',action='store_true',
                        help="Read from cached hdf5 file.")
    
    parser.set_defaults(retrain=False)
    args = parser.parse_args(sys.argv[1:])
    
    if args.train_cache is None:
        args.train_cache = args.data_dir + '/train_cache.hdf5'
    if (args.valid_cache is None) and (args.validation is not None):
        args.valid_cache = args.data_dir + '/valid_cache.hdf5'
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    run(args)


