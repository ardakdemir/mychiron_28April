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
#import chiron.chiron_model as model

from data_input import read_raw_data_sets
#from chiron.cnn import getcnnfeature
#from chiron.cnn import getcnnlogit
from six.moves import range

from keras import Input, models, layers, regularizers, metrics
from keras.optimizers import RMSprop, SGD, Adam
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras.layers import Lambda

from models.model_unet import *
from models.model_cnn import *
from models.model_prediction import *

import h5py
import numpy as np

CB = [callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="auto", restore_best_weights=True)]


def loading_data(cacheFile):

    if os.path.exists(cacheFile):

        with h5py.File(cacheFile, "r") as hf:

            X = hf["X_data"][:]
            seq_len = hf["seq_len"][:]
            label = [hf["Y_ctc/index"][:], hf["Y_ctc/value"][:], hf["Y_ctc/shape"]]
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


def get_categorical_acc(pred, gold, l):

    assert(len(pred) == len(gold))
    correct = 0
    for i in range(l):
        if pred[i] == gold[i]:
            correct += 1

    return correct/float(l)



# UNet train for the model, non-generator version
def test(lossType="dice", modelSavePath="../experiment/model/"):

    # loading from cache data
    # seqLen records the real size containing the intergate event
    
    trainX, seqLen, label, label_vec, label_seg = loading_data("../data/cache/test_cache.h5")
    
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1).astype("float32")
    trainY = to_categorical(label_vec)

    print(trainX.shape)
    print(trainY.shape)

    ########################################
    # loading keras model and train
    #######################################
    kernel_size =[128, 128]
    maxpooling_len = [3, 2, 2, 2, 2, 3]
    conv_window_len = 5
    dropoutRate = 0.2
    lr = 0.01

    # model parameters
    model_name = "UNet_model_loss-" + lossType \
            + "-maxpoolingLen_"+"-".join([str(l) for l in maxpooling_len]) \
            + "-convWindowLen_" + str(conv_window_len) \
            + "-lr_" + str(lr) \
            + "-dropout" + str(dropoutRate)
    
    print("@ Loading model ...")
    model = models.load_model(modelSavePath+"/"+model_name+".h5", custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef})

    # prediction
    
    preds = model.predict(trainX, verbose=1)
    golds = label_vec

    eds = []
    categorical_acc = []
    for i in range(trainX.shape[0]):
        
        logits = toBases(np.argmax(preds[i], -1), seqLen[i])
        gold = toBases(golds[i])
        
        if i%10000 == 0:
            print(logits)
            print(gold)

        ed =editDistance(logits, gold)
        eds.append(ed)

        # calcuate categorical acc
        acc = get_categorical_acc(np.argmax(preds[i],-1), golds[i], seqLen[i])
        categorical_acc.append(acc)

        print("pred=%d, gold=%d, ed=%f, acc=%f" %(len(logits), len(gold), ed, acc))


    print("* Averaged edit distance for the U-net model %d samples is: ed=%f, acc=%f" \
            %(trainX.shape[0], np.mean(eds), np.mean(categorical_acc))) 
    

def run(args):
    global FLAGS
    FLAGS = args
    FLAGS.data_dir = FLAGS.data_dir + os.path.sep
    FLAGS.log_dir = FLAGS.log_dir + os.path.sep
    
    print(FLAGS)
    print("-"*30)
    
    test()


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
    parser.add_argument('-loss', '--loss', default="dice", help="loss function used to learn the segmentation model.")

    parser.set_defaults(retrain=False)
    args = parser.parse_args(sys.argv[1:])
    
    if args.train_cache is None:
        args.train_cache = args.data_dir + '/train_cache.hdf5'
    if (args.valid_cache is None) and (args.validation is not None):
        args.valid_cache = args.data_dir + '/valid_cache.hdf5'
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    run(args)


