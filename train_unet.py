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
# 20190220 re-crafted U-net training


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

CB = [callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="auto", restore_best_weights=True)]


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


# UNet train for the model, non-generator version
def train(lossType="dice", modelSavePath="../experiment/model/"):

    # loading from cache data
    # seqLen records the real size containing the intergate event
    
    trainX, seqLen, label, label_vec, label_seg = loading_data("../data/cache/train_cache.h5")
    
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

    signals = Input(name='input', shape=[trainX.shape[1], 1], dtype=np.float32)
    model = UNet_networkstructure_basic(signals, conv_window_len, maxpooling_len, True, dropoutRate)
    
    fig=plt.figure()
    
    if lossType == "dice":
        model.compile(optimizer=Adam(lr = lr) , loss= dice_coef_loss, metrics=[dice_coef])
    else:
        model.compile(optimizer=Adam(lr=lr), loss = 'categorical_crossentropy' , metrics=[metrics.categorical_accuracy])
    
    history = model.fit(trainX, trainY, epochs=100, batch_size=128, verbose=1, callbacks=CB, validation_split=0.2)
    
    print("@ Saving model ...")

    model.save(modelSavePath + "/" + model_name + ".h5")

    figureSavePath="../experiment/devLOG/"+ model_name + ".png"
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Training curve of the whole training set " + lossType)
    
    plt.savefig(figureSavePath)
    plt.close("all")

    # prediction
    
    num_eval = 5000
    preds = model.predict(trainX[:num_eval], verbose=1)
    golds = label_vec[:num_eval]

    eds = []
    for i in range(num_eval):
        logits = toBases(np.argmax(preds[i], -1), seqLen[i])
        gold = toBases(golds[i])
        print("pred=%d, gold=%d" %(len(logits), len(gold)))
        ed =editDistance(logits, gold)
        eds.append(ed/len(gold))

    print("Averaged edit distance for the U-net model %d samples is: %f" %(num_eval, np.mean(eds))) 
    #y_pred = layers.Activation('Softmax')(output)
    
    """
    base_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    base_model.fit(trainX, trainY, epochs=10, batch_size=64, verbose=1, callbacks=CB, validation_split=0.2)
    """

    #model = UNet_networkstructure_basic(signals,conv_window_len, maxpooling_len, True, dropoutRate)
    #model = CNN_networkstructure(kernel_size, window_len, maxpooling_len, False, dropoutRate)

    """
    labels = Input(name="labels", shape=[trainY.shape[1]],dtype='int64')
    input_length = Input(name="input_length", shape=[1], dtype='int64')
    label_length = Input(name="label_length", shape=[1],dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = models.Model(input=[signals, labels, input_length, label_length], outputs=[loss_out])
    model.compile(loss={'ctc':lambda y_true, y_pred: y_pred}, optimizer='adam')
    
    #model = UNet_networkstructure_basic(rd_input, conv_window_len, maxpooling_len, True, dropoutRate)
    n_train = trainX.shape[0]
    model.summary()

    hisotry = model.fit([trainX, trainY, np.array(np.ones(n_train)*trainY.shape[1]), np.array(np.ones(n_train)*trainY.shape[1])],\
            trainY, epochs=10, batch_size=64, verbose=1, callbacks=CB, validation_split=0.2)
    """

# note the data, note this only support tf data

def generate_train_valid_datasets():
    
    if FLAGS.read_cache:
        train_ds = read_cache_dataset(FLAGS.train_cache)
        if FLAGS.validation is not None:
            valid_ds = read_cache_dataset(FLAGS.valid_cache)
        else:
            valid_ds = train_ds
        if train_ds.event.shape[1]!=FLAGS.sequence_len:
            raise ValueError("The event length of training cached dataset %d is inconsistent with given sequene_len %d"%(train_ds.event.shape()[1],FLAGS.sequence_len))
        if valid_ds.event.shape[1]!=FLAGS.sequence_len:
            raise ValueError("The event length of training cached dataset %d is inconsistent with given sequene_len %d"%(valid_ds.event.shape()[1],FLAGS.sequence_len))
        return train_ds,valid_ds
    sys.stdout.write("Begin reading training dataset.\n")
    
    train_ds = read_raw_data_sets(FLAGS.data_dir, FLAGS.train_cache,\
            FLAGS.sequence_len, FLAGS.k_mer)

    """
    train_ds = read_tfrecord(FLAGS.data_dir, 
                             FLAGS.tfrecord, 
                             FLAGS.train_cache,
                             FLAGS.sequence_len, 
                             k_mer=FLAGS.k_mer,
                             max_segments_num=FLAGS.segments_num)
    """
    sys.stdout.write("Begin reading validation dataset.\n")
    
    if FLAGS.validation is not None:
        valid_ds = read_tfrecord(FLAGS.data_dir, 
                                 FLAGS.validation,
                                 FLAGS.valid_cache,
                                 FLAGS.sequence_len, 
                                 k_mer=FLAGS.k_mer,
                                 max_segments_num=FLAGS.segments_num)
    else:
        valid_ds = train_ds
    
    return train_ds,valid_ds


def run(args):
    global FLAGS
    FLAGS = args
    FLAGS.data_dir = FLAGS.data_dir + os.path.sep
    FLAGS.log_dir = FLAGS.log_dir + os.path.sep
    
    print("@ Training U-net model for base-calling ...")
    print("*"*40)
    print(FLAGS)
    print("*"*40)
    print("\n")
    
    train(FLAGS.loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')

    parser.add_argument('-g', '--gpu', default='0', help="Assigned GPU for running the code")

    parser.add_argument('-i', '--data_dir', default="/data/workspace/nanopore/data/chiron_data/train/" ,required= False,
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


