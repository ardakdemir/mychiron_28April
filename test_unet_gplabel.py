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

from data_input import *
#from data_input import read_raw_data_sets
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CB = [callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="auto", restore_best_weights=True)]

## find the adjacent sequences
def seqCompact(seq):

    idx = 0
    nc_seq, nc_count = [],[]

    current = seq[idx]
    count, output_seq = 1,""

    for i in range(1, len(seq)):
        if seq[i] == current:
            count += 1
        else:

            output_seq += current
            
            nc_seq.append(current)
            nc_count.append(count)
            
            #if count > 1:
            #    output_seq += "("+str(count)+ ")"
            count, current = 1, seq[i]


    if str(current) != '0':

    	output_seq += current
    	#if count > 1: output_seq += "(" + str(count) +")"
    	nc_seq.append(current)
    	nc_count.append(count)

    return (nc_seq, nc_count)


# bi-directional mapping of the nc with the adjacent count information 
def nc2newIdx(nc, count, max_len=12):

	Alphabeta=['0','1','2','3', '4']
	#alphabeta=['a','c','g','t']
	idx = Alphabeta.index(nc)
	new_idx = max_len*idx + count - 1
	
	return new_idx

def newIdx2nc(new_idx, max_len=12):
	
	Alphabet = ['A', 'C', 'G', 'T', "X"]
	idx = int(new_idx/max_len)
        count = new_idx%max_len + 1
	#return Alphabet[idx]+"_"+str(count)
	return (Alphabet[idx], count)

# tranform the adjacent labels with attached number information.
def adjLabelTransform(label_raw, debug=False):

	new_labels = []
	# according to the adjacent numbers, generate new labels
        nc_seq, nc_count = seqCompact(np.array(label_raw).astype(str))

	for idx in range(len(nc_count)):
		new_idx = nc2newIdx(nc_seq[idx], nc_count[idx]) + 1
		new_labels.extend([new_idx]*nc_count[idx])
	
	if debug:
		print("@Original label sequence is:")
		print(label_raw)
		print("* Generated new labels is:")
		print(new_labels)
		print("* Recover for the new labels list is:")
		print([newIdx2nc(idx-1) for idx in new_labels])
	
	return new_labels

# get final labels
def getFinalLabel(label_seqs):

	final_labels = []

	for label in label_seqs:
		nc, count = newIdx2nc(label)
		final_labels.extend([nc]*count)

	return final_labels


def loading_data(cacheFile):

    if os.path.exists(cacheFile):

        with h5py.File(cacheFile, "r") as hf:

            X = hf["X_data"][:]
            seq_len = hf["seq_len"][:]
            label = [hf["Y_ctc/index"][:], hf["Y_ctc/value"][:], hf["Y_ctc/shape"]]
            label_vec = hf["Y_vec"][:]

            label_seg = [ hf["Y_seg/"+str(i)][:] for i in range(len(X)) ]
            # checking the h5py loading process
	    label_raw = [ hf["label_raw/"+str(i)][:] for i in range(len(X))]

    else:
	print("Now caching the data ... ")
        
        ds = read_raw_data_sets(FLAGS.data_dir, FLAGS.train_cache,FLAGS.sequence_len, FLAGS.k_mer)
        X, seq_len, label, label_vec, label_seg, label_raw = ds.next_batch(ds._reads_n)
     
        with h5py.File(cacheFile, "w") as hf:

            hf.create_dataset("X_data", data=X)
            hf.create_dataset("seq_len", data=seq_len)

            hf.create_dataset("Y_vec", data=label_vec)
            #hf.create_dataset("Y_seg", data=label_seg)

            hf.create_dataset("Y_ctc/index", data=label[0])
            hf.create_dataset("Y_ctc/value", data=label[1])
            hf.create_dataset("Y_ctc/shape", data=label[2])

            for i in range(len(label_raw)):
	
		# problem issues brought by the padding
                # Note after change to the array, the fusion extra padding is incorporated. 
		# @@ re-checking data loading part. 
		hf.create_dataset("Y_seg/"+str(i), data=label_seg[i])
		hf.create_dataset("label_raw/"+str(i), data=np.array(label_raw[i], dtype=int))
			
		
    print("Data loading Done!")

    # perform the label tranformation
    label_vec_new = []  
    ## note, this need to be based on label_raw
    for i in range(len(label_raw)):

	newLabels = adjLabelTransform(label_raw[i])
	newLabelVec = []

	# expend for vectors
	for j in range(len(newLabels)):
		newLabelVec.extend([ newLabels[j] ] * label_seg[i][j])
	# do the padding for the rest with 0
	padding(newLabelVec, len(label_vec[i]), [0]*(len(label_vec[i]-len(newLabelVec))))	
		
	label_vec_new.append(newLabelVec)

    label_new = np.array(label_vec_new)

    return X, seq_len, label, label_vec, label_seg, label_raw, label_new

      
def test(lossType="dice", modelSavePath="../experiment/model/"):

    # loading from cache data

    trainX, seqLen, label, label_vec, label_seg, label_raw, label_vec_new = loading_data("../data/cache/t_eval.h5")

    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1).astype("float32")
    trainY = to_categorical(label_vec_new, num_classes=49)

    print(trainX.shape) # fixed length
    print(trainY.shape) # could be padding with extra 0 in the end part. 

    # model parameters
    kernel_size =[128, 128]
    maxpooling_len = [3, 2, 2, 2, 2, 3]
    conv_window_len = 5
    dropoutRate = 0
    lr = 0.1

    model_name = "UNet_model_loss-" \
            + "-maxpoolingLen_"+"-".join([str(l) for l in maxpooling_len]) \
            + "-convWindowLen_" + str(conv_window_len) \
            + "-lr_" + str(lr) \
            + "-dropout" + str(dropoutRate)

    model = models.load_model(modelSavePath + model_name + ".h5", custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef})

    ####################################################
    # prediction, inside training, which should be used
    ####################################################
    
    # inside training
    preds = model.predict(trainX, verbose=1)
    golds = label_vec

    eds = []
    for i in range(trainX.shape[0]):
        logits = toBases(np.argmax(preds[i], -1), seqLen[i])
        gold = ind2base(toBases(golds[i]))
	pred = getFinalLabel(logits)
       
	print("-"*40)
	print(gold)
  	print(pred)
	print(logits)
	print("-"*40)
	# print("pred=%d, gold=%d" %(len(logits), len(gold)))
        ed =editDistance(pred, gold)
        eds.append(ed)

    print("Averaged edit distance for the U-net model %d samples is: %f" %(trainX.shape[0], np.mean(eds))) 
    


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

    #calucation of the adjacent statistics. 
    #getCINstatistics()

    ## training the model
    # train(FLAGS.loss)

    ## test the model
    test(FLAGS.loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')

    parser.add_argument('-g', '--gpu', default='0', help="Assigned GPU for running the code")

    parser.add_argument('-i', '--data_dir', default="/data/workspace/nanopore/data/chiron_data/t/" ,required= False,
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
    
    """
    if args.train_cache is None:
        args.train_cache = args.data_dir + '/train_cache_gplabel.hdf5'
    if (args.valid_cache is None) and (args.validation is not None):
        args.valid_cache = args.data_dir + '/valid_cache_gplabel.hdf5'
    """
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    run(args)


