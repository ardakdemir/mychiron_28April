import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv1D,Conv2D,Dense, Dropout,Flatten,Bidirectional, Activation,BatchNormalization
from keras.layers import TimeDistributed
from keras.optimizers import SGD
from keras.models import Sequential,Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.backend import ctc_decode, variable,get_value
import keras.backend as K
from keras import *
from keras import callbacks,losses
from keras.layers import Dense, Activation,Input,LSTM, Lambda
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_module
#from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_module
from my_ctc_utils import *
import matplotlib
matplotlib.use('Agg')
import pickle
import matplotlib.pyplot as plt
import sys

from collections import Counter
from random import shuffle

import os
import logging
import datetime
import argparse
from evaluate import evaluate_preds,evaluate_preds2,editDistance

from read_data import read_h5,read_from_dict, split_data,read_raw_into_segments,unet_loading_data
CB = [callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="auto", restore_best_weights=True),keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')]

n = 300000
test_size = n/10
class_num = 5
rnn_layers = 5
cnn_filter_num = 256
window_len = 3
res_layer_num = 5
hidden_num = 200
batch_size = 32
epoch_num = 3
seq_len = 300
max_nuc_len = 48
pickle_path = "toy_data.pk"

# #max_nuc_len is interesting
def create_model(input_shape=(300,1),cnn_filter_num =256,window_len = 3,res_layers = 5,rnn_layers = 5,rnn_hidden_num = 200,class_num=5,max_nuc_len = 48):
    inputs = Input(shape=input_shape)
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    outputs = chiron_cnn(inputs,cnn_filter_num,1,window_len,res_layers = res_layers)
    outputs2 = chiron_rnn(outputs,hidden_num = rnn_hidden_num,rnn_layers = rnn_layers, class_num = class_num)
    dense =  TimeDistributed(Dense(class_num))(outputs2)
    preds = TimeDistributed(Activation('softmax',name = 'softmax'))(dense)
    labels = Input(name='the_labels', shape=[max_nuc_len], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([preds,labels,input_length,label_length])
    model3 = Model(inputs= [inputs,labels,input_length,label_length],outputs=loss_out)
    return inputs,input_length,outputs,outputs2,dense,preds,labels,input_length,label_length,loss_out,model3

def delete_blanks(preds,key = -1):
    deleted_preds = []
    for pred in preds:
        if key in pred:
            deleted_preds.append(pred[:np.where(pred==key)[0][0]])
        else:
            deleted_preds.append(pred)
    return deleted_preds

def delete_blanks_2(preds,key = 4):
    deleted_preds = []
    for pred in preds:
        if key in pred:
            deleted_preds.append(pred[:pred.index(key)])
        else:
            deleted_preds.append(pred)
    return deleted_preds
def test_model(load_type,model_path,test_name,rnn_layers=5, read_raw = False,test_size = 100,sample_num = 100,seq_len = 300,out_file = "results.txt"):

    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    #x_data = np.array(pickle.load(open('x_data.pk',"rb")))
    #shuffle(x_data)
    #y_data = pickle.load(open('y_data.pk',"rb"))
    #x_data = x_data[:test_size].reshape(len(x_data[:test_size]),300,1)
    test_folder = ""
    if read_raw:
        print("Reading raw data")
        x_tr, y_labels , label_lengths,max_label_length= read_raw_into_segments(test_name,seq_length = seq_len, sample_num = sample_num,y_pad = 4)
        print(y_labels[:100])
        print("Number of segments used for testing: %d"%len(x_tr))
    else:
        #h5_dict = read_h5(test_folder,test_name,example_num = test_size)
        #x_tr,y_tr,y_categorical,y_labels,label_lengths = read_from_dict(h5_dict,example_num = test_size , class_num = 5 , seq_len = 300 ,padding = True)
        #assert len(x_tr)== len(y_tr) == len(y_categorical )== len(y_labels) == len(label_lengths), "Dimension not matched"
        #print("Reading h5 data")
        #print(len(x_tr[0]))
        print("Reading h5 data")
        #h5_dict = read_h5(test_folder,inputpath,example_num = size)
        #x_tr,y_tr,y_categorical,y_labels,label_lengths = read_from_dict(h5_dict,example_num = size , class_num = 5 , seq_len = 300 ,padding = True)
        X,seq_lens,label,label_vec,label_seg,label_raw ,label_new= unet_loading_data(test_name)
        example_num = X.shape[0]
        x_tr = X.reshape(example_num,seq_len,1)
        y_labels = label_raw
        label_lengths = [len(label_raw[i])for i in range(len(label_raw))]
        print(x_tr.shape)
        print(len(y_labels))
        print(y_labels[0])
    if  load_type == 0:
        model = load_model(model_path)
        layer_name = 'softmax'
        preds = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        #preds = K.function([model.input],
        #                          [model.get_layer(layer_name).output])
    else:
        inputs,input_length,outputs,outputs2,dense,preds,labels,input_length,label_length,loss_out,model = create_model(rnn_layers=rnn_layers)
        model.load_weights(model_path)
    #model.summary()
    #preds.summary()
    ##read data from h5 file
    flattened_input_x_width = keras.backend.squeeze(input_length, axis=-1)
    top_k_decoded, _ = my_ctc_decode(preds, flattened_input_x_width,greedy=False,
    beam_width=50,
    top_paths=1)
    decoder = K.function([inputs, flattened_input_x_width], [top_k_decoded[0]])
    inputs = np.array(x_tr).reshape(len(x_tr),seq_len,1)
    shapes = [len(x_tr[0])for i in range(len(x_tr))]
    decoded = []
    test_batch_size = 32
    for i in range(int(len(inputs)/test_batch_size)):
        decoded_ = decoder([inputs[i*test_batch_size:(i+1)*test_batch_size],shapes[i*test_batch_size:(i+1)*test_batch_size]])[0]
        #print(i)
        for d in decoded_:
            decoded.append(d)
    #print(inputs.shape)
    #print(inputs[0])
    #print(len(y_labels[0]))
    #print(y_labels.shape)
    #decoded = decoder([inputs, shapes])[0]
    #print(decoded)
    compact_preds = delete_blanks(decoded)
    if read_raw:
        compact_truths = delete_blanks_2(y_labels,key=4)
    else:
        compact_truths = delete_blanks(y_labels,key=4)
    #print(compact_preds)
    #print(compact_truths)
    samp = 100
    for p,t in zip(compact_preds[:samp],compact_truths[:samp]):
      print("pred:")
      print(p)
      print("truth:")
      print(t)
    lens = Counter()
    lens.update(label_lengths)
    print(lens)
    out = open(out_file,"w")
    mean,std = evaluate_preds(compact_preds, compact_truths)
    mean2,std2 = evaluate_preds2(compact_preds, compact_truths)
    print("Average normalized edit distance : %0.5f"%mean )
    print("Standard deviation: %0.5f"%std )
    print("My edit distance")
    print("Average normalized edit distance : %0.5f"%mean2 )
    print("Standard deviation: %0.5f"%std2 )
    out.write("Average normalized edit distance : %0.5f\n"%mean )
    out.write("Standard deviation: %0.5f\n"%std )
    out.close()
    return 0

##read nucleotide sequence for each x,y pair and store in arrays
## pad the nucleotide sequences into max_length with 4 (denoting blank)
def read_pickle(pickle_path,example_num = 100 , class_num = 5 , seq_len = 300 ,padding = True):
    all_data = pickle.load(open(pickle_path,"rb"))
    keys = list(all_data.keys())
    x_tr = []
    y_tr = []
    labels = []
    print(example_num)
    label_lengths = []
    for key in keys:
        x_tr.append(all_data[key]['x_data'])
        y_tr.append(all_data[key]['y_vec'])
        labels.append(np.array(all_data[key]["nucleotides"])-1)
    x_train = np.array(x_tr[:example_num]).reshape(example_num,seq_len,1)
    y_train = np.array(y_tr[:example_num]).reshape(example_num,seq_len,1)
    y_labels =labels[:example_num]
    label_lengths = list(map(lambda x : len(x),y_labels))
    if padding:
        pad = 4
        max_length = max(list(map(lambda x : len(x),y_labels)))
        for i in range(len(y_labels)):
            leng = len(y_labels[i])
            y_labels[i] = np.pad(y_labels[i],(0,max_length-leng),'constant', constant_values=(4,4))
    #print(y_labels[0])
    y_train_class = keras.utils.to_categorical(y_train,num_classes = class_num)
    return x_train,y_train,y_train_class,y_labels,label_lengths


## conv-layer
def conv1D_layer(inputs,filternum,filtersize,activation='relu'):
    conv = Conv1D(filternum,filtersize,padding='same',input_shape = inputs.shape)
    x = inputs
    x = conv(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x
def chiron_cnn(inputs,filternum1,filtersize1,filtersize2,res_layers = 5,activation = 'relu'):
    x = inputs
    for i in range(res_layers):
        x = chiron_res_layer(x,filternum1,filtersize1,filtersize2,activation = activation)
    #x = Flatten()(x)
    return x
## two branches of cnn
def chiron_res_layer(inputs,filternum1,filtersize1,filtersize2,activation = 'relu'):
    x   = inputs
    b_1 = conv1D_layer(x,filternum1,filtersize1,activation = activation)
    b_2 = conv1D_layer(x,filternum1,filtersize1,activation = activation)
    b_2 = conv1D_layer(b_2,filternum1,filtersize2,activation = activation)
    b_2 = conv1D_layer(b_2,filternum1,filtersize1,activation = activation)
    y = keras.layers.add([b_1,b_2])
    y = Activation(activation)(y)
    return y

def chiron_rnn(inputs,hidden_num =200,rnn_layers = 3,class_num = class_num ):
    x = inputs
    for i in range(rnn_layers):
        x = chiron_bilstm_layer(x,hidden_num = hidden_num)
    #FC = Dense(class_num,activation = 'softmax',input_shape=(hidden_num*2,))
    #x = FC(x)
    return x
def chiron_bilstm_layer(inputs,hidden_num):
    firstbi = Bidirectional(LSTM(hidden_num, return_sequences=True),
                        input_shape=inputs.shape)
    x = inputs
    x = firstbi(x)
    x = BatchNormalization()(x)
    return x

# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return my_ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_predict(model,inputs,beam_width = 100, top_paths = 1):
    lens = lambda l :list(map (lambda x:len(x),l))
    preds = model.predict(inputs)
    #print(preds)
    if top_paths !=1:
        decoded_preds = ctc_decode(preds,lens(inputs),greedy=False,beam_width=beam_width,top_paths=top_paths)
    else:
        decoded_preds = ctc_decode(preds,lens(inputs),beam_width=beam_width)
    return decoded_preds

def evaluation(args):
    global Flags

    Flags = args
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    test_model(Flags.loadtype,Flags.model,Flags.input, read_raw = Flags.readraw, sample_num= Flags.samplenum,test_size = Flags.size,out_file  = Flags.out_file )

def train():
    loadtype = FLAGS.savetype
    readraw  = FLAGS.readraw
    inputpath = FLAGS.input
    seq_length = FLAGS.sequence_len
    model_name = FLAGS.model
    size = FLAGS.size
    epochs = FLAGS.epoch_num
    #dev_size = int(size/10)
    batch_size = FLAGS.batch_size
    epoch_num = FLAGS.epoch_num
    #train_size = int(size - dev_size)
    test_folder = ""
    if readraw:
        print("Reading raw data")
        x_tr, y_labels , label_lengths,max_label_length= read_raw_into_segments(inputpath,seq_length = seq_len, sample_num = size,y_pad = 4)
        x_tr = np.array(x_tr).reshape(len(x_tr),seq_len,1)
    else:

        #return X, seq_len, label, label_vec, label_seg, label_raw
        print("Reading h5 data")
        #h5_dict = read_h5(test_folder,inputpath,example_num = size)
        #x_tr,y_tr,y_categorical,y_labels,label_lengths = read_from_dict(h5_dict,example_num = size , class_num = 5 , seq_len = 300 ,padding = True)
        X,seq_lens,label,label_vec,label_seg,label_raw ,label_new= unet_loading_data(inputpath)
        example_num = X.shape[0]
        x_tr = X.reshape(example_num,seq_len,1)
        y_labels = label_raw
        label_lengths = [len(label_raw[i])for i in range(len(label_raw))]
        print(x_tr.shape)
        print(len(y_labels))
        print(y_labels[0])
        #assert len(x_tr)== len(y_tr) == len(y_categorical )== len(y_labels) == len(label_lengths), "Dimension not matched"
    input_shape = x_tr.shape[1:]
    max_nuc_len = max(label_lengths)
    print(max_nuc_len)
    print(y_labels[0])
    all_model = create_model(input_shape=input_shape,cnn_filter_num =256,window_len = 3,res_layers = 5,rnn_layers = 5,rnn_hidden_num = 200,class_num=5,max_nuc_len = max_nuc_len)
    inputs,input_length,outputs,outputs2,dense,preds,labels,input_length,label_length,loss_out,model = all_model
    flattened_input_x_width = keras.backend.squeeze(input_length, axis=-1)
    top_k_decoded, _ = my_ctc_decode(preds, flattened_input_x_width)
    decoder = K.function([inputs, flattened_input_x_width], [top_k_decoded[0]])
    #model3 = Model(inputs= [inputs,labels,input_length,label_length],outputs=loss_out)
    model.summary()
    model.compile(loss = {'ctc': lambda y_true, y_pred: y_pred},optimizer = Adam())
    print(x_tr.shape)
    train_x = x_tr
    train_y_labels= y_labels
    train_input_lengths = np.array([seq_length for i in range(x_tr.shape[0])])
    train_label_lengths = np.array(label_lengths)
    #dev_input_lengths = np.array([seq_length for i in range(dev_size)])
    #dev_label_lengths = np.array(label_lengths[:dev_size])
    outputs = {'ctc': np.zeros(x_tr.shape[0])}
    fig=plt.figure()
    print(model_name)
    history = model.fit([train_x,np.array(train_y_labels),np.array(train_input_lengths),np.array(train_label_lengths)],outputs,batch_size = batch_size,callbacks=CB,epochs=epoch_num,validation_split=0.2)
    #model.save( "model/" + model_name + ".h5")
    model.save_weights(model_name+"_weights.h5")
    figureSavePath=model_name + ".png"
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
	  #plt.title("Training curve of the whole training set " + lossType)
    plt.savefig(figureSavePath)
    plt.close("all")

def run_train(args):
    global FLAGS
    FLAGS = args
    train()



def main(arguments=sys.argv[1:]):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    currentDT = datetime.datetime.now()
    current_time = currentDT.strftime("%Y%m%d.%H%M%S")
    parser = argparse.ArgumentParser(prog='UoTchiron', description='A deep neural network basecaller.')
    subparsers = parser.add_subparsers(title='sub command', help='sub command help')

    parser_call = subparsers.add_parser('test', description='Segmented basecalling', help='Perform basecalling.')
    parser_call.add_argument('-i', '--input', required=True, help= "File path to the cached_data in h5 format or folder of the raw signals and labels.")
    parser_call.add_argument('-t', '--loadtype', default = 0,type=int,help="Binary value 0 for loading model directly 1 for loading weights")
    parser_call.add_argument('-r', '--readraw', default = False,type=bool,help="Boolean False for reading from h5 True for reading raw")
    parser_call.add_argument('-m', '--model', required=True, help="File path to the model file or the model weight file in h5 format.")
    parser_call.add_argument('-s', '--size',  type = int , default = '100', help="Number of segments to be read from the input file")
    parser_call.add_argument('-ss', '--samplenum',  type = int , default = 1, help="Number of signals to be read from the input file")
    parser_call.add_argument('-o', '--out_file', type= str, default = "scores.txt", help="File name to output scores.")

    parser_call.add_argument('-g', '--gpu', type=str, default='0', help="GPU ID")
    parser_call.add_argument('--beam', type=int, default=50, help="Beam width used in beam search decoder")
    parser_call.set_defaults(func=evaluation)


    parser_train = subparsers.add_parser('train', description='Segmented basecalling', help='Train a  basecaller.')
    parser_train.add_argument('-i', '--input', required=True, help= "File path to the cached_data in h5 format or folder of the raw signals and labels.")
    parser_train.add_argument('-r', '--readraw', default = False,type=bool,help="Boolean False for reading from h5 True for reading raw")
    parser_train.add_argument('-t', '--savetype', default = 0,type=int,help="Binary value 0 for saving model directly 1 for saving weights")
    parser_train.add_argument('-m', '--model', default = "model%s"%current_time, help="File name of the model file or the model weight file in h5 format.")
    parser_train.add_argument('-s', '--size',  type = int,default = 10,  help="Number of samples to be read from the input file")
    parser_train.add_argument('-o', '--out_file', default = "scores.txt", help="File name to output scores.")
    parser_train.add_argument('-b', '--batch_size', default = 32, help="Batch size")
    parser_train.add_argument('-e', '--epoch_num',type=int, default = 10, help="Epoch number")
    parser_train.add_argument('-l', '--sequence_len', type=int, default=300, help="Segment length to be divided into.")
    parser_train.add_argument('-g', '--gpu', type=str, default='0', help="GPU ID")

    parser_train.set_defaults(func=run_train)

    args = parser.parse_args(arguments)

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(sys.argv[1:])
    main()
    exit()
    test = 1
    out_file = "scores_new"
    model_weight_path = "model3_weights.h5"
    test_folder = "../../work/data/cache/"
    test_file = "train_cache.h5"
    if test == 1:
        test_model(model_weight_path,test_folder,test_file)
        exit()
    args = sys.argv
    with_ctc = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    evaluate = 0
    h5file_path = "../../work/data/cache/train_cache.h5"
    if len(args)>1:
        h5file_path = args[1]
        #with_ctc = int(args[2])
    h5_dict = read_h5("",h5file_path,example_num = n)
    x_tr,y_tr,y_categorical,y_labels,label_lengths = read_from_dict(h5_dict,example_num = n , class_num = 5 , seq_len = 300 ,padding = True)

    #(x_tr,y_tr,y_categorical,y_labels,label_lengths ),(test_x_tr,test_y_tr,test_y_categorical,test_y_labels,test_label_lengths ) = split_data((x_tr,y_tr,y_categorical,y_labels,label_lengths),test_size)
    assert len(x_tr)== len(y_tr) == len(y_categorical )== len(y_labels) == len(label_lengths), "Dimension not matched"
    #len(test_x_tr)
    inputs = Input(shape=x_tr.shape[1:])
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    outputs = chiron_cnn(inputs,cnn_filter_num,1,window_len)
    outputs2 = chiron_rnn(outputs,hidden_num)
    dense =  TimeDistributed(Dense(class_num))(outputs2)
    preds = TimeDistributed(Activation('softmax',name = 'softmax'))(dense)
    #print(np.ones(outputs2.shape[1])*int(outputs2.shape[2]))
    model2 = Model(inputs= inputs,outputs=preds)
    sgd = SGD()
    model2.summary()




    ##ctc decoding only used during prediction
    ## during training cross-entropy is used to calculate loss over each softmax output
    ## works fine
    if with_ctc == 0:
        model2.compile(loss = "categorical_crossentropy",optimizer = sgd)
        model2.fit(x_tr,y_categorical,batch_size = batch_size,epochs=epoch_num)
        decodeds = ctc_predict(model2,x_tr[:10])
        vals = get_value(decodeds[0][0])
        print(vals)

    ## ctc_batch_cost is used during training
    ## now obtaining too high loss values and model only predicts 1
    else:
        # Define CTC loss
        max_nuc_len = len(y_labels[0])
        labels = Input(name='the_labels', shape=[max_nuc_len], dtype='int32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([preds,labels,input_length,label_length])
