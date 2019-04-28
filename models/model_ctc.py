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
# 2. RCNN+CTC
########################################
## CTC loss, decodes needed

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    #y_pred = y_pred[:,2,:]
    #return tf.nn.ctc_loss(labels, y_pred, label_length, ctc_merge_repeated=True, time_major=False, ignore_longer_outputs_than_inputs=True)
    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length,)

## Network output prediction

def decoder_ctc(out, top_path=1):
    results = []
    beam_width=5
    
    if beam_width < top_path:
        beam_width = top_path
    
    for i in range(top_path):
        labels = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],\
                greedy=False, beam_width=beam_wdith, top_paths=top_path)[0][i])[0]
        results.append(labels)
    

    return results



## Network structure
def CTC(signals, out_len, label_len, kernel_size, conv_window_len, lr, dropoutRate, training=True):

    inner = signals
    n_label = 5
    
    # CNN
    inner = layers.Conv1D(kernel_size[0], conv_window_len, padding='same', activation="relu")(inner)
    inner = layers.Conv1D(kernel_size[0], conv_window_len, padding='same',activation="relu")(inner)
    inner = layers.MaxPooling1D(2)(inner)

    inner = layers.Conv1D(kernel_size[1], conv_window_len, padding='same',activation="relu")(inner)
    inner = layers.Conv1D(kernel_size[1], conv_window_len, padding='same',activation="relu")(inner)
    inner = layers.MaxPooling1D(2)(inner)

    
    # CNN to RNN
    #print(inner.get_shape())
    inter = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(inner)

    # 2"RNN layer
    
    gru_1 = layers.GRU(128, return_sequences=True, kernel_initializer='he_normal')(inner)
    gru_1b = layers.GRU(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(inner)
    gru1_merged = layers.Add()([gru_1, gru_1b])

    gru_2 = layers.GRU(128, return_sequences=True, kernel_initializer='he_normal')(gru1_merged)
    gru_2b = layers.GRU(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(gru1_merged)
    gru2_merged = layers.Concatenate()([gru_2, gru_2b])
    
    inner = gru2_merged


    inner = layers.Dense(n_label)(inner)
    y_pred = layers.Activation('softmax')(inner)

    # loss compute module
    labels = Input(name="labels", shape=[label_len],dtype='int64')
    input_length = Input(name="input_length", shape=[1], dtype='int64')
    label_length = Input(name="label_length", shape=[1],dtype='int64')

    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    if training:
        return models.Model([signals, labels, input_length, label_length], outputs=loss_out)
    else:
        return models.Model(inputs=[signals], outputs=y_pred)



