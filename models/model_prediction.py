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
# Final result prediction
########################################
def editDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min(distances[i1], distances[i1+1], distances_[-1]))
        distances = distances_

    return distances[-1]/(len(s2))


def toBases(logits, seq_len = None):
    
    idx = 0
    seq, count = [], []
    
    if seq_len == None:
        seq_len = len(logits)

    while(idx < seq_len):
        current = logits[idx]
        tmp_count = 1
        while( idx + 1 < seq_len and logits[idx+1] == current ): 
            idx += 1
            tmp_count += 1
            
        seq.append(current)
        count.append(tmp_count)
        idx += 1

    return(seq)

def ed(logits, seq_len, gold):

    ## calcuate the editor distannce between predicted and gold label
    seq = toBases(logits, seq_len)
    return editDistance(seq, gold)/len(gold)



if __name__ == "__main__":

    logits = [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 5]
    gold =   [0, 1, 0, 1, 2, 2 ]

    print(ed(logits, 6, gold))



