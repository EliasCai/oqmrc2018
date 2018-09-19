# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Input, concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D,GlobalMaxPooling1D
from keras.layers.wrappers import Bidirectional, TimeDistributed
import keras.backend as K
import numpy as np
from Attention import *


def test_embde(x, embedding_matrix):

    maxlen = x.shape[1]
    embedding_layers = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 input_length=maxlen,
                                 # mask_zero=False,
                                 trainable=False)  
      
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = embedding_layers(sequence)
    model = Model(inputs=sequence, 
                  outputs=embedded)
                  
    model.compile(loss='binary_crossentropy',# 'mae', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model
    
def one_model(x, embedding_matrix):
    maxlen = x.shape[1]
    embedding_layers = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 input_length=maxlen,
                                 # mask_zero=True,
                                 trainable=False)  
      
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = embedding_layers(sequence)

    blstm = Bidirectional(LSTM(maxlen * 2, return_sequences=True), 
                               merge_mode='sum')(embedded)
    blstm = Dropout(0.2)(blstm)
    blstm = Bidirectional(LSTM(maxlen,return_sequences=True), 
                               merge_mode='sum')(blstm)
    blstm = Dropout(0.2)(blstm)
    # blstm = Bidirectional(LSTM(max(2, maxlen // 2)), 
                          # merge_mode='sum')(blstm)
    out = Attention(return_attention=False)(blstm)                  
    # out = Bidirectional(LSTM(max(2, maxlen // 2)), 
                             # merge_mode='sum')(blstm)
    
    
    return sequence, out
    
def get_model(x_inputs, embedding_matrix):
        
    sequence_list = []
    out_list = []
    for (s, o) in [one_model(x, embedding_matrix) for x in x_inputs]:
        sequence_list.append(s)
        out_list.append(o)
        
    out = concatenate(out_list)
    out = Dense(32)(out)
    out = Dense(3)(out)
    out = Activation('relu')(out)
    model = Model(inputs=sequence_list, 
                  outputs=out)
                  
    model.compile(loss='categorical_crossentropy',# 'mae', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


def one_model_cnn(x, embedding_matrix):
    maxlen = x.shape[1]
    embedding_layers = Embedding(embedding_matrix.shape[0],
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 input_length=maxlen,
                                 # mask_zero=True,
                                 trainable=False)  
      
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = embedding_layers(sequence)
    blstm = Bidirectional(LSTM(maxlen, return_sequences=True), 
                               merge_mode='sum')(embedded)
    # blstm = Dropout(0.5)(blstm)
    # blstm = Attention(return_attention=False)(blstm)  
    filter_sizes = [1, 3, 5, 7, 15, 30, 50]
    convs = []

    for filter_size in filter_sizes:
        conv = Conv1D(filters=128, 
                      kernel_size=filter_size, 
                      padding='same', 
                      activation='relu')(blstm)
        conv = Conv1D(filters=128, 
                      kernel_size=filter_size, 
                      padding='same', 
                      activation='relu')(conv)
        conv = MaxPooling1D(2)(conv)
        convs.append(conv)
    out = concatenate(convs, axis=1)
    return sequence, out    
    
def get_model_cnn(x_inputs, embedding_matrix):
        
    sequence_list = []
    out_list = []
    x = np.hstack(x_inputs)
    # for (s, o) in [one_model_cnn(x, embedding_matrix)]:
        # sequence_list.append(s)
        # out_list.append(o)
    seq, out = one_model_cnn(x, embedding_matrix) 
    # out = concatenate(out_list,axis=1)
    out = MaxPooling1D()(out)
    out = Flatten()(out)
    out = Dropout(0.5)(out)
    # out = Dense(128)(out)
    # out = Dropout(0.5)(out)
    out = Dense(32)(out)
    out = Dropout(0.5)(out)
    out = Dense(3)(out)
    out = Activation('softmax')(out)
    model = Model(inputs=seq, 
                  outputs=out)
                  
    model.compile(loss='categorical_crossentropy',# 'mae', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model
    
if __name__ == '__main__':  
    
    K.clear_session()
    
    x = np.zeros((1000,128))
    embedding_matrix = np.zeros((20000,128))
    # model = get_model([x, x], embedding_matrix)
    # model.summary()
    model = get_model_cnn([x, x], embedding_matrix)
    model.summary()
    

