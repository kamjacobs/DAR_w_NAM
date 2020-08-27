# import packages

from models.attention_layers.multihead_attention import MultiHeadAttention
from models.attention_layers.cs_attention import CSAttention

import tensorflow as tf
from tensorflow.python.keras import backend as K

from gensim.models import KeyedVectors
import numpy as np
from itertools import chain

import time
import datetime
import random, os, sys
import numpy as np
import pickle
import torch.nn as nn

import skopt # !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer  

import keras
from keras.models import *
from keras.layers import *
from keras.engine.topology import Layer
from keras.callbacks import *
from keras.initializers import *
from keras.optimizers import RMSprop, Adam, Adadelta, SGD
from keras import initializers, regularizers, constraints
from keras.initializers import RandomUniform
from keras_contrib.layers import CRF
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences



class MHACsaCRF():

    def __init__(self, metadata, word_embd, word_embd_dim=None,word_embd_matrix_google=None, word_embd_matrix_train200=None,
                word_embd_matrix_train300 = None, cnn_dropout = 0.0, kernel_size = 3, filters = 30,
                hidden_layers_utterance = 128, utterance_dropout_rate = 0.0, utterance_recurrent_dropout = 0.0,
                hidden_layers_dialogue = 128, dialogue_dropout_rate = 0.0, dialogue_recurrent_dropout = 0.0,
                optimizer = None, heads = 8, optimizer_algorithm = None):

        self.metadata = metadata
        self.word_embd = word_embd
     

        if self.word_embd == 'google':
            self.word_embd_dim = 300
            self.word_embd_matrix = word_embd_matrix_google
        elif self.word_embd == 'train200':
            self.word_embd_dim = 200
            self.word_embd_matrix = word_embd_matrix_train200
        elif self.word_embd == 'train300':
            self.word_embd_dim = 300
            self.word_embd_matrix = word_embd_matrix_train300

        self.cnn_dropout = cnn_dropout
        self.kernel_size = kernel_size
        self.filters = filters

        self.hidden_layers_utterance = hidden_layers_utterance
        self.utterance_dropout_rate = utterance_dropout_rate
        self.utterance_recurrent_dropout = utterance_recurrent_dropout

        self.hidden_layers_dialogue = hidden_layers_dialogue
        self.dialogue_dropout_rate = dialogue_dropout_rate
        self.dialogue_recurrent_dropout = dialogue_recurrent_dropout

        self.heads = heads

        self.optimizer = optimizer
        if self.optimizer == 'adam':
            self.optimizer_algorithm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif self.optimizer == 'RMSprop':
            self.optimizer_algorithm = RMSprop(lr=0.001, rho = 0.9) 
        elif self.optimizer == 'SGD':
            self.optimizer_algorithm = SGD(lr=0.1)

    def create_model(self,):

        def slice_one(x, aim):
            """
            Small helper function that slices the concatenated
            input back to the word and character data
            """
            if aim == 'word':
                return x[:,:,0]
            elif aim == 'char':
                return x[:,:,1:]


        max_utterance_len = self.metadata['max_utterance_len']
        max_word_len = 27# self.metadata['max_word_len']
        vocabulary_size = self.metadata['vocabulary_size']
        num_da_tags = self.metadata['num_da_tags']

        alphabet_size = 69

        # embedding inputs
        input_shape = Input(shape=(max_utterance_len,max_word_len+1)) 
        word_input = keras.layers.Lambda(slice_one, arguments={'aim': 'word'})(input_shape)
        character_input = keras.layers.Lambda(slice_one, arguments={'aim': 'char'})(input_shape)


        # word embedding layer
        word_embedding = Embedding(input_dim = vocabulary_size+1,
                                        output_dim = self.word_embd_dim,
                                        input_length = max_utterance_len,
                                        weights = [self.word_embd_matrix])(word_input)


        # character embedding layer
        embed_char_out = TimeDistributed(Embedding(alphabet_size,
                                            max_word_len,
                                            embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)))(character_input)
        dropout = Dropout(self.cnn_dropout)(embed_char_out) 
        conv1d_out = TimeDistributed(Conv1D(kernel_size = self.kernel_size, filters = self.filters, 
        									padding = 'same', activation = 'relu', strides = 1))(dropout)
        max_pool_out = TimeDistributed(MaxPooling1D(max_word_len))(conv1d_out)
        char_embedding = TimeDistributed(Flatten())(max_pool_out)

        # concatenate
        output = concatenate([char_embedding, word_embedding])

        LSTM_dialogue = LSTM(self.hidden_layers_utterance, dropout=self.utterance_dropout_rate, 
        					recurrent_dropout=self.utterance_recurrent_dropout, return_sequences=True) 
        utterance_bilstm = Bidirectional(LSTM_dialogue)(output)

        h = self.heads
        d_model = int(utterance_bilstm.shape[2])
        d_k = round(d_model / h)
        d_v = round(d_model / h)
        x, slf_attn = MultiHeadAttention(n_head=h, d_model=d_model, d_k = d_k, d_v = d_v,dropout=0.0)(utterance_bilstm, utterance_bilstm, utterance_bilstm)

        model1 = Model(inputs=input_shape, outputs=x)


        dialogue_input_word = Input(shape=(None,max_utterance_len,1))
        dialogue_input_char = Input(shape=(None,max_utterance_len, max_word_len,))
        dialogue_input = concatenate([dialogue_input_word, dialogue_input_char])
        model_input = [dialogue_input_word, dialogue_input_char]

        output_utterance_bilstm =TimeDistributed(model1)(dialogue_input)

        utterance_att_vec, utt_att_vec_coeff = CSAttention(metadata= self.metadata, hu_utterance = self.hidden_layers_utterance, hu_dialogue = 2*self.hidden_layers_utterance,
                                                                hu_fc = 2*self.hidden_layers_utterance, return_coefficients=True)(output_utterance_bilstm)

        LSTM_dialogue = LSTM(self.hidden_layers_dialogue, dropout=self.dialogue_dropout_rate, recurrent_dropout=self.dialogue_recurrent_dropout, return_sequences=True) 
        utterance_bilstm = Bidirectional(LSTM_dialogue)(utterance_att_vec)

        crf = CRF(num_da_tags, learn_mode='marginal')
        final_preds = crf(utterance_bilstm)

        model2 = Model(model_input,final_preds)

        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))

        model2.compile(loss='categorical_crossentropy', optimizer = self.optimizer_algorithm, metrics =['acc', precision_m, recall_m, f1_m])

        return model2