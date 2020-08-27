# import packages
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



class Bert2RNN():

    def __init__(self, metadata, bert_model = None, hidden_layers_dialogue = 128, dialogue_dropout_rate = 0.0, 
                 hidden_layers_labels = 64, labels_dropout_rate = 0.0, labels_recurrent_dropout = 0.0,
                dialogue_recurrent_dropout = 0.0,optimizer = None, optimizer_algorithm = None):

        self.metadata = metadata
        self.bert_model = bert_model
        
        self.hidden_layers_dialogue = hidden_layers_dialogue
        self.dialogue_dropout_rate = dialogue_dropout_rate
        self.dialogue_recurrent_dropout = dialogue_recurrent_dropout

        self.hidden_layers_labels = hidden_layers_labels
        self.labels_dropout_rate = labels_dropout_rate
        self.labels_recurrent_dropout = labels_recurrent_dropout

        self.optimizer = optimizer
        if self.optimizer == 'adam':
            self.optimizer_algorithm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif self.optimizer == 'RMSprop':
            self.optimizer_algorithm = RMSprop(lr=0.001, rho = 0.9) 
        elif self.optimizer == 'SGD':
            self.optimizer_algorithm = SGD(lr=0.1)

    def create_model(self,):

        def slice_two(x, aim, max_utterance_len):
            """
            Small helper function that slices the concatenated
            input back to the word and character data
            """

            if aim == 'zero':
                return x[:,(max_utterance_len+2):]
            elif aim == 'one':
                return x[:,:(max_utterance_len+2)]


        max_utterance_len = self.metadata['max_utterance_len']
        max_word_len = 27# self.metadata['max_word_len']
        vocabulary_size = self.metadata['vocabulary_size']
        num_da_tags = self.metadata['num_da_tags']

        alphabet_size = 69

        # embedding inputs
        input_shape = Input(shape=(2*(max_utterance_len+2),))
        in_token = keras.layers.Lambda(slice_two, arguments = {'aim': 'zero', 'max_utterance_len': max_utterance_len})(input_shape)
        in_segment = keras.layers.Lambda(slice_two, arguments = {'aim': 'one', 'max_utterance_len': max_utterance_len})(input_shape)
             
        bert_inputs = [in_token, in_segment]

        # Instantiate the custom Bert Layer defined above
        bert_output = self.bert_model(bert_inputs)

        slice_bert = keras.layers.Lambda(lambda x: x[:,1,:])(bert_output) #retrieve the embeddings for [cls]

        model1 = Model(input_shape, slice_bert)

        dialogue_input_segments = Input(shape=(None, (max_utterance_len+2),))
        dialogue_input_tokens = Input(shape=(None, (max_utterance_len+2),))
        dialogue_input = concatenate([dialogue_input_segments, dialogue_input_tokens])
        model_input = [dialogue_input_segments, dialogue_input_tokens]

        output_utterance_bigru =TimeDistributed(model1)(dialogue_input)

        LSTM_dialogue = LSTM(self.hidden_layers_dialogue, dropout=self.dialogue_dropout_rate, recurrent_dropout=self.dialogue_recurrent_dropout, return_sequences=True) 
        utterance_bilstm = Bidirectional(LSTM_dialogue)(output_utterance_bigru)

        # prediction
        preds = Dense(self.metadata['num_da_tags'], activation='softmax')(utterance_bilstm)

        LSTM_labels = LSTM(self.hidden_layers_labels, dropout=self.labels_dropout_rate, recurrent_dropout= self.labels_recurrent_dropout, return_sequences=True) #, kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform')
        label_LSTM = Bidirectional(LSTM_labels)(preds)

        final_preds = Dense(num_da_tags, activation='softmax')(label_LSTM)

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