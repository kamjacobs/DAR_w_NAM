print('==== start importing packages ====')

from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# # Check TensorFlow Version
# assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use Tensorflow version 1.0 or newer. You are using {}'.format(tf.__version__)
# print('Tensorflow Version: {}'.format(tf.__version__))

# # Check for GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please ensure you have installed TensorFlow correctly')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import os
from collections import Counter
import time
import datetime
from datetime import datetime
from gensim.models import KeyedVectors
import numpy as np
from itertools import chain
import random, os, sys


import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations

# import DAR_model_new
import final_new_models2
import data_generators

import keras
from keras import backend as K

from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Lambda, merge, Input, Reshape, Embedding, Dropout, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Flatten, TimeDistributed,Concatenate
from keras.optimizers import RMSprop
from keras.models import Model, load_model
from keras.layers import Conv1D, ThresholdedReLU, MaxPooling1D, Activation, concatenate # TimeDistributedDense
from keras.initializers import RandomUniform
from keras_contrib.layers import CRF
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.callbacks import EarlyStopping
from keras.engine.topology import Layer

from keras_bert import load_trained_model_from_checkpoint

from models.mimic_raheja import RahejaModel
from models.pool_crf import PoolCRF
from models.att_crf import AttCRF
from models.mha_pool_crf import MHAPoolCRF
from models.mha_att_crf import MHAAttCRF
from models.mha_csa_crf import MHACsaCRF
from models.pool_2rnn import Pool2RNN
from models.att_2rnn import Att2RNN
from models.csa_2rnn import CSA2RNN
from models.bert_crf import BertCRF
from models.bert_2rnn import Bert2RNN



# specify corpus and model
data = 'MRDA' 			 #['SwDA', 'MRDA']
model_name = 'pool-crf'  #['raheja_tetreault', 'pool-crf', 'att-crf', 'mha-pool-crf', 'mha-att-crf', 'mha-csa-crf', 'pool-2rnn', 'att-2rnn', 'csa_2rnn', 'bert-crf', 'bert-2rnn']

if data == 'MRDA':
	data_dir = "C://Users//31642//anaconda3//envs//thesis_dar//KimThesis//processed_data/MRDA/"
	embedding_dir = "C://Users//31642//anaconda3//envs//thesis_dar//KimThesis//embeddings/MRDA/"
elif data == 'SwDA':
	data_dir = "C://Users//31642//anaconda3//envs//thesis_dar//KimThesis//processed_data/SwDA/"
	embedding_dir = "C://Users//31642//anaconda3//envs//thesis_dar//KimThesis//embeddings/SwDA/"

print('======LOAD DATA =====')
# metadata
metadata = pickle.load(open(data_dir + "metadata.pkl", "rb"))
word_embd_matrix_train300 = pickle.load(open(embedding_dir + 'embd_matrix_train_300D.pkl', 'rb'))

# embeddings
embd_matrix_google = pickle.load(open(embedding_dir + 'embd_matrix_pre_train_Google_300D.pkl', 'rb'))
embd_matrix_train200 = pickle.load(open(embedding_dir + 'embd_matrix_train_200D.pkl', 'rb'))
embd_matrix_train300 = pickle.load(open(embedding_dir + 'embd_matrix_train_300D.pkl', 'rb'))


# load training, test and validation sets
train_x =  pickle.load(open(data_dir + "train_x.pkl", "rb"))
train_y = pickle.load(open(data_dir + "train_y.pkl", "rb"))

test_x = pickle.load(open(data_dir + "test_x.pkl", "rb"))
test_y = pickle.load(open(data_dir + "test_y.pkl", "rb"))

valid_x = pickle.load(open(data_dir + "valid_x.pkl", "rb"))
valid_y = pickle.load(open(data_dir + "valid_y.pkl", "rb"))

# load training, test and validation sets made for character embeddings
train_char_x =  pickle.load(open(data_dir + "train_char_x.pkl", "rb"))
test_char_x =  pickle.load(open(data_dir + "test_char_x.pkl", "rb"))
valid_char_x =  pickle.load(open(data_dir + "valid_char_x.pkl", "rb"))

#load training, test and validation sets made for BERT embeddings
train_bert_tokens = pickle.load(open(data_dir + "train_bert_tokens.pkl", "rb"))
train_bert_segments = pickle.load(open(data_dir + "train_bert_segments.pkl", "rb"))

valid_bert_tokens = pickle.load(open(data_dir + "valid_bert_tokens.pkl", "rb"))
valid_bert_segments = pickle.load(open(data_dir + "valid_bert_segments.pkl", "rb"))

test_bert_tokens = pickle.load(open(data_dir + "test_bert_tokens.pkl", "rb"))
test_bert_segments = pickle.load(open(data_dir + "test_bert_segments.pkl", "rb"))



print('==== set hyperparameter space ====')
hidden_layers_utterance = Integer(low=20, high =200, name='hidden_layers_utterance')
utterance_dropout_rate = Real(low=0.0, high=0.6, name='utterance_dropout_rate')
utterance_recurrent_dropout = Real(low=0.0, high=0.6, name='utterance_recurrent_dropout')

hidden_layers_dialogue = Integer(low=20, high =300, name='hidden_layers_dialogue')
dialogue_dropout_rate = Real(low=0.0, high=0.6, name='dialogue_dropout_rate')
dialogue_recurrent_dropout = Real(low=0.0, high=0.6, name='dialogue_recurrent_dropout')

hidden_layers_labels = Integer(low=20, high =300, name='hidden_layers_labels')
labels_dropout_rate = Real(low=0.0, high=0.6, name='labels_dropout_rate')
labels_recurrent_dropout = Real(low=0.0, high=0.6, name='labels_recurrent_dropout')

word_embd = Categorical(['google', 'train200', 'train300'], name = 'word_embd')

cnn_dropout = Real(low=0.1, high=0.6, name='cnn_dropout')
kernel_size = Categorical([3, 4, 5,6,7], name = 'kernel_size')
filters = Integer(low=15, high =100, name='filters')

heads_utterance = Categorical([4, 6, 8,10,12], name = 'heads_utterance')
heads_dialogue = Categorical([4, 6, 8,10,12], name = 'heads_dialogue')

attention_dim = Integer(low=50, high =200, name='attention_dim')

optimizer = Categorical(['adam', 'RMSprop'], name = 'optimizer')


print('========== hyperparameter tuning =======================')

r = 30 
epochs = 3

dimensions = [
    word_embd,
    cnn_dropout,
    kernel_size,
    filters,
    hidden_layers_utterance,
    utterance_dropout_rate,
    utterance_recurrent_dropout,
    hidden_layers_dialogue,
    dialogue_dropout_rate,
    dialogue_recurrent_dropout,
    optimizer
]


default_parameters = ['google', 0.1, 3, 30, 128, 0.1, 0.0,128, 0.1, 0.0,'adam']



@use_named_args(dimensions=dimensions)
def fitness(word_embd, cnn_dropout, kernel_size, filters, 
			hidden_layers_utterance, utterance_dropout_rate,
			utterance_recurrent_dropout,hidden_layers_dialogue,
			dialogue_dropout_rate, dialogue_recurrent_dropout, 
			heads_utterance,optimizer):

    # Print the hyper-parameters.
    print('word_embd:', word_embd)
    print('cnn_dropout:', cnn_dropout)
    print('kernel_size:', kernel_size)
    print('filters:', filters)
    print('hidden_layers_utterance:',hidden_layers_utterance )
    print('utterance_dropout_rate:', utterance_dropout_rate)
    print('utterance_recurrent_dropout:', utterance_recurrent_dropout)
    print('hidden_layers_dialogue:',hidden_layers_dialogue )
    print('dialogue_dropout_rate:', dialogue_dropout_rate)
    print('dialogue_recurrent_dropout:', dialogue_recurrent_dropout)
    print('optimizer:', optimizer)
    print()
    
    # Create the neural network with these hyper-parameters.
    model = PoolCRF(metadata, word_embd, word_embd_matrix_google=embd_matrix_google, word_embd_matrix_train200=embd_matrix_train_200D,
                word_embd_matrix_train300 = embd_matrix_train_300D, cnn_dropout = cnn_dropout, kernel_size = kernel_size, filters = filters,
                hidden_layers_utterance = hidden_layers_utterance, utterance_dropout_rate = utterance_dropout_rate, utterance_recurrent_dropout = utterance_recurrent_dropout,
                hidden_layers_dialogue = hidden_layers_dialogue, dialogue_dropout_rate = dialogue_dropout_rate, dialogue_recurrent_dropout = dialogue_recurrent_dropout,
                optimizer = optimizer).create_model()

    
    # Use Keras to train the model.
    callback = keras.callbacks.EarlyStopping(monitor='acc', patience=epochs)

    steps_per_epoch = len(train_x)
    validation_steps = len(valid_x)

    
    history =  model.fit_generator(generator = data_generators.DataGenerators(embedding = 'word_char', data_y = train_y,  data_x = train_x, 
                                                                             data_char_x = train_char_x, data_x_segments=None, 
                                                                             data_x_tokens=None).data_generator(),
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    verbose=1,
                                    callbacks=[callback],
                                    validation_data= data_generators.DataGenerators(embedding = 'word_char', data_y = valid_y,  data_x = valid_x, 
                                                                                    data_char_x = valid_char_x, data_x_segments=None, 
                                                                                    data_x_tokens=None).data_generator(),
                                    validation_steps= validation_steps,
                                    class_weight=None,
                                    max_queue_size=10,
                                    workers=1,
                                    use_multiprocessing=False,
                                    shuffle=True)
    
    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    global best_accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    
    return -accuracy



print('==== start tuning ====')
start_time = datetime.now()
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=r,
                            x0=default_parameters)

important_data = [search_result.x, search_result.x_iters, search_result.func_vals]
pickle.dump(important_data, open(results_save_dir + model_name, 'wb'))


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
