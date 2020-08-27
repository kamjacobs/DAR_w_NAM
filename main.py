from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use Tensorflow version 1.0 or newer. You are using {}'.format(tf.__version__)
print('Tensorflow Version: {}'.format(tf.__version__))

# Check for GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please ensure you have installed TensorFlow correctly')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

import pickle
import os 
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


print('======CREATE MODEL=====')
if model_name == 'raheja_tetreault':
	model = RahejaModel(metadata, word_embd = 'train300', word_embd_matrix_train300 = word_embd_matrix_train300, 
				cnn_dropout = 0.2792, kernel_size = 5, filters = 100,
	            hidden_layers_utterance = 210, utterance_dropout_rate = 0.6, utterance_recurrent_dropout = 0.2431,
	            hidden_layers_dialogue = 215, dialogue_dropout_rate = 0.0, dialogue_recurrent_dropout = 0.1103,
	            optimizer = 'RMSprop', optimizer_algorithm = None).create_model()
elif model_name == 'pool-crf':
	model = PoolCRF(metadata, word_embd = 'train300', word_embd_matrix_train300 = word_embd_matrix_train300, 
				cnn_dropout = 0.1577, kernel_size = 4, filters = 83,
	            hidden_layers_utterance = 292, utterance_dropout_rate = 0.2681, utterance_recurrent_dropout = 0.0148,
	            hidden_layers_dialogue = 293, dialogue_dropout_rate = 0.0615, dialogue_recurrent_dropout = 0.1733,
	            optimizer = 'RMSprop', optimizer_algorithm = None).create_model()
elif model_name == 'att-crf':
	model = AttCRF(metadata, word_embd = 'google', word_embd_matrix_google = embd_matrix_google, 
				cnn_dropout = 0.6, kernel_size = 5, filters = 15,
	            hidden_layers_utterance = 300, utterance_dropout_rate = 0.0, utterance_recurrent_dropout = 0.6,
	            hidden_layers_dialogue = 300, dialogue_dropout_rate = 0.0, dialogue_recurrent_dropout = 0.0,
	            optimizer = 'RMSprop', attention_dim = 200, optimizer_algorithm = None).create_model()
elif model_name == 'mha-pool-crf':
	model = MHAPoolCRF(metadata, word_embd = 'train300', word_embd_matrix_train300 = word_embd_matrix_train300, 
				cnn_dropout = 0.6, kernel_size = 4, filters = 77,
	            hidden_layers_utterance = 236, utterance_dropout_rate = 0.6, utterance_recurrent_dropout = 0.0217,
	            hidden_layers_dialogue = 212, dialogue_dropout_rate = 0.0392, dialogue_recurrent_dropout = 0.1897,
	            optimizer = 'adam', heads = 6, optimizer_algorithm = None).create_model()
elif model_name == 'mha-att-crf':
	model = MHAAttCRF(metadata, word_embd = 'train300', word_embd_matrix_train300 = word_embd_matrix_train300, 
				cnn_dropout = 0.1, kernel_size = 7, filters = 15,
	            hidden_layers_utterance = 300, utterance_dropout_rate = 0.1, utterance_recurrent_dropout = 0.0,
	            hidden_layers_dialogue = 287, dialogue_dropout_rate = 0.0, dialogue_recurrent_dropout = 0.0,
	            optimizer = 'adam', heads = 8, attention_dim = 50, optimizer_algorithm = None).create_model()
elif model_name == 'mha-csa-crf':
	model = MHACsaCRF(metadata, word_embd = 'train300', word_embd_matrix_train300 = word_embd_matrix_train300, 
				cnn_dropout = 0.2863, kernel_size = 5, filters = 30,
	            hidden_layers_utterance = 215, utterance_dropout_rate = 0.0, utterance_recurrent_dropout = 0.1103,
	            hidden_layers_dialogue = 195, dialogue_dropout_rate = 0.0, dialogue_recurrent_dropout = 0.1043,
	            optimizer = 'adam', heads = 8, optimizer_algorithm = None).create_model()
elif model_name == 'pool-2rnn':
	model = Pool2RNN(metadata, word_embd = 'train300', word_embd_matrix_train300 = word_embd_matrix_train300, 
				cnn_dropout = 0.1753, kernel_size = 7, filters = 84,
	            hidden_layers_utterance = 231, utterance_dropout_rate = 0.2173, utterance_recurrent_dropout = 0.0475,
	            hidden_layers_dialogue = 100, dialogue_dropout_rate = 0.0186, dialogue_recurrent_dropout = 0.5242,
	            hidden_layers_labels = 57, labels_dropout_rate = 0.5427, labels_recurrent_dropout = 0.3804,
	            optimizer = 'RMSprop', optimizer_algorithm = None).create_model()
elif model_name == 'att-2rnn':
	model = Att2RNN(metadata,  word_embd = 'google', word_embd_matrix_google = embd_matrix_google, 
				cnn_dropout = 0.1209, kernel_size = 6, filters = 99,
	            hidden_layers_utterance = 299, utterance_dropout_rate = 0.2182, utterance_recurrent_dropout = 0.5072,
	            hidden_layers_dialogue = 197, dialogue_dropout_rate = 0.1102, dialogue_recurrent_dropout = 0.5911,
	            hidden_layers_labels = 44, labels_dropout_rate = 0.3480, labels_recurrent_dropout = 0.5715,
	            optimizer = 'RMSprop', attention_dim = 52, optimizer_algorithm = None).create_model()
elif model_name == 'csa_2rnn':
	model = CSA2RNN(metadata, word_embd = 'train300', word_embd_matrix_train300 = word_embd_matrix_train300, 
				cnn_dropout = 0.2810, kernel_size = 6, filters = 30,
	            hidden_layers_utterance = 225, utterance_dropout_rate = 0.1397, utterance_recurrent_dropout = 0.3267,
	            hidden_layers_dialogue = 90, dialogue_dropout_rate = 0.1025, dialogue_recurrent_dropout = 0.4189,
	            hidden_layers_labels = 50, labels_dropout_rate = 0.4289, labels_recurrent_dropout = 0.3216,
	            optimizer = 'RMSprop', optimizer_algorithm = None).create_model()
elif model_name == 'bert-crf':
	# load bert model
	pretrained_embedding_dir = "C://Users//31642//anaconda3//envs//thesis_dar//KimThesis//embeddings//pre-trained//"
	pretrained_path = pretrained_embedding_dir + 'uncased_L-12_H-768_A-12'
	config_path = os.path.join(pretrained_path, 'bert_config.json')
	checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
	vocab_path = os.path.join(pretrained_path, 'vocab.txt')
	max_utterance_len = metadata['max_utterance_len']
	bert_model = load_trained_model_from_checkpoint(
	        config_path,
	        checkpoint_path,
	        training = False,
	        trainable = False,
	        seq_len = max_utterance_len +2
	        )
	model = BertCRF(metadata, bert_model,
	            hidden_layers_dialogue = 293, dialogue_dropout_rate = 0.0615, dialogue_recurrent_dropout = 0.1733,
	            optimizer = 'RMSprop', optimizer_algorithm = None).create_model()
elif model_name == 'bert-2rnn':
	# load bert model
	pretrained_embedding_dir = "C://Users//31642//anaconda3//envs//thesis_dar//KimThesis//embeddings//pre-trained//"
	pretrained_path = pretrained_embedding_dir + 'uncased_L-12_H-768_A-12'
	config_path = os.path.join(pretrained_path, 'bert_config.json')
	checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
	vocab_path = os.path.join(pretrained_path, 'vocab.txt')
	max_utterance_len = metadata['max_utterance_len']
	bert_model = load_trained_model_from_checkpoint(
	        config_path,
	        checkpoint_path,
	        training = False,
	        trainable = False,
	        seq_len = max_utterance_len +2
	        )
	model = Bert2RNN(metadata, bert_model,
	            hidden_layers_dialogue = 293, dialogue_dropout_rate = 0.0615, dialogue_recurrent_dropout = 0.1733,
	            hidden_layers_labels = 50, labels_dropout_rate = 0.4289, labels_recurrent_dropout = 0.3216,
	            optimizer = 'RMSprop', optimizer_algorithm = None).create_model()



epochs = 30
patience = 0
steps_per_epoch = len(train_x)
validation_steps = len(valid_x)


print('====== training model =====')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
history =  model.fit_generator(generator = data_generators.DataGenerators(embedding = 'word_char', data_y = train_y,  data_x = train_x, 
                                                                         data_char_x = train_char_x, data_x_segments=None, 
                                                                         data_x_tokens=None).data_generator(),
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                verbose=1,
                                callbacks=[early_stopping],
                                validation_data= data_generators.DataGenerators(embedding = 'word_char', data_y = valid_y,  data_x = valid_x, 
                                                                                data_char_x = valid_char_x, data_x_segments=None, 
                                                                                data_x_tokens=None).data_generator(),
                                validation_steps= validation_steps,
                                class_weight=None,
                                max_queue_size=10,
                                workers=1,
                                use_multiprocessing=False,
                                shuffle=True)



print('===== test results ======')
loss, accuracy, precision, recall, f1_score = model.evaluate_generator(
                                                generator = data_generators.DataGenerators(embedding = 'word_char', data_y = test_y, data_x = test_x, 
                                                                                         data_char_x = test_char_x, data_x_segments=None, 
                                                                                         data_x_tokens=None).data_generator(),
                                                steps=len(test_y),
                                                verbose=0,
                                                callbacks=None,
                                                use_multiprocessing=False,
                                                max_queue_size=10,
                                                workers=1)


print('loss: ', loss)
print('accuracy: ', accuracy)
print('precision: ', precision)
print('recall: ', recall)
print('f1 score: ', f1_score)

# model.save(model_dir + model_name + '.hdf5', overwrite=True)
model2.save_weights(results_save_dir + 'weights_' + model_name + '.hdf5')
with open(results_save_dir + model_name + '_history.pkl', 'wb') as f:
    pickle.dump(history.history, f) 


import pickle
result_dict = {'loss' : loss, 'accuracy' : accuracy, 'precision' : precision, 'recall': recall, 'f1 score': f1_score }
pickle.dump(result_dict, open(results_save_dir + model_name + '_results.pkl', 'wb'))


def get_predictions(model,metadata, test_x, test_char_x):
    tmp = {}
    for i in range(0,len(test_x)):
        tmp['x_batch_utt_size_' + str(i)] = []
        tmp['x_char_batch_utt_size_' + str(i)] = []

        tmp['x_batch_utt_size_' + str(i)].append(test_x[i])
        tmp['x_char_batch_utt_size_' + str(i)].append(test_char_x[i])

    pred = []
    for i in range(0,len(test_x)):
        batch_data_x = np.array(tmp['x_batch_utt_size_' + str(i)])
        batch_data_x = np.reshape(batch_data_x, (batch_data_x.shape[0], batch_data_x.shape[1], batch_data_x.shape[2], 1))
        batch_char_x = np.array(tmp['x_char_batch_utt_size_' + str(i)])

        new_pred = model.predict([batch_data_x, batch_char_x], batch_size = 1, verbose = 2)
        pred.append(new_pred)

    predictions = np.array(([pred[i][0] for i in range(0,len(pred))]))

    index_to_label = metadata['index_to_label']

    return predictions

print('===== predictions ======')
predictions_mrda = get_predictions(model, metadata, test_x, test_char_x)
np.save('predictions' + model_name,predictions_mrda)