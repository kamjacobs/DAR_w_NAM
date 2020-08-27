
"""
This class returns the data generators for each embedding type (word, character and BERT)

Dialogues that contain the same number of utterances are grouped together into mini-batches
"""

import numpy as np

class DataGenerators():

    def __init__(self, embedding,data_y,  data_x = None, data_char_x = None, data_x_segments = None, data_x_tokens = None):
        self.embedding = embedding
        self.data_x = data_x
        self.data_y = data_y
        self.data_char_x = data_char_x
        self.data_x_segments = data_x_segments
        self.data_x_tokens = data_x_tokens


    def unique(self, list1):
        """
        Function returns list with unique values
        in a list
        """ 
      
        # intilize a null list 
        unique_list = [] 
          
        # traverse for all elements 
        for x in list1: 
            # check if exists in unique_list or not 
            if x not in unique_list: 
                unique_list.append(x) 
        
        return unique_list
 
    def train_metadata(self):
        """
        Function returns metadata necessary to
        create batches of dialogues with equal number
        of utterances
        """
        utterance_len = []
        for i in range(0,len(self.data_y)):
            utterance_len.append(self.data_y[i].shape[0])

        number_of_utterances_dialogue = (self.unique(utterance_len))  
        batch_size = len(number_of_utterances_dialogue) 
        
        return number_of_utterances_dialogue, batch_size


    def data_generator_character(self):
        """
        Function returns batches in case both word-embeddings and
        character-embeddings are applied
        """
    
        number_of_utterances_dialogue = self.train_metadata()[0]
        batch_size = self.train_metadata()[1]
        
        tmp = {}
        for i in range(0,batch_size):
            tmp['x_batch_utt_size_' + str(number_of_utterances_dialogue[i])] = []
            tmp['x_char_batch_utt_size_' + str(number_of_utterances_dialogue[i])] = []
            tmp['y_batch_utt_size_' + str(number_of_utterances_dialogue[i])] = []
            for j in range(0,len(self.data_x)):
                if self.data_x[j].shape[0] == number_of_utterances_dialogue[i]:
                    tmp['x_batch_utt_size_' + str(number_of_utterances_dialogue[i])].append(self.data_x[j])
                    tmp['x_char_batch_utt_size_' + str(number_of_utterances_dialogue[i])].append(self.data_char_x[j])
                    tmp['y_batch_utt_size_' + str(number_of_utterances_dialogue[i])].append(self.data_y[j])

        while 1:
            for i in range(0,batch_size):
                batch_data_x = np.array(tmp['x_batch_utt_size_' + str(number_of_utterances_dialogue[i])])
                batch_data_x = np.reshape(batch_data_x, (batch_data_x.shape[0], batch_data_x.shape[1], batch_data_x.shape[2], 1))
                batch_char_x = np.array(tmp['x_char_batch_utt_size_' + str(number_of_utterances_dialogue[i])])
                batch_y = np.array(tmp['y_batch_utt_size_' + str(number_of_utterances_dialogue[i])])
                yield [batch_data_x, batch_char_x], batch_y

    def data_generator_bert(self):
        """
        Function returns batches in case of BERT embeddings
        """
        number_of_utterances_dialogue = self.train_metadata()[0]
        batch_size = self.train_metadata()[1]

        tmp = {}
        for i in range(0,batch_size):
            tmp['x_segments_batch_utt_size_' + str(number_of_utterances_dialogue[i])] = []
            tmp['x_tokens_batch_utt_size_' + str(number_of_utterances_dialogue[i])] = []
            tmp['y_batch_utt_size_' + str(number_of_utterances_dialogue[i])] = []
            for j in range(0,len(self.data_x_tokens)):
                if self.data_x_tokens[j].shape[0] == number_of_utterances_dialogue[i]:
                    tmp['x_segments_batch_utt_size_' + str(number_of_utterances_dialogue[i])].append(self.data_x_segments[j])
                    tmp['x_tokens_batch_utt_size_' + str(number_of_utterances_dialogue[i])].append(self.data_x_tokens[j])
                    tmp['y_batch_utt_size_' + str(number_of_utterances_dialogue[i])].append(self.data_y[j])

        while 1:
            for i in range(0,batch_size):
                batch_x_segments = np.array(tmp['x_segments_batch_utt_size_' + str(number_of_utterances_dialogue[i])])
                batch_x_tokens = np.array(tmp['x_tokens_batch_utt_size_' + str(number_of_utterances_dialogue[i])])
                batch_y = np.array(tmp['y_batch_utt_size_' + str(number_of_utterances_dialogue[i])])
                yield [batch_x_segments, batch_x_tokens], batch_y


    def data_generator(self):
        if self.embedding == 'word_char':
            data_generator = self.data_generator_character() 
        elif self.embedding == 'BERT':
            data_generator = self.data_generator_bert()

        return data_generator 

