#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 07:47:56 2018

@author: vlad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Model, Input
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, LSTM, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import os
from keras.utils import to_categorical



train_data = pd.read_csv('~/Projects/Toxic/train.csv').fillna('VK')
test_data = pd.read_csv('~/Projects/Toxic/test.csv').fillna('VK')
train_data['overall'] = 0
train_data['overall'] = train.sum(axis = 1)

train_data['overall'].value_counts()
'''
0    143346
1      6360
3      4209
2      3480
4      1760
5       385
6        31
out of 159571 rows
'''


#BUILD A MORE BALANCED DATA SET
train_data['overall'] = 0
train_data['overall'] = train_data.drop(['id', 'comment_text'], axis = 1).max(axis = 1)

x_ones = train_data.ix[train_data['overall'] == 1]
x_zeros = train_data.ix[train_data['overall'] == 0]
#shuffle data
x_zeros = x_zeros.sample(n = 90000, replace = False, axis = 0)
data_df = pd.concat([x_ones, x_zeros, x_ones, x_ones])
data_df = data_df.sample(frac=1).reset_index(drop=True)




#DATA CLEANING AND FEATURE ENGINEERING UTILITIES

def feature_enhancement(input_column = train_data['comment_text']):
#The function scross through the list of comments and builds features for the number
#and ratio of exclamation marks and caps in the comment
#returns:
#   number of caps  
#   ratio of caps
#   number of excl marks
#   ratio of excl marks
    
    exclmarks_ = []
    exclmarks_ratio_ =[]
    caps_ = [] 
    caps_ratio_ = []
    

    for comment in range(len(input_column)):
        exam_string = input_column[comment]
        exam_chars = list(exam_string)
        exam_chars = [w for w in exam_chars if w.istitle()]
        exclmarks = exam_string.count('!')
        exclmarks_ratio = exclmarks/float(len(exam_string))
        caps = len(exam_chars)
        caps_ratio = caps/float(len(exam_string))
        
        exclmarks_.append(exclmarks)
        exclmarks_ratio_.append(exclmarks_ratio)
        caps_.append(caps) 
        caps_ratio_.append(caps_ratio)

    return exclmarks_, exclmarks_ratio_, caps_, caps_ratio_


exclmarks, exclmarks_ratio, caps, caps_ratio = feature_enhancement(input_column = train_data['comment_text'])


#CREATE TRAIN AND TEST SETS
X_train = data_df['comment_text'].values
#X_test = test_data['comment_text'].values
y_train = data_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
                   
max_features = 250000  # number of words to keep
maxlen = 100  # max length of the comments in the model
batch_size = 64  # batch size for the model
embedding_dims = 100  # dimension of the hidden variable, i.e. the embedding dimension


#BUILD MODEL
#tokenizer = Tokenizer(num_words=max_features)
tokenizer = Tokenizer(num_words=max_features, filters='"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True)
#tokenizer.fit_on_texts(list(X_train) + list(X_test))
tokenizer.fit_on_texts(list(X_train))
x_train = tokenizer.texts_to_sequences(X_train)
#x_train = np.array(x_train).reshape((np.array(x_train.shape[0])), 1)
print(len(x_train), 'train sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = sequence.pad_sequences(x_train, maxlen=maxlen)
#labels = y_train[:,1]
labels = to_categorical(y_train,num_classes = None)
#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)



#prepare the embeddings layer, compute an index mapping words 
#to known embeddings, by parsing the data dump of pre-trained embeddings
embeddings_index = {}
GLOVE_DIR = '/Users/vlad/Projects/toxic/glove'
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


#compute embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dims))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# load this embedding matrix into an Embedding layer. trainable=False
embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dims,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False)


sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

X = LSTM(128, return_sequences=True)(embedded_sequences)
X = Dropout(0.4)(X)
X = LSTM(128, return_sequences = False)(X)
X = Dropout(0.4)(X)
X = Dense(2, activation = 'sigmoid')(X)
#X = Activation('sigmoid')(X)


model = Model(sequence_input, X)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

 
model.fit(x = data, y = labels, batch_size=batch_size, epochs=1, validation_split=0.2)





#TEST UTILITY
sample = train_data.iloc[[6,12,16,42,43,51,55,65],]
x_sample = tokenizer.texts_to_sequences(sample)
sample_data = sequence.pad_sequences(x_sample, maxlen=maxlen)
sample_result = model.predict(sample_data)

sample_result = []
#TEST INDIVIDUAL LABEL TYPE PERFORMANCE
def evaluate_model(epochs = 1):
    for i in range(6):
        labels = y_train[:,i]
        labels = to_categorical(labels,num_classes = None)
        model.fit(x = data, y = labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        model.save('/Users/vlad/Projects/Toxic/model_' + str(i) +'.h5')
        


def get_predictions():

    pred_X = train_data['comment_text']
    pred_X = tokenizer.texts_to_sequences(pred_X)
    pred_X = sequence.pad_sequences(pred_X, maxlen=maxlen)
    
    threshold = 0.5
    predictions = []

    for i in range(6):
        model.load_weights('/Users/vlad/Projects/Toxic/model_' + str(i) +'.h5', by_name=True)
        prediction = model.predict(pred_X)
        prediction = (prediction[:,0] < threshold).astype(np.int)
        predictions.append(prediction)
        print(i)
        
    return predictions


predictions = get_predictions()


def evaluate_predictions(column = 0):
    comment = []
    groundtruth = []
    predict = []
    
    groundtruth_labels = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
    predicted_labels = np.concatenate((predictions[0].reshape(len(predictions[0]),1), predictions[1].reshape(len(predictions[1]),1), predictions[2].reshape(len(predictions[2]),1), predictions[3].reshape(len(predictions[3]),1), predictions[4].reshape(len(predictions[4]),1), predictions[5].reshape(len(predictions[5]),1)), axis = 1)
    for i in range(len(predicted_labels)):
        if np.array_equal(groundtruth_labels[i,column], predicted_labels[i,column]) != True:
            comment.append(train_data.loc[i, 'comment_text'])
            groundtruth.append(str(groundtruth_labels[i,column]))
            predict.append(str(predicted_labels[i,column]))
            
    return comment, groundtruth, predict
    

result = pd.DataFrame({'comment': comment, 'groundtruth': groundtruth, 'predict': predict})    


  
