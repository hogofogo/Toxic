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



X_train = train_data['comment_text'].values
X_test = test_data['comment_text'].values
y_train = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
                   
max_features = 25000  # number of words to keep
maxlen = 100  # max length of the comments in the model
batch_size = 64  # batch size for the model
embedding_dims = 100  # dimension of the hidden variable, i.e. the embedding dimension



#tokenizer = Tokenizer(num_words=max_features)
tokenizer = Tokenizer(num_words=max_features, filters='"!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
x_train = tokenizer.texts_to_sequences(X_train)
#x_train = np.array(x_train).reshape((np.array(x_train.shape[0])), 1)
print(len(x_train), 'train sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = sequence.pad_sequences(x_train, maxlen=maxlen)
labels = y_train
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
X = Dropout(0.5)(X)
X = LSTM(128, return_sequences = False)(X)
X = Dropout(0.5)(X)
X = Dense(6)(X)
X = Activation('sigmoid')(X)


model = Model(sequence_input, X)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


 
model.fit(data, labels, batch_size=batch_size, epochs=1, validation_split=0.2)

       
