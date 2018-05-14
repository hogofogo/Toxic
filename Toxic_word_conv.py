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
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, Dense, MaxPooling1D, Embedding, Flatten, Dropout
from keras.utils import to_categorical
from numpy import array
import os

train_data = pd.read_csv('/home/vlad/Documents/Toxic/train.csv').fillna('VK')
test_data = pd.read_csv('/home/vlad/Documents/Toxic/test.csv').fillna('VK')


#def remove_non_ascii(text):   
#    return ''.join([i if ord(i) < 128 else ' ' for i in text])

comments_train = list(train_data['comment_text'])
comments_test = list(test_data['comment_text'])


#set the max length of the sequence based on histogram output
maxlen = 500
embedding_dims = 100  # dimension of the hidden variable, i.e. the embedding dimension


max_features = 100000
#BUILD MODEL
#tokenizer = Tokenizer(num_words=max_features)
tokenizer = Tokenizer(num_words=max_features, char_level = False)
#tokenizer.fit_on_texts(list(X_train) + list(X_test))
tokenizer.fit_on_texts(comments_train)
x_train = tokenizer.texts_to_sequences(comments_train)
word_index = tokenizer.word_index

#prepare the embeddings layer, compute an index mapping words 
#to known embeddings, by parsing the data dump of pre-trained embeddings
#prepare the embeddings layer, compute an index mapping words 
#to known embeddings, by parsing the data dump of pre-trained embeddings
embeddings_index = {}
GLOVE_DIR = '/home/vlad/Documents/Toxic/glove'
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



data = sequence.pad_sequences(x_train, maxlen=maxlen)

y_train = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
labels = to_categorical(y_train,num_classes = None)

batch_size = 64  # batch size for the model




sequence_input = Input(shape=(maxlen,), dtype='int32')

embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dims,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False)(sequence_input)

X = Conv1D(128, 5, padding='same', activation='relu')(embedding_layer)
X = MaxPooling1D(5)(X)
X = Conv1D(128, 5, padding='same', activation='relu')(X)
X = MaxPooling1D(5)(X)
X = Conv1D(128, 5, padding='same', activation='relu')(X)
X = MaxPooling1D(3)(X)  # global max pooling
X = Dropout(0.4)(X)
#X = Flatten()(X)
X = Dense(128, activation='relu')(X)
preds = Dense(2, activation='sigmoid')(X)



model = Model(sequence_input, preds)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


model.fit(x = data, y = labels, batch_size=batch_size, epochs=1, validation_split=0.2)


