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

train_data = pd.read_csv('/home/vlad/Documents/Toxic/train.csv').fillna('VK')
test_data = pd.read_csv('/home/vlad/Documents/Toxic/test.csv').fillna('VK')


def remove_non_ascii(text):   
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

comments_train = list(train_data['comment_text'])
comments_test = list(test_data['comment_text'])

#this is a character model; utf8 results in a massive number of rare characters;
#get rid of them by converting inputs into ascii
def build_ascii_list(comments_input):
    comments =[]
    for item in comments_input:
        comments.append(remove_non_ascii(item))
    return comments
    
comments_train = build_ascii_list(comments_train)
comments_test = build_ascii_list(comments_test)

#map all unique characters to numbers
raw_text = ''
for item in comments_train:
    raw_text = raw_text + ' '.join(item.split())
for item in comments_test:
    raw_text = raw_text + ' '.join(item.split())


chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))



#evaluate sequence length distribution of the data set
comments = train_data['comment_text'].tolist()
max_len = 0
len_data = []
for item in comments:
    length = len(item)
    if length > max_len:
        max_len = length
    len_data.append(length)

len_data = pd.DataFrame(len_data)
len_data.hist()   

#set the max length of the sequence based on histogram output
maxlen = 400

#get the list of sequences of the desired length
def cut_and_pad(comments):
    sequences = []
    for item in comments:
    # select sequence of tokens of given length and pad with whitespace where necessary
        seq = item[:length]
        seq = seq.ljust(length)
        sequences.append(seq)    
    return sequences   

sequences = cut_and_pad(comments_train)

max_features = len(int_to_char)
#BUILD MODEL
#tokenizer = Tokenizer(num_words=max_features)
tokenizer = Tokenizer(num_words=max_features, char_level = True)
#tokenizer.fit_on_texts(list(X_train) + list(X_test))
tokenizer.fit_on_texts(sequences)
x_train = tokenizer.texts_to_sequences(sequences)

data = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')

y_train = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
labels = to_categorical(y_train,num_classes = None)

batch_size = 64  # batch size for the model
embedding_dims = len(int_to_char)  # dimension of the hidden variable, i.e. the embedding dimension



sequence_input = Input(shape=(maxlen,), dtype='int32')

embedding_layer = Embedding(97,
                            embedding_dims,
                            input_length=maxlen,
                            trainable=False)(sequence_input)

X = Conv1D(128, 5, padding='same', activation='relu')(embedding_layer)
X = MaxPooling1D(5)(X)
X = Conv1D(128, 5, padding='same', activation='relu')(X)
X = MaxPooling1D(4)(X)
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


model.fit(x = data, y = labels, batch_size=batch_size, epochs=5, validation_split=0.2)


