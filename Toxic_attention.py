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
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, LSTM, Dropout, Activation, GRU
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import os
import keras.backend as K
from keras.layers import Bidirectional, RepeatVector, Concatenate, Permute, Dot



train_data = pd.read_csv('~/Projects/Toxic/train.csv').fillna('VK')
test_data = pd.read_csv('~/Projects/Toxic/test.csv').fillna('VK')



#BUILD A MORE BALANCED DATA SET
train_data['overall'] = 0
train_data['overall'] = train_data.drop(['id', 'comment_text'], axis = 1).max(axis = 1)

x_ones = train_data.ix[train_data['overall'] == 1]
x_zeros = train_data.ix[train_data['overall'] == 0]
#shuffle data
x_zeros = x_zeros.sample(n = 30000, replace = False, axis = 0)
data_df = pd.concat([x_ones, x_zeros])
data_df = data_df.sample(frac=1).reset_index(drop=True)




#CREATE TRAIN AND TEST SETS
X_train = data_df['comment_text'].values
X_test = test_data['comment_text'].values
y_train = data_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
                   
max_features = 25000  # number of words to keep
maxlen = 100  # max length of the comments in the model
batch_size = 64  # batch size for the model
embedding_dims = 100  # dimension of the hidden variable, i.e. the embedding dimension
n_a = 32 #hidden state size of the Bi-LSTM
n_s = 64 #hidden state size of the post-attention LSTM



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
labels = y_train
#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
s_input = np.zeros((len(x_train), n_s))
c_input = np.zeros((len(x_train), n_s))



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


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')



def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    s_prev = RepeatVector(maxlen)(s_prev)
    concat = Concatenate(axis=-1)([a, s_prev])
    e = Dense(10, activation = "tanh")(concat)
    energies = Dense(1, activation = "relu")(e)
    alphas = Activation(softmax)(energies)
    context = Dot(axes = 1)([alphas, a])
    
    return context

post_activation_LSTM_cell = LSTM(n_s, return_state = True)

sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

s0 = Input(shape=(n_s,), name='s0')
c0 = Input(shape=(n_s,), name='c0')
s = s0
c = c0


a = Bidirectional(LSTM(n_a, return_sequences=True))(embedded_sequences)
a = Dropout(0.3)(a)

   
context = one_step_attention(a, s)        
s, _, c = post_activation_LSTM_cell(context, initial_state = [s, c])        
s = Dropout(0.3)(s)
outputs = Dense(2, activation='sigmoid')(s)        


model = Model(inputs = [sequence_input, s0, c0], outputs = outputs)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])



#model.fit(x = data, y = labels, batch_size=batch_size, epochs=1, validation_split=0.2)
model.fit(x = [data, s_input, c_input], y = labels, batch_size=batch_size, epochs=1, validation_split=0.2)



#TEST UTILITY
sample = train_data.iloc[[6,12,16,42,43,51,55,65],]
x_sample = tokenizer.texts_to_sequences(sample)
sample_data = sequence.pad_sequences(x_sample, maxlen=maxlen)
sample_result = model.predict(sample_data)

sample_result = []
#TEST INDIVIDUAL LABEL TYPE PERFORMANCE
def evaluate_model():
    for i in range(6):
        labels = y_train[:,i]
        labels = to_categorical(labels,num_classes = None)
        model.fit(x = [data, s_input, c_input], y = labels, batch_size=batch_size, epochs=3, validation_split=0.2)
     