# Toxic 

## Overview

Data is from Kaggle and is a recent competition:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

It represents a corpus of wikipedia comments some of which are labeled as toxic, severe toxic, obscene, etc. basically representing various degrees of online harassment. What made this problem interesting is that these comments are not mutually exclusive, i.e. it's a multi class classification, plus this creates a great opportunity for testing pre-trained embeddings.

## Architecture

I built several models which complemented each other in ensemble and fall into three categories:

- LSTM or GRU based on pre-trained word embeddings (I used GloVe and Word2Vec in different version of implementation). 
- Word-based 1D convolutional neural network trained on word embeddings
- Character-based 1D convolutional neural network

After I built my initial LSTM model and examined the errors, it became clear that the model was misclassifying as 'offensive' the offensive words in an inoffensive context. Also, the model was not properly picking accidentally or deliberately misspelled words. The word- and character-based convolutional neural networks helped solve this problem. The models were stacked by fitting a linear regression on meta-features.

## Data cleaning

I experimented with several approaches:

- Undersampling: the dataset has a skewed nature: ~90 percent of entries were non-toxic. I undersampled non-toxic comments to make labels more balanced; this proved helpful. I built a dataset split approx 1:2 (all toxic comments and 2/3 random non-toxic comments) and trained it with 94 percent accuracy; mind that the baseline changed from ~90 percent non-toxic to ~65 percent non-toxic). It still remains skewed with respect to individual label types, but much less.

- for character-based models, data preprocessing included conversion into ascii from utf8 to get rid of rare characters and make the dataset more manageable. 

- lower case conversion: I ended up leaving the cases untouched for character-based models, but converted inputs to lower-case for conversion into embeddings
etc.

## Training

I generally ran the models on some 3-8 epochs before experiencing overfitting, typically with 10-20 percent validation and tuning dropout regularization for optimum performance. 

In addition, I experimented with two different kinds of setup: the task required a binary prediction of several non-mutually exclusive categorical values. I ran the models to predict all the categories at ones, as well as several models dedicated to predicting individual categories. The separate models gave a marginally better performance.

## Results

I selected five best-performing models, including LSTM built on GloVE, LSTM built on Word2Vec, 1D convolutional model built on GloVe and two chacater-based 1D convolutional models built with different kernel sizes. The assembly resulted in 0.9840 AUC score.



