# Toxic 

## Overview

Data is from Kaggle and is a current competition:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

It represents a corpus of wikipedia comments some of which are labeled as toxic, severe toxic, obscene, etc. basically representing various degrees of online harassment. What made this problem interesting is that these comments are not mutually exclusive, i.e. it's a multi class classification, plus this creates a great opportunity for testing pre-trained embeddings.

# Architecture

I approached this problem by transferring embeddings from GloVe. I built a two-layer LSTM model to capture context. Accuracy on validation set resulted in 98.17 percent on the validation set of 20 percent of the training data set after just 3 epochs -- not bad -- , although ~90 percent of the training set is represented by comments not labeled as either one of the 'toxic' labels. 

Given average train sequence length is 65, it might make sense to try attention model next - TBD.

I downsampled non-labeled data (see data clearning below) and got accuracy of ~94 percent + in various training sets which is better than that on the full set.

# Data cleaning

An obvious problem with the dataset is its skewed nature: ~90 percent of entries are non-toxic. I undersampled non-toxic comments to make labels more balanced; this proved helpful. I built a dataset split approx 1:2 (all toxic comments and 2/3 random non-toxic comments) and trained it with 94 percent accuracy; mind that the baseline changed from ~90 percent non-toxic to ~65 percent non-toxic). It still remains skewed with respect to individual label types, but much less.

Data examination revealed that a lot of toxic comments are caps and have a lot of exclamation marks, in addition to obscenities; I experimented with tokenizer to suppress lower case conversion and removing exclamation mark from the filters. 

## Training

In the mean time, I ran a model prediction on a sample and there's work to be done. First, the model output is probabilities, not binary values and it appears that Keras deprecated the conversion of probability outputs to binary values - not a problem, but will require a careful choice of threshold, and possibly separate for each label type if I decide to run separate trainings for each label type, like the below.

I ran training separately for each label type on 1 epoch with this results:
0: 0.8952 
1: 0.9665
2: 0.9320
3: 0.9897
4: 0.9015
5: 0.9718
mean = 0.9428

If I run all 6 together, after 1 epoch I get 0.9348. I wonder whether this really has to do with separate trainings for each label type or just a random variation, but it may be safer to run each label separately, and also come up with individual cut-off values for each label type. After I added back the '!' I get 0.9364 after 1 epoch, and the words are from the shrunk train set only. Again, small and possibly random difference, but let's keep '!' going forward. Train two more epochs for a total 3 and get 0.9435.

Removed \' from dictionary after 1 epoch get 0.9361 = good. On two more epochs for a total 3 get 0.9424  => put it back in.

## Results

As the next step, it may make sense to build a predictor of categorical values on a dev set to see the misclassified outcome in order to understand what may be done to improve input data - TBD I think running regex replacement of multiple characters for uniformity might be productive. Summarizing, there are three directions to be tried: 1. consider splitting models for each label type; 2. complicate the model a bit, i.e. add attention; 3. play with the input data once it's clear what kind of items have been misclassified.
