# Toxic 

## Overview

Data is from Kaggle and is a current competition:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

It represents a corpus of wikipedia comments some of which are labeled as toxic, severe toxic, obscene, etc. basically representing various degrees of online harassment. What made this problem interesting is that these comments are not mutually exclusive, i.e. it's a multi class classification, plus this creates a great opportunity for testing pre-trained embeddings.

## Architecture

I approached this problem by transferring embeddings from GloVe. I built a two-layer LSTM model to capture context. Accuracy on validation set resulted in 98.17 percent on the validation set of 20 percent of the training data set after just 3 epochs -- not bad -- , although ~90 percent of the training set is represented by comments not labeled as either one of the 'toxic' labels. 

Given average train sequence length is 65, it might make sense to try attention model next - TBD.

I downsampled non-labeled data (see data clearning below) and got accuracy of ~94 percent + in various training sets which is better than that on the full set.

## Data cleaning

An obvious problem with the dataset is its skewed nature: ~90 percent of entries are non-toxic. I undersampled non-toxic comments to make labels more balanced; this proved helpful. I built a dataset split approx 1:2 (all toxic comments and 2/3 random non-toxic comments) and trained it with 94 percent accuracy; mind that the baseline changed from ~90 percent non-toxic to ~65 percent non-toxic). It still remains skewed with respect to individual label types, but much less.

Data examination revealed that a lot of toxic comments are caps and have a lot of exclamation marks, in addition to obscenities; I experimented with tokenizer to suppress lower case conversion and removing exclamation mark from the filters. 

## Training

In the mean time, I ran a model prediction on a sample and there's work to be done. First, the model output is probabilities, not binary values and it appears that Keras deprecated the conversion of probability outputs to binary values - not a problem, but will require a careful choice of threshold, and possibly separate for each label type if I decide to run separate trainings for each label type, like the below.

[Feb25 separate run for each categorical value on 3 epochs, 20 percent validation and 30 percent drop; essentially the model gets retrained consecutively for each new label, and 3 epochs was enough for the loss function to stop decreasing]:
0: 0.9002 	
1: 0.9644
2: 0.9346
3: 0.9898
4: 0.9090
5: 0.9734
mean = 0.9452

[Feb 24: I have also subsequently built an attention model with the following results after 1 epoch:
0: 0.8932
1: 0.9672
2: 0.9316
3: 0.9902
4: 0.9051
5: 0.9764
mean = 0.9440
Slight improvement, but it appears that model changes have largely exhausted the potential. I get marginal changes with different model, bigger model, different building blocks, etc. I see no significant improvement of the training set performance. Going forward, I will focus mainly on the inputs and error correction. that said, looks promising; i will try 3 epochs and perhaps drop]

If I run all 6 together, after 1 epoch I get 0.9348. I wonder whether this really has to do with separate trainings for each label type or just a random variation, but it may be safer to run each label separately. After I added back the '!' I get 0.9364 after 1 epoch, and the words are from the shrunk train set only. Again, small and possibly random difference, but let's keep '!' going forward. Train two more epochs for a total 3 and get 0.9435.

[Feb 25: on all 6 together with attention model got 0.9275 after 1 epoch, and 0.9350  after 3 epochs]

Removed \' from dictionary after 1 epoch get 0.9361 = good. On two more epochs for a total 3 get 0.9424  => put it back in.

Also, I have created additional features (percentage of caps in comment and percentage of exclamation marks in comment) and merged output of two layer LSTM model with additional features input for several different configurations with dense layers but performance was worse in each case than for the simple 2-layer LSTM.

I have tried replacing LSTM with GRU and got 0.9377 accuracy after 1 epoch, and 0.9429 after 3 =~ the same.

## Results

As the next step, it may make sense to build a predictor of categorical values on a dev set to see the misclassified outcome in order to understand what may be done to improve input data - TBD I think running regex replacement of multiple characters for uniformity might be productive. Summarizing, there are three directions to be tried: 1. consider splitting models for each label type; 2. complicate the model a bit, i.e. add attention [done but saw no significant improvement; one last experiment with the size of the attention model TBD]; 3. play with the input data once it's clear what kind of items have been misclassified.



