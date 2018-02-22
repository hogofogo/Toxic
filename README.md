# Toxic comments

Data is from Kaggle and is a current competition:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

It represents a corpus of wikipedia comments some of which are labeled as toxic, severe toxic, obscene, etc. basically representing various degrees of online harassment. What made this problem interesting is that these comments are not mutually exclusive, i.e. it's a multi class classification. Also, this creates a great opportunity for testing pre-trained embeddings.

I approached this problem by transferring embeddings from GloVe. I built a two-layer LSTM model to capture context. Accuracy on validation set resulted in 98.17 percent on the validation set of 20 percent of the training data set after just 3 epochs -- not bad -- , although ~90 percent of the training set is represented by comments not labeled as either one of the 'toxic' labels. 

Next steps: data examination revealed that a lot of toxic comments are caps and have a lot of exclamation marks, in addition to obscenities; I made changes to tokenizer to suppress lower case conversion and removed exclamation mark from the filters. Kept the same max number of tokenized words (should possibly be increased). After 3 epochs the results are: 97.65 on the validation set - worse.

Next try increasing max words to keep to 75K (triple) to offset thinning due to allowing caps, run in on 1 epoch. Same result on validation 97.61 percent as with 25K. Should have worked better, but I don't think 75K is enough and it has already slowed down the computation a lot. This makes sense, give there are 400K+ unique tokens even with lower case = True.

Go back to 25K, caps to lower, but keep exclamation marks. 97.71 percent = better. Train two more epochs to compare to the first configuration, see above. Validation accuracy = 98.06 percent = again not bad, but nothing special and worse than the earlier model. 

Given average train sequence length is 65, it might make sense to try attention model next. Additional simple strategy to try includes limiting the size of the non-offensive set to make the non-offensive/offensive more balanced - TBD.

In the mean time, I ran a model prediction on a sample and there's work to be done. First, the model output is probabilities, not binary values. This is not a problem. The don't add to 1 per sample, i.e. looks like separate lable classification works (i.e. not like softmax). Setting a threshold for conversion of values into 1 is not a problem. The problem is that based on my sample of 8 entries, about 23 percent of labels would be misclassified - I picked the optimal threshold for this particular set that may or may not work for the population in general. 

A question that needs answer: what exactly did the model mean when it measured accuracy? Ground truth labels are binary; predictions are probability, i.e. they are incompatible. Keras doesn't appear to have a function to convert probability predictions into binary. At any rate, the desired threshold would be much lower than 0.01.

A possible solution: run separate predictive models for each of the classes separately, i.e. last dense layer will be dim = 1 with a sigmoid, assuming binary classification behaves properly for one-dimensional output. Tried, same result and the function giving binary categorical value has apparently been deprecated.

Next I undersampled non-toxic comments to make labels more balanced; this proved helpful. I built a dataset split approx 1:2 (all toxic comments and 2/3 random non-toxic comments) and trained it with 94 percent accuracy; mind that the baseline changed from ~90 percent non-toxic to ~65 percent non-toxic). It still remains skewed with respect to individual label types, but much less.

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

As the next step, it may make sense to build a predictor of categorical values on a dev set to see the misclassified outcome in order to understand what may be done to improve input data - TBD I think running regex replacement of multiple characters for uniformity might be productive. Summarizing, there are three directions to be tried: 1. consider splitting models for each label type; 2. complicate the model a bit, i.e. add attention; 3. play with the input data once it's clear what kind of items have been misclassified.
