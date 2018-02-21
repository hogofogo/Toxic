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

A possible solution: run separate predictive models for each of the classes separately, i.e. last dense layer will be dim = 1 with a sigmoid, assuming binary classification behaves properly for one-dimensional output. 