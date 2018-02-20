# Toxic comments

Data is from Kaggle and is a current competition:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

It represents a corpus of wikipedia comments some of which are labeled as toxic, severe toxic, obscene, etc. basically representing various degrees of online harassment. What made this problem interesting is that these comments are not mutually exclusive, i.e. it's a multi class classification. Also, this creates a great opportunity for testing pre-trained embeddings.

Accuracy on validation set resulted in 98.17 percent on the validation set of 20 percent of the training data set after 3 epochs, although ~90 percent of the training set is represented by comments not labeled as either one of the 'toxic' labels. 

I approached this problem by transferring embeddings from GloVe. I built a two-layer LSTM model. The initial result looks good. Next steps: TBD

