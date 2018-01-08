# Email-Spam-Classifier-Using-Naive-Bayes

* This is a basic spam classifier using Naive Bayes algorithm
* We have a bunch of emails classified as ['spam'](https://github.com/Surya-Murali/Email-Spam-Classifier-Using-Naive-Bayes/tree/master/emails/spam)
and a bunch of emails classified as ['ham'](https://github.com/Surya-Murali/Email-Spam-Classifier-Using-Naive-Bayes/tree/master/emails/ham)
(not spam)
* The emails are first read and stored in a dataframe. They are then parsed using CountVectorizer
* This information is used to train the model and its prediction is then tested with a sample input

**Python Libraries used:** pandas, numpy, io, os, CountVectorizer and MultinomialNB from sklearn

The [Spam classifier](https://github.com/Surya-Murali/Email-Spam-Classifier-Using-Naive-Bayes/blob/master/SpamClassifier.py) classifies the given input as a spam/ham. A sample output is found [here.](https://github.com/Surya-Murali/Email-Spam-Classifier-Using-Naive-Bayes/blob/master/SpamClassifierOutput.txt)
