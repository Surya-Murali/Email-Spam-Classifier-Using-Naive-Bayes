# -*- coding: utf-8 -*-
# coding: utf-8
#Naive Bayes
import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Function to read files (emails) from the local directory
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

#An empty dataframe with 'message' and 'class' headers
data = DataFrame({'message': [], 'class': []})

#Including the email details with the spam/ham classification in the dataframe
data = data.append(dataFrameFromDirectory('C:/Users/surya/Desktop/DecemberBreak/emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('C:/Users/surya/Desktop/DecemberBreak/emails/ham', 'ham'))

#Head and the Tail of 'data'
data.head()
data.tail()

#CountVectorizer is used to split up each message into its list of words
#Then we throw them to a MultinomialNB classifier function from scikit
#2 inputs required: actual data we are training on and the target data
vectorizer = CountVectorizer()

#Take the message values from the data
#Vectorizer.fit_transformer: tokenises/ converts individual words into numbers(values). and counts how many times each word occurs.
#How many times each word occurs in an email
#Represents the count of each word in a sparse matrix
counts = vectorizer.fit_transform(data['message'].values)
print(counts)
#ham/spam
targets = data['class'].values

classifier = MultinomialNB()
classifier.fit(counts, targets)

#Inputs
examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
#example_counts = vectorizer.transform(examples)
#Convert the examples to the same format as how we first trained the data
example_counts = vectorizer.transform(examples)
print(example_counts)
predictions = classifier.predict(example_counts)
print(predictions)
