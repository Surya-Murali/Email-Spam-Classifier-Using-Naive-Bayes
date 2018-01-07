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
print(data.tail())

vectoriser = CountVectorizer()
count = vectoriser.fit_transform(data['message'].values)
print(count)

target = data['class'].values
print(target)

classifier = MultinomialNB()
classifier.fit(count, target)
print(classifier)

exampleInput = ["Hey. This is John Cena. You can't see me", "Free Viagra boys!!", "Please reply to get this offer"]
excount = vectoriser.transform(exampleInput)
print(excount)

prediction = classifier.predict(excount)
print(prediction)
