import numpy as np
from sklearn.naive_bayes import GaussianNB

#Assigning features and target variables
#Training data
features = np.array([[1,2],[3,4], [5,6], [-1,-2], [-3,-4], [-5,-6]])
target = np.array([10, 20, 20, 10, 20, 20])

#Creating a Gaussian Classifier
model = GaussianNB()

#Train the model using training data
model.fit(features, target)

#Predict Output 
predictedOutput = model.predict([[3,2],[1,3]])
print (predictedOutput)

#Output: ([20 10])
