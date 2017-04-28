# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
print ("# Load the dataset ...")
#X = pd.read_csv('titanic_data.csv')
X = pd.read_csv('\\Udacity\\machine-learning\\projects\\titanic_survival_exploration\\titanic_data.csv')

print ("# Limit to numeric data ...")
X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

print ("# Run from sklearn import ...")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
print ("# Split the data into training and testing sets ...")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

# The decision tree classifier
print ("# The decision tree classifier ...")
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
answerDTCrecall = recall(y_test,clf1.predict(X_test))
answerDTCprecision = precision(y_test,clf1.predict(X_test))
print "Decision Tree recall:",answerDTCrecall,"and precision:",answerDTCprecision

# The naive Bayes classifier
print ("# The Naive Bayes classifier ...")
clf2 = GaussianNB()
clf2.fit(X_train, y_train)
answerNBCrecall = recall(y_test,clf2.predict(X_test))
answerNBCprecision = precision(y_test,clf2.predict(X_test))
print "Gaussian Naive Bayes recall:",answerNBCrecall,"and precision:",answerNBCprecision

results = {
  "Naive Bayes Recall": answerNBCrecall,
  "Naive Bayes Precision": answerNBCprecision,
  "Decision Tree Recall": answerDTCrecall,
  "Decision Tree Precision": answerDTCprecision
}

# EOF
print ("EOF")