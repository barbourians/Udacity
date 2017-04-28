# 14-build-confusion-matrix.py
# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
print ("# Load the dataset ...")
from sklearn import datasets
#X = pd.read_csv('titanic_data.csv')
X = pd.read_csv('\\Udacity\\machine-learning\\projects\\titanic_survival_exploration\\titanic_data.csv')

# Limit to numeric data
print ("# Limit to numeric data ...")
X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

print ("# from sklearn import ...")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

# TODO: split the data into training and testing sets,
# using the default settings for train_test_split (or test_size = 0.25 if specified).
# Then, train and test the classifiers with your newly split data instead of X and y.
print ("# split the data into training and testing sets ...")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)

# The decision tree classifier
print ("# The decision tree classifier ...")
clf1 = DecisionTreeClassifier()
clf1.fit(X_train,y_train)
answerDTC = confusion_matrix(y_test,clf1.predict(X_test))
print "Confusion matrix for this Decision Tree:\n",answerDTC

# The naive Bayes classifier
print ("# The naive Bayes classifier ...")
clf2 = GaussianNB()
clf2.fit(X_train,y_train)
answerNBC = confusion_matrix(y_test,clf2.predict(X_test))
print "GaussianNB confusion matrix:\n",answerNBC

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": answerNBC,
 "Decision Tree": answerDTC
}

# EOF
print ("EOF")