# 32_mean_absolute_error.py
# As usual, use a train/test split to get a reliable F1 score from two classifiers, and
# save it the scores in the provided dictionaries.

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
from sklearn.metrics import f1_score
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
answerDTC = f1_score(y_test, clf1.predict(X_test))
print "Decision Tree F1 score:",answerDTC

# The naive Bayes classifier
print ("# The Naive Bayes classifier ...")
clf2 = GaussianNB()
clf2.fit(X_train, y_train)
answerNBC = f1_score(y_test, clf2.predict(X_test))
print "GaussianNB F1 score:",answerNBC

F1_scores = {
 "Naive Bayes": answerNBC,
 "Decision Tree": answerDTC
}

# EOF
print "EOF"