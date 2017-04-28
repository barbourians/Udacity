# 32_compute_mean_absolute_error.py
import numpy as np
import pandas as pd

# Load the dataset
print ("# Load the dataset ...")
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

print ("# Run from sklearn import ...")
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
print ("# Split the data into training and testing sets ...")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

# The decision tree classifier
print ("# Decision tree regressor ...")
reg1 = DecisionTreeRegressor()
reg1.fit(X_train, y_train)
answerDT = mae(y_test,reg1.predict(X_test))
print "Decision Tree mean absolute error:",answerDT

# The naive Bayes classifier
print ("# Linear regression ...")
reg2 = LinearRegression()
reg2.fit(X_train, y_train)
answerLR = mae(y_test,reg2.predict(X_test))
print "Linear regression mean absolute error:",answerLR

results = {
 "Linear Regression": answerLR,
 "Decision Tree": answerDT
}

# EOF
print "EOF"