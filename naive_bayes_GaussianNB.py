import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

# Import statement
from sklearn.naive_bayes import GaussianNB

# Create the Classifier
clf = GaussianNB()

# Fit iy using X and Y
clf.fit(X, Y)

# Create a vector of predictions
pred = clf.predict([[-0.8, -1]])
print (pred)

# Create the Classifier
clf_pf = GaussianNB()

# Fit iy using X and Y and uique(?)
clf_pf.partial_fit(X, Y, np.unique(Y))

# Create a vector of predictions
pred = clf.predict([[-0.8, -1]])
print (pred)
