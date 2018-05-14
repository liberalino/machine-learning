'''
This code is an example of classification using three types of classifiers.
It trains the classifiers using data from people and try to predict the gender
of a given person.

Dependencies:
Scikit-learn (http://scikit-learn.org/stable/install.html)
numpy (pip install numpy)
scipy (pip install scipy)
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np


# Taining features
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
# Training labels
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Test sample:
sample = [190, 70, 43]

# Define classifiers

# 1. Decision tree
# 2. Support Vector Machine
# 3. Gaussian Naive Bayes

clf_tree = DecisionTreeClassifier()
clf_svm = SVC()
clf_nb = GaussianNB()


# Training
clf_tree = clf_tree.fit(X, Y)
clf_svm = clf_svm.fit(X,Y)
clf_nb = clf_nb.fit(X,Y)

# Prediction
prediction_tree = clf_tree.predict(X)
prediction_svm = clf_svm.predict(X)
prediction_nb = clf_nb.predict(X)

# Accuracy scores
acc_tree = accuracy_score(Y, prediction_tree) * 100
acc_svm = accuracy_score(Y, prediction_svm) * 100
acc_nb = accuracy_score(Y, prediction_nb) * 100
print('Accuracy for DecisionTreeClassifier: {}'.format(acc_tree))
print('Accuracy for SVC: {}'.format(acc_svm))
print('Accuracy for GaussianNB: {}'.format(acc_nb))

# The best classifier:
index = np.argmax([acc_tree, acc_svm, acc_nb])
classifiers = {0: 'Tree', 1: 'SVM', 2: 'Naive Bayes'}
print('Best gender classifier is {}'.format(classifiers[index]))
