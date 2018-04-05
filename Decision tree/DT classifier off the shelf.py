# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:35:05 2018

@author: Chad
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
iris = datasets.load_iris()


X = iris.data
Y = iris.target



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=200, max_depth=3, min_samples_leaf=4).fit(X_train, y_train)
#clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=400, max_depth=4, min_samples_leaf=5)
#scores = cross_val_score(clf_entropy, X, Y, cv=4)
#print("Accuracy clf entropy : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
y_pred = clf_entropy.predict(X_test)
print("Desicion tree:", accuracy_score(y_test, y_pred) * 100)

with open("gini_tree.dot", "w") as f:
    f = tree.export_graphviz(clf_entropy, out_file=f)
