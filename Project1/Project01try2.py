# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:58:51 2018

@author: Chad
"""

import sys
from sklearn import datasets

iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def shuffleAndGroup():
    df = iris.data
    a_train, a_test, b_train, b_test = train_test_split(df, iris.target, test_size=.3, random_state=5)
    return a_train, a_test, b_train, b_test

def experimentalShell(data_train, data_test , targets_train, targets_test, classifier):
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    return targets_predicted
    

def DisplayTest(targets_predicted, targets_train):
        count = 0
        i = 0
        while i < len(targets_predicted):
            if targets_predicted[i] == targets_train[i]:
                count += 1
            i += 1
        print("count = ", count)
        print("target similarity percent =", count/ len(targets_predicted) * 100)
        
        
class HardCodedModel:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        return None
    
    def predict(self, testData):
        targets = []
        for row in testData:
            targets.append(self.predict_one(row))
        return targets
    
    def predict_one(self,row):
        return 0 
    
class HardCodedClassifier:
    def __init__(self):
        pass  
    
    #why does it break when I pass self into fit as a parameter
    def fit( data, targets):
        m = HardCodedModel(data,targets)
        return m
    
#class KNeighborsClassifier:
 #   def __init__(self,n_neighbors):
  #      self.neighbors = n_neighbors
        
    #why does it break when I pass self into fit as a parameter
   # def fit( data, targets):
    #    m = HardCodedModel(data,targets)
     #   return m
    



def main(argv):
    data_train, data_test , targets_train, targets_test = shuffleAndGroup()
    
    K_neigh = KNeighborsClassifier(n_neighbors=3, algorithm = 'kd_tree')
    K_neighPredicted = experimentalShell(data_train, data_test , targets_train, targets_test, K_neigh)
    DisplayTest(K_neighPredicted, targets_train)
    
    gauss = GaussianNB()
    gaussPredicted = experimentalShell(data_train, data_test , targets_train, targets_test, gauss)
    DisplayTest(gaussPredicted, targets_train)

    hardCode = HardCodedClassifier
    hardCodedPredicted = experimentalShell(data_train, data_test , targets_train, targets_test, hardCode)
    DisplayTest(hardCodedPredicted, targets_train)
    
if __name__== "__main__":
    main(sys.argv)


