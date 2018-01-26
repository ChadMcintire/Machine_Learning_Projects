# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:58:51 2018

@author: Chad
"""

import sys
from sklearn import datasets
import math
import numpy as np
import operator

iris = datasets.load_iris()
from sklearn.model_selection import train_test_split

def shuffleAndGroup():
    df = iris.data
    a_train, a_test, b_train, b_test = train_test_split(df, iris.target, test_size=.3, random_state=3)
    return a_train, a_test, b_train, b_test

def experimentalShell(data_train, data_test , targets_train, targets_test, classifier):
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    return targets_predicted
    

def DisplayTest(targets_predicted, targets_test):
        count = 0
        i = 0
        
        while i < len(targets_predicted):
            if targets_predicted[i] == targets_test[i]:
                count += 1
            i += 1
        print("count = ", count)
        print("target similarity percent =", count/ len(targets_predicted) * 100)
        
class KNNModel:
    def __init__(self, data_train, targets_train):
        self.data = data_train
        self.targets = targets_train
        self.k = int(input("what k would you like to use?"))
        return None
        
    def neighborVote(self, votearray):
        votes = {}
        for k in range (self.k):
            response = votearray[k][1]
            if response in votes:
                votes[response] += 1
            else:
                votes[response] = 1
        
        sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        return(sortedVotes[0][0])     
            
        
    
    def getNeighbors(self, row):
        dist = []
        
        def euclideanDistance(instance1, instance2, length):
            distance = 0
            for x in range(length):
                distance += pow((instance1[x] - instance2[x]), 2)
            return math.sqrt(distance)
        
        for i in range(len(self.data)):
            distance = euclideanDistance(self.data[i],row, 4)
            dist.append([distance,self.data[i],self.targets[i]])
        sorted_Neighbors = sorted(dist, key = lambda flower: flower[0])        
        neighbors = []

        for i in range(self.k):
            neighbors.append([sorted_Neighbors[i][1],sorted_Neighbors[i][2]])

        return neighbors
    
    def predict(self, testData):
        targets = []
        for row in range(len(testData)):
            neighbors = self.getNeighbors(testData[row])
            self.neighborVote(neighbors)
                
            targets.append(self.neighborVote(neighbors))
        return targets
    

class KNNClassifier:
    def __init__(self, k):
        self.k = k
    
    def fit(self, data, targets):
        m = KNNModel(data, targets)
        return m
    
def main(argv):
    data_train, data_test , targets_train, targets_test = shuffleAndGroup()
    
    K_neigh = KNNClassifier(20)
    K_neighPredicted = experimentalShell(data_train, data_test , targets_train, targets_test, K_neigh)
    DisplayTest(K_neighPredicted, targets_test)
    
if __name__== "__main__":
    main(sys.argv)


