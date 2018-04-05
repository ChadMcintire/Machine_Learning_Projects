# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:23:41 2018

@author: Chad
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold

import sys
import math
import operator

def readCSV1():    
    filename = 'C:/Users/Chad/Desktop/450/Project3/Project/car.csv'
    data = pd.read_csv(filename, skipinitialspace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['buying', 'maint','doors', 'persons', 'trunk-space', 'safety', 'targets0']
    return df

def readCSV2():    
    filename = 'C:/Users/Chad/Desktop/450/Project3/Project/pima-indians-diabetes.data'
    data = pd.read_csv(filename, skipinitialspace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['pregnant', 'plasma_glucose_level','Diastolic blood pressure', 'skin fold', 'serum insulin', 'BMI', 'Diabetes pedigree function', 'age(years)','targets1']
    return df

def readCSV3():    
    filename = 'C:/Users/Chad/Desktop/450/Project3/Project/auto-mpg.data'
    data = pd.read_csv(filename, delim_whitespace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['mpg', 'cylinders','displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin','car_name']
    return df

def changeValueForAllColumnsToMostCommon(dataframe):
    for column in range(len(dataframe.columns)):
        #this changes the first value to the highest count value in the column
        dataframe.iloc[:,column].replace("?", dataframe.iloc[:,column].value_counts().index[0], inplace=True)

def categoricalToNumerical(df, columnNum):
    count = 0
    di = dict()
    s = set(df.iloc[:,columnNum])
    for e in s:
    #print(count)
    #print(e)
        count += 1
        di.update({e:count})
    #df.replace
    df.iloc[:,columnNum].replace(di, inplace=True)
    return df.iloc[:,columnNum]

def kFoldSplit(n, data,target_data):
    kf = KFold(n_splits=n, random_state=1, shuffle=True)
    kf.get_n_splits(data)
    
    #should I shuffle the data before I put it into a k-fold?
    KFold(n_splits=n)
    for train_index, test_index in kf.split(data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target_data[train_index], target_data[test_index]
    return X_train, X_test, y_train, y_test

def separateDataAndTargetsEnd(dataframe,a):
    target = dataframe.iloc[:,a]
    target = pd.DataFrame.as_matrix(target,columns=None)

    #dat = dataframe.loc[:, dataframe.columns != dataframe.columns[a]]
    dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=True)
    #print(a)
    #print(dataframe.columns[a])
    dat = pd.DataFrame.as_matrix(dataframe,columns=None)

    return dat, target

def separateDataAndTargetsBeginning(dataframe,b):
    target = dataframe.iloc[:,0]
    target = pd.DataFrame.as_matrix(target,columns=None)
    #dat = dataframe.iloc[:,1:b]
    dataframe.drop(dataframe.columns[0], axis=1, inplace=True)
    dat = pd.DataFrame.as_matrix(dataframe,columns=None)
    
    return dat, target

def experimentalShell(data_train, data_test , targets_train, targets_test, classifier):
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    return targets_predicted
    

def DisplayTestFloat(targets_predicted, targets_test):
        count = 0
        i = 0
        
        while i < len(targets_predicted):
            #if targets_predicted[i] == targets_test[i]:
            if abs(targets_predicted[i] - targets_test[i]) <= 2:    
                count += 1
            i += 1
        print("count = ", count)
        print("target similarity percent =", count/ len(targets_predicted) * 100)
        
def DisplayTestInt(targets_predicted, targets_test):
        count = 0
        i = 0
        
        while i < len(targets_predicted):
            if targets_predicted[i] == targets_test[i]:
            #if abs(targets_predicted[i] - targets_test[i]) <= 2:    
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
    def __init__(self):
        pass
    
    def fit(self, data, targets):
        m = KNNModel(data, targets)
        return m


    
def main(argv):
    print("Cars Data Set predictor")
    df = readCSV1()
    for i in range(len(df.columns)):
        df.replace(df.iloc[:,i] ,categoricalToNumerical(df, i))
    data, targets = separateDataAndTargetsEnd(df,6)
    data = preprocessing.normalize(data, norm='l2')
    data_train, data_test , targets_train, targets_test = kFoldSplit(10, data,targets)
    
    K_neigh = KNNClassifier()
    K_neighPredicted = experimentalShell(data_train, data_test , targets_train, targets_test, K_neigh)
    DisplayTestInt(K_neighPredicted, targets_test)

    print("Pima Data Set Predictor")   
    df2 = readCSV2()
    #taking out the 0 values from columns 3, or 4 like below reduces the accuracy
    #now it doesn't, weird
    df2 = df2[df2.iloc[:,3] != 0]
    data, targets = separateDataAndTargetsEnd(df2,8)

    #normalizing brings down the results 
    data = preprocessing.normalize(data, norm='l2')  
    #train the data
    data_train, data_test , targets_train, targets_test = kFoldSplit(10, data, targets)

    K_neigh = KNNClassifier()
    K_neighPredicted = experimentalShell(data_train, data_test , targets_train, targets_test, K_neigh)
    DisplayTestInt(K_neighPredicted, targets_test)
  
    print("MPG Data Set Predictor")   
    df3 = readCSV3()
    #df3.replace(df3.iloc[:,8] ,categoricalToNumerical(df3, 8))
    df3.drop(df3.columns[len(df3.columns)-1], axis=1, inplace=True)
    changeValueForAllColumnsToMostCommon(df3)
    #print(df3)
    data, targets = separateDataAndTargetsBeginning(df3,3)

    data = preprocessing.normalize(data, norm='l2') 
    data_train, data_test , targets_train, targets_test = kFoldSplit(10, data, targets)

    K_neigh = KNNClassifier()
    K_neighPredicted = experimentalShell(data_train, data_test , targets_train, targets_test, K_neigh)
    DisplayTestFloat(K_neighPredicted, targets_test)
    
    
if __name__== "__main__":
    main(sys.argv)


 






#print(df)