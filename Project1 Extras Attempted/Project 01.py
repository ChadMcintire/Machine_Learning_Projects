# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:34:15 2018

@author: Chad
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#from sklearn.cross_validation import train_test_split
from sklearn import datasets
iris = datasets.load_iris()
import pandas as pd
from sklearn.model_selection import KFold

#should this be a class?
def readFile(filename):
    if ".txt" in filename: 
        df = pd.read_csv(filename, header = None)
    elif ".csv" in filename:
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(data= np.c_[filename['data']],
                     columns= filename['feature_names'])
    return df


def displayIris():
    # Show the data (the attributes of each instance)
    print(iris.data)
    
    # Show the target values (in numeric format) of each instance
    print(iris.target)
    
    # Show the actual target names that correspond to each number
    print(iris.target_names)
    
    print(iris.data.shape)

    print(readFile(iris))

       #  df = pd.DataFrame(data= np.c_[filename['data'], filename['target']],
        #             columns= filename['feature_names'] + ['target'])


def randomizeData():
    
    #create a dataframe with readFile
    df = readFile(iris)
    
    #randomize the data, frac is tells how much of the data to manipulate out of 1, so all the data
    shuffle = df.sample(frac=1)
    
    print(shuffle)
    
    return shuffle

#print(randomizeData())

def shuffleAndGroup():
    #shuffle1 = randomizeData()

    #print("shuffle")

    df = readFile(iris)

    #random_state makes reproduceable random results
    
    #response = input("What percent of the data do you want for the test size(enter as a whole number): ")    
    #testsize = float(response)/100
    #print(testsize)
    a_train, a_test, b_train, b_test = train_test_split(df, iris.target, test_size= .3, random_state = 5 )
    #print(a_test.shape)
    return a_train, a_test, b_train, b_test





def displayShuffleAndGroup():

    a, b, c, d = shuffleAndGroup()
    
    print(a.shape)
    print(b.shape)
    print("a_train", a, )
    
    print("a_test", b)
    
    print("b_train", c)
    
    print("b_test", d)
    

#data_train, data_test , targets_train, targets_test = shuffleAndGroup()
#classifier = GaussianNB()
#model = classifier.fit(data_train, targets_train)
#targets_predicted = model.predict(data_test)

print("more")

class GausianModel:
    def __init__(self):
        pass
    data_train, data_test , targets_train, targets_test = shuffleAndGroup()
    classifier = GaussianNB()
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    count = 0
    i = 0
    while i < len(targets_predicted):
        #print(targets_predicted[i], targets_test[i])
        if targets_predicted[i] == targets_test[i]:
            count += 1
        i += 1    
    print("count = ", count)    
    print("target similarity percent =", count/ len(targets_predicted) * 100)


print("done")
class HardCodedModel:
    def __init__(self):
        pass
    
    def predict(self, data):
        targets = []
        
        for row in data:
            targets.append(self.predict_one(row))
        
        return targets
    
    def predict_one(self,row):
        return 0 
    
class HardCodedClassifier:
    def __init__(self):
        pass  
    
    def fit(self, data, target):
        return HardCodedModel

print("done more")    
   

    

    