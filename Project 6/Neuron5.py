# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:52:04 2018

@author: Chad
"""

import sys
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing

def readCSV2():    
    filename = 'C:/Users/Chad/Desktop/450/Project3/Project/pima-indians-diabetes.data'
    data = pd.read_csv(filename, skipinitialspace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['pregnant', 'plasma_glucose_level','Diastolic blood pressure', 'skin fold', 'serum insulin', 'BMI', 'Diabetes pedigree function', 'age(years)','targets1']
    return df

def separateDataAndTargetsEnd(dataframe,a):
    target = dataframe.iloc[:,a]
    target = pd.DataFrame.as_matrix(target,columns=None)

    #dat = dataframe.loc[:, dataframe.columns != dataframe.columns[a]]
    dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=True)
    #print(a)
    #print(dataframe.columns[a])
    dat = pd.DataFrame.as_matrix(dataframe,columns=None)

    return dat, target
def experimentalShell(data_train, data_test , targets_train, targets_test, classifier):
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    return targets_predicted

class NNModel:
    def __init__(self, data_train, targets_train):
        self.data = data_train
        self.targets = targets_train
        return None
    
    def addBias(self,inputs):
        return np.append(inputs, -1)
    
    def checkActivation(self,inputs, weights):
        return np.dot(inputs,weights)
    
    def getActivation(self, hval):
        return 1/(1 + np.exp(-hval))
    
    def createNode(self, inputs):
        inputAndBias = self.addBias(inputs)
        #weights = [.5,-.3,-.2]
        weights = [.3,-.6,.1]
        #print(inputs)
        #print(inputAndBias)
        hval = self.checkActivation(inputAndBias,np.transpose(weights))
        #print(hval)
        hval = self.checkActivation(inputAndBias,np.transpose(weights))    
        activationVal = self.getActivation(hval)
        #print(activationVal)
        return weights, activationVal
    
    def makeNode(self, size):
        weights = []
        activationVal = 0
        for i in range(size+1): 
            weights.append(random.uniform(-.5, .5))
        return weights, activationVal
    
    def createNodelist(self, numNodes, numWeights):
        nodeList = []
        #print("look at me2", self.makeNode(2))
        for i in range(numNodes): 
            nodeList.append(self.makeNode(numWeights))
        #print("nodelist",nodeList)
        return nodeList
    
    def createLayer(self, layerArray, numInputs):
        layerList = []
        #print("numinp",numInputs )
        layerList.append(self.createNodelist(layerArray[0],numInputs))
        
        for i in range(len(layerArray)-1):
            #print(i)
            #print("forLayerParameter",layerArray[i+1])
            #print("forLayerParameter1",layerArray[i])
            layerList.append(self.createNodelist(layerArray[i+1],layerArray[i]))
                    
  
        return layerList
    
    def feedForward(self, layerList, inputs):
        inputAndBias = self.addBias(inputs)
        array = []
        array.append(-1)
        for i in range(len(layerList[0])):
            hval = self.checkActivation(inputAndBias,np.transpose(layerList[0][i][0]))
            activationVal = self.getActivation(hval)
            layerList[0][i]=  layerList[0][i][0],activationVal
            array.append(layerList[0][i][1])
        
        activationArray = [[] for d in range(len(layerList))]
        weightsArray = [[] for d in range(len(layerList))]
        for i in range(len(layerList)):
            #print("ival",i)
            for j in range(len(layerList[i])):
            #print(stuff)
                #print("jval",j)
                weightsArray[i].append(layerList[i][j][0])
                activationArray[i].append(layerList[i][j][1])
                
         
        activationArray[0].append(-1)    
            
        for i in range(len(layerList)-1):
           # print("ival", i)
            if i > 0:
                activationArray[i].append(-1)
          #      print("this is the value I want",activationArray)
        
        #print("look here look her,",len(layerList)-1)
        
        for i in range(len(layerList)-1):       
            for j in range(len(layerList[i+1])):
                hval = self.checkActivation(activationArray[i],np.transpose(weightsArray[i+1][j]))
                activationVal = self.getActivation(hval)
                activationArray[i+1][j] = activationVal
         
        for i in range(len(layerList)):       
            for j in range(len(layerList[i])):  
                layerList[i][j] = weightsArray[i][j],activationArray[i][j]
        
        
        #print("look here look her weight list,", weightsArray)
        #print("look here look her activation list,", activationArray[0][0])
        #print("\n")
        #print("layerlist =", layerList[0][0])        
        return layerList
#        
    def backwardProp(self, layerList, target):
        #print("backwardProp")
        
        errorArray = [[] for d in range(len(layerList))]
        #print("errorArray",errorArray)
        for j in range(len(layerList[-1])):
            errorArray[-1].append(layerList[-1][j][1]*(1-layerList[-1][j][1])*(layerList[-1][j][1]  - target))
        
        
        predictions = []
        for i in range(len(layerList[-1])):
            predictions.append(layerList[-1][i][1])
#        print("first node",layerList[-1][0][1])
#        print("second node",layerList[-1][1][1])
        
        
        
        
        #print(layerList[-1][j][1])  
        #print("target", target[0])
        #print(errorArray)

        #make hidden layer errors
        for i in range(len(layerList)-1):
            #print(-(i+2))
            for j in range(len(layerList[-(i+2)])):
                errorArray[-(i+2)].append(0) 
        
        #print(errorArray)
        
        for i in range(len(layerList)-1):
            #print("ival = ",  i)
            for j in range(len(layerList[-(i+2)])):
                #print("jvalue = ",j)
                for k in range(len(layerList[-(i+1)])):
                    #print("kvaz = ",k)
                    errorArray[-(i+2)][j] += layerList[-(i+2)][j][1]*(1-layerList[-(i+2)][j][1])*(errorArray[-(i+1)][k] * layerList[-(i+1)][k][0][j])
        #print(errorArray)     
                        
        #print("weights",layerList[2][0][0][1])
        #print("weights",layerList[2][1][0][1])
        #print("Activations", layerList[1][1][1])
        lr = .2
        for i in range(len(layerList)):
  #          print("ival = ",  i)
            for j in range(len(layerList[i])):
 #               print("jvalue = ",j)
                for k in range(len(layerList[i][j][0])):
#                    print("kvaz = ",k)
                    layerList[i][j][0][k] = layerList[i][j][0][k] - lr*errorArray[i][j]*layerList[i][j][1]
                    #print(layerList[i][j][0][k])
        
        for i in range(len(layerList)):
            for j in range(len(layerList[i])):
                layerList[i][j] = layerList[i][j][0],0
        
        #print(layerList)
        return layerList,predictions
class NNClassifier:
    def __init__(self):
        pass
    
    def fit(self, data, targets):
        m = NNModel(data, targets)
        return m


def main(argv):
    loan = [[1,1],
            [1,2],
            [1,1],
            [1,2]]
    
    targets = [[0],
            [1],
            [0],
            [1]]
    
    weights = [[.5,-.3,-.2],
            [.3,-.6,.1],
            [.3,.9,1],
            [.4,1,1]]
    df2 = readCSV2()
    
    df2 = df2[df2.iloc[:,3] != 0]
    data, targets = separateDataAndTargetsEnd(df2,8)
    data = preprocessing.normalize(data, norm='l2')  
    
    Classi = NNClassifier()
    model = Classi.fit(data, targets)
    targets = targets
    #model.createNode(loan[0])
     
    numInputs = len(data[0])

    forLayerParameter = [8,2,6,2]


        
    layerList = model.createLayer(forLayerParameter, numInputs)
    
    for i in range(200):
        for j in range(len(loan)):
            listtry = model.feedForward(layerList, data[j])
            layerList,predictons = model.backwardProp(listtry, targets[j])
    
    array = []
    for i in range(100):
        listtry = model.feedForward(layerList, data[i])
        layerList,predictons = model.backwardProp(listtry, targets[i])
        
        if predictons[1] > predictons[0]:
            array.append(1) 
        else:
            array.append(0)
        print("first node",predictons[0])
        print("second node",predictons[1])
        print(i)
    print(array)
    print(targets[0:20])
    count = 0.
    for i in range(len(array)):
        if int(array[i]) == int(targets[i]):
            count += 1
    
    print("Accuracy =, ",count/len(array))        
    print("[1, 0, 1, 0, 1, 0, 1, 0, 1, 1]" )
    


if __name__== "__main__":
    main(sys.argv)
    
    
