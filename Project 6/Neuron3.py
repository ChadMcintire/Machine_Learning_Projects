# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:52:04 2018

@author: Chad
"""

import sys
import random
import numpy as np


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
            weights.append(random.uniform(-1, 1))
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
        print("numinp",numInputs )
        layerList.append(self.createNodelist(layerArray[0],numInputs))
        
        for i in range(len(layerArray)-1):
            print(i)
            print("forLayerParameter",layerArray[i+1])
            print("forLayerParameter1",layerArray[i])
            layerList.append(self.createNodelist(layerArray[i+1],layerArray[i]))
            #layerList.append(self.createNodelist(layerArray[i+1],layerArray[i+1]))
            
        
        for i in range(len(layerList)):
            for j in range(len(layerList[i])):
                print("see shape,",i,j, np.shape(layerList[i][j][0]))
        
        print(layerList[3])
        
        return layerList
    
    def feedForward(self, layerList, inputs):
        inputAndBias = self.addBias(inputs)
        array = []
        array.append(-1)
        for i in range(len(layerList[0])):
            hval = self.checkActivation(inputAndBias,np.transpose(layerList[0][i][0]))
            #print(hval)
            activationVal = self.getActivation(hval)
            layerList[0][i]=  layerList[0][i][0],activationVal
            array.append(layerList[0][i][1])
        
        activationArray = [[] for d in range(len(layerList))]
        weightsArray = [[] for d in range(len(layerList))]
        for i in range(len(layerList)):
            print("ival",i)
            for j in range(len(layerList[i])):
            #print(stuff)
                print("jval",j)
                weightsArray[i].append(layerList[i][j][0])
                activationArray[i].append(layerList[i][j][1])
         
        activationArray[0].append(-1)    
            
        for i in range(len(layerList)-1):
            print("ival", i)
            if i > 0:
                activationArray[i].append(-1)
                print(activationArray)
        
        print("look here look her,",len(layerList)-1)
        
        for i in range(len(layerList)-1):       
            for j in range(len(layerList[i+1])):
                hval = self.checkActivation(activationArray[i],np.transpose(weightsArray[i+1][j]))
                activationVal = self.getActivation(hval)
                activationArray[i+1][j]=  activationVal
                
        print("look here look her,",activationArray)
             
#        for j in range(len(layerList[1])):
#            hval = self.checkActivation(activationArray[0],np.transpose(weightsArray[1][j]))
#            activationVal = self.getActivation(hval)
#            activationArray[1][j]=  activationVal
#            
#        print("try two array",activationArray)
#        for j in range(len(layerList[2])):
#            hval = self.checkActivation(activationArray[1],np.transpose(weightsArray[2][j]))
#            activationVal = self.getActivation(hval)
#            activationArray[2][j]=  activationVal
#            
#        print("try 3 array",activationArray)
#        
#        for j in range(len(layerList[3])):
#            hval = self.checkActivation(activationArray[2],np.transpose(weightsArray[3][j]))
#            activationVal = self.getActivation(hval)
#            activationArray[3][j]=  activationVal
            
            
        print("array after 1",array)
        print("list after 1",layerList[0][1][1])
        print("activationArray array total", activationArray)
        print("activationArray array total shape", np.shape(activationArray))
        print("weightsArray array total", weightsArray[1])
        print("activationArray array", activationArray[3])
        print("weightsArray array", weightsArray[3][1])
        print("end test")

        
class NNClassifier:
    def __init__(self):
        pass
    
    def fit(self, data, targets):
        m = NNModel(data, targets)
        return m


def main(argv):
    loan = [[.4,.2],
            [.7,.1],
            [.3,.9],
            [.4,1]]
    
    targets = [[1],
            [2],
            [1],
            [2]]
    
    weights = [[.5,-.3,-.2],
            [.3,-.6,.1],
            [.3,.9,1],
            [.4,1,1]]
    
    Classi = NNClassifier()
    model = Classi.fit(loan[0], targets)
    model.createNode(loan[0])
 


#    personList = []
    #personList.append(model.makeNode(2))
#    for i in range(4+1): 
#        personList.append(model.makeNode(2))
#    print(personList)
#    personList.append(model.createNode(loan[1]))

#    a,b = personList[0]
#    print(b,a)
#    personList[0] = [0,1,2], 2

    
#    try1 = model.createNodelist(5)
#    try1[0] = [0,1,2], 2

    
    numInputs = len(loan[0])

    
    #model.feedForward(layerList, loan[0])
    
    forLayerParameter = [2,4,4,2]
    for i in forLayerParameter:
        print("paramval = i", i)
    for i in range(len(forLayerParameter)-1):
        print(i)
        print("forLayerParameter",forLayerParameter[i])
        
    layerList = model.createLayer(forLayerParameter, numInputs)
    
    model.feedForward(layerList, loan[0])

if __name__== "__main__":
    main(sys.argv)
    
    