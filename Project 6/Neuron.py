# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:07:58 2018

@author: Chad
"""
import sys
import random
import numpy as np

class Neuron:
    def __init__(self):
        self.rowCount = 0
        return None
    
    def activate(self, weights, targets, row, whichNeuron):
#        for i in range(len(row)):
#            print(i)
        eta = .1
        print("rowCount =" ,self.rowCount)
        activation = np.dot(row,np.array(weights[whichNeuron]).T)
        print(activation)
        if activation >= 0.0 :
            activation = 1.0
        else:
            activation = 0.0
        print(activation)
        if activation == targets[self.rowCount]:
            print("yes")
        else:
            print("heck no")
            for j in range(len(weights[0])):
#                [a[:,:j] for j in i]
                #[weights[whichNeuron][j]] = weights[whichNeuron][j]# - eta*(activation - targets[row])*row[self.rowCount]
                print("targets" ,np.array(targets[self.rowCount]))
                print("weights 1",weights[whichNeuron][j])
                weights[whichNeuron][j] = weights[whichNeuron][j]  - eta*(activation)*row[j]
                print("Weights 2", weights[whichNeuron][j])
                #weights[whichNeuron][j] = weights[whichNeuron][j] - eta*(activation - np.array(targets[self.rowCount]))*row[j]
                #print(weights[whichNeuron][j])
        print(np.shape(weights))
        self.rowCount += 1
            
                
            
            
            #np.dot(biasPlusInputs[0],np.array(weights[whichNeuron]).T))
        
        
#    def updateWeight(self, weights,whichNeuron, targets, actvation, row):
        
#        eta = .1
#        for i in range(len(weights[0])):
#        weights[whichNeuron][j] = weights[whichNeuron][j] - eta*(activation - targets[row])*row[i]
    
    def getWeights(self, numInputs, neuronLength):
        print("num inputs plus one", numInputs+1)
        print("number of neurons created", neuronLength)
        weightSize = numInputs + 1
        weights = []
        #for x in range(0,weightSize):
        #    weights.append(random.uniform(-1, 1))
        
        
        for x in range(0,weightSize):
            weights.append([])
            for y in range(0,neuronLength):
                weights[x].append(random.uniform(-1, 1))
        print("weight array",weights)
        print("shape of weight array", np.shape(weights))
        return weights 

    def createBias(self,numInputs):
        #biasArray = []
        #for y in range(0,numInputs):
        #    biasArray.append(-1)
        #print("bias array", biasArray)
        biasArray = np.full((1, numInputs), -1)
        return biasArray
    
    #def createNeuron(self,data, bias, weights)
        


def main(argv):
    labels1 = ['A', 'B', 'C', 'D']
 
    loan = [[1,1 , 2 , 1 ],
            [2,2 , 4, 2 ],
            [3,3 ,5,1],
            [4,4 ,6,2],
            [5,1 ,2,1],
            [6,2 ,4,2],
            [7,3 ,4,1],
            [8,4 ,6,2],
            [9,1 ,8,1],
            [10,2 ,7,2],
            [11,3 ,3,1],
            [12,4 ,5,2]]
    
    targets = [[0],
               [0],
               [0],
               [1],
               [1],
               [1],
               [0],
               [0],
               [0],
               [1],
               [0],
               [1]]
    
    
    neuron = Neuron()
    numNeuron = 5
    numattributes = len(loan[0])
    datLength = len(loan)
    weights = neuron.getWeights(numattributes, numNeuron)
    bias = neuron.createBias(datLength)

    print("weigths shape", np.shape(weights))
    #print(np.shape(loan))
    biasPlusInputs = np.concatenate((loan, bias.T), axis=1)
    #print(np.shape(np.concatenate((loan, b.T), axis=1)))
    print("trying to get a feel for weights", weights[0][1])
    print(weights)
    #row, column
    print("trying to get a feel for bias and inputs", biasPlusInputs[2][3])
    print("get a row",weights[0])
    print("bias inputs feels", biasPlusInputs[0])
    print("dot product of the first row",np.dot(biasPlusInputs[0],np.array(weights[0]).T))

    for row in biasPlusInputs:
        neuron.activate(weights, targets, row, 0)
    
    print(np.array(weights[0]).T)  
    print(np.array(weights[0][0]).T)  
    
    print(len(weights[0]))
    
    for i in range(len(weights[0])):
        print(i)
    print(np.shape(weights))
    a =biasPlusInputs[0]
    print(weights[0][0])    
    weights[0][0] = weights[0][0] -  1*(1 - np.array(targets[5])*5)*a[0]
    print(weights[0][0])    
    print(np.array(targets[5])*5)
if __name__== "__main__":
    main(sys.argv)
