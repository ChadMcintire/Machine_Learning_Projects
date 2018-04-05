# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:00:25 2018

@author: Chad
"""

import sys
import random
import numpy as np


def main(argv):
    loan = [[.4,.2],
            [.7,.1],
            [.3,.9],
            [.4,1]]
    
    weights = [[.5,-.3,-.2],
            [.3,-.6,.1],
            [.3,.9,1],
            [.4,1,1]]
    
    print(loan[0])
    print(np.append(loan[0], 0))
    stuff = np.append(loan[0], -1)
    print(stuff)
    print("dot product of the first row",np.dot(stuff,np.array(weights[0]).T))
    h1 = np.dot(stuff,np.array(weights[0]).T)
    activation1 = 1/(1 + np.exp(-h1))
    print(activation1)
    stuff = np.append(loan[0], -1)
    print("dot product of the second row",np.dot(stuff,np.array(weights[1]).T))
    h1 = np.dot(stuff,np.array(weights[1]).T)
    activation2 = 1/(1 + np.exp(-h1))
    print(activation2)
    
    biasArray = np.full((1, 1), -1)
    #biasPlusInputs = np.concatenate((loan[0], biasArray.T), axis=0)
    #try = np.stack((loan[0],biasArray), axis = 0)
    print(biasArray)

if __name__== "__main__":
    main(sys.argv)
