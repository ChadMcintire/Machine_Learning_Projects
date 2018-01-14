# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:41:15 2018

@author: Chad
"""

from sklearn import datasets

iris = datasets.load_iris()
import csv

with open(iris.data) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row)
      
        
def readFile(filename):
    if ".txt" in filename: 
        df = pd.read_csv(filename, header = None)
    elif ".csv" in filename:
        with open(filename) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                print(row)
        #df = pd.read_csv(filename)
    else:
        #df = loadtxt(datasets.load_iris(), comments="#", delimiter=",", unpack=False)
        #how can I get this in the right format, it adds counts and a header
        #df = pd.DataFrame(data= filename['data'])
        #columns= filename['feature_names']
    return df