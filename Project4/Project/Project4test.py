# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:44:25 2018

@author: Chad
"""
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
import pandas as pd
import operator



def readCSV1():    
    filename = 'C:/Users/Chad/Desktop/450/Project4/Project/iris.data'
    data = pd.read_csv(filename, skipinitialspace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['sep_len', 'sep_wid','sep_len', 'sep_wid', 'targets']
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


def calc_entropy(p):
    if p !=0:
        return -p * np.log2(p)
    else: 
        return 0  
 
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
    
#print(iris.data)    
#print(iris.target)

#print(len(iris.target))

df = readCSV1()
df.replace(df.iloc[:,4] ,categoricalToNumerical(df, 4))
data = pd.DataFrame.as_matrix(df,columns=None)
class_values = list(set(row[-1] for row in data))
#print(class_values)

for index in range(len(data[0])-1):
    for row in data:
        print(index)
        print(row)
        print(row[index],"\n")
        
print(data[148][3])
w, h = 8, 5;
Matrix = [[0 for x in range(w)] for y in range(h)]
Matrix[1][0] = 1
Matrix[2][0] = 1
print(Matrix[1])
Matrix[1] = [x+3 for x in Matrix[1]]
print(Matrix[1])
print(Matrix)




#mylist = [0]
#print(mylist)

#new_list = [x+3 for x in mylist]
#print(new_list)



#f1.append([5])
#map(lambda x:x+1, f1)
#[x+1 for x in f1]

#f1[1].append(5)
#df.replace(df.iloc[:,4] ,categoricalToNumerical(df, 4))
#data, targets = separateDataAndTargetsEnd(df,4)


#def get_split(dataset):
    
    #class_values = list(set(row[:,0] for row in targets))
    
    #print(type(class_values))
    #np.set_printoptions(precision=4)
    #print(class_values)

def doStuff(data, target):
    nData = len(data)
    print(nData)
    print(data)
    print(target)

#doStuff(data,targets)
#print(data)

#get_split(data)
