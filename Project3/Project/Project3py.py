# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:32:55 2018

@author: Chad
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold


def readCSV1():    
    filename = 'C:/Users/Chad/Desktop/450/Project3/Project/car.csv'
    data = pd.read_csv(filename, skipinitialspace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['buying', 'maint','doors', 'persons', 'trunk-space', 'safety', 'targets']
    return df

def getColumnDataTypes(dataframe):
    print("Column Data Types", "\n")
    
    for column in dataframe: 
        print( column, ",    " , dataframe[column].dtype)
    print("\n")

def changeValueForAllColumnsToMostCommon(dataframe):
    for column in range(len(dataframe.columns)):
        #this changes the first value to the highest count value in the column
        dataframe.iloc[:,column].replace("?", dataframe.iloc[:,column].value_counts().index[0], inplace=True)




def getOtherCounts(dataframe):
    print("print the non-numeric data types counts")
    
    for column in dataframe:     
        print(dataframe[column].value_counts(), "\n")
                 

df = readCSV1()
#getOtherCounts(df)

print("separation")

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

df.replace(df.iloc[:,0] ,categoricalToNumerical(df, 0))
df.replace(df.iloc[:,1] ,categoricalToNumerical(df, 1))
df.replace(df.iloc[:,2] ,categoricalToNumerical(df, 2))
df.replace(df.iloc[:,3] ,categoricalToNumerical(df, 3))
df.replace(df.iloc[:,4] ,categoricalToNumerical(df, 4))
df.replace(df.iloc[:,5] ,categoricalToNumerical(df, 5))
df.replace(df.iloc[:,6] ,categoricalToNumerical(df, 6))


#getOtherCounts(df)

targets = df.iloc[:,6]
df = df.iloc[:,0:6] 

convertedData = pd.DataFrame.as_matrix(df,columns=None)
convertedTargets = pd.DataFrame.as_matrix(targets,columns=None)
#print(convertedData)
#print(convertedTargets)
#print(targets)

def kFoldSplit(k, data,target_data):
    kf = KFold(n_splits=k,random_state=1, shuffle=True)
    kf.get_n_splits(data)
    
    #should I shuffle the data before I put it into a k-fold?
    #KFold(n_splits=4, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target_data[train_index], target_data[test_index]
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = kFoldSplit(10, convertedData,convertedTargets)

#print(X_train.shape)
#print(y_train.shape)
#print(y_test)

def readCSV2():    
    filename = 'C:/Users/Chad/Desktop/450/Project3/Project/pima-indians-diabetes.data'
    data = pd.read_csv(filename, skipinitialspace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['pregnant', 'plasma_glucose_level','Diastolic_blood_pressure', 'skin_fold ', 'serum_insulin', 'BMI', 'Diabetes_pedigree_function', 'age(years)','targets']
    return df

df2 = readCSV2()
#df2 = df2[df2.iloc[:,3] != 0]
#df2 = df2[df2.iloc[:,4] != 0]
#getOtherCounts(df2)

targets2 = df2.iloc[:,8]
df2 = df2.iloc[:,0:8]

data = pd.DataFrame.as_matrix(df2,columns=None)

#print(df2)
    #np.set_printoptions(suppress=True)
#print(data)

convertedData = preprocessing.normalize(data, norm='l2')
#print(convertedData)


#getOtherCounts(df2)
#print(df2.shape)
#print(df2)

def readCSV3():    
    filename = 'C:/Users/Chad/Desktop/450/Project3/Project/auto-mpg.data'
    data = pd.read_csv(filename, delim_whitespace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['mpg', 'cylinders','displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin','car name']
    return df

df3 = readCSV3()
df3.replace(df3.iloc[:,8] ,categoricalToNumerical(df3, 8))
changeValueForAllColumnsToMostCommon(df3)
#print(df3)
print(df3.iloc[:,8].value_counts())
targets3 = df3.iloc[:,8]
targets3 = pd.DataFrame.as_matrix(targets2,columns=None)
df3 = df3.iloc[:,0:8]
data3 = pd.DataFrame.as_matrix(df3,columns=None)

