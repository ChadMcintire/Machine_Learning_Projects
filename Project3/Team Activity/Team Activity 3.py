# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:22:09 2018

@author: Chad
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

#pd.read_csv('pandas_dataframe_importing_csv/example.csv')
filename = 'C:/Users/Chad/Desktop/450/adult.data'
data = pd.read_csv(filename, skipinitialspace=True)
df = pd.DataFrame(data = data)
df.columns = ['age', 'workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'GDP']
#print(df)
#print(list(df.columns.values))


#print(df.dtypes)

obj_df = df.select_dtypes(include=['object']).copy()

#print(obj_df)

nulls = obj_df[obj_df.isnull().any(axis=1)]

def getColumnDataTypes(dataframe):
    print("Column Data Types", "\n")
    
    for column in dataframe: 
        print( column, ",    " , dataframe[column].dtype)
    print("\n")

def getNumericCounts(dataframe):
    print("print the numeric data types counts")
    print("\n")
    
    for column in df:
         if str(df[column].dtype) == 'int64':        
             print(df[column].value_counts(), "\n")

def getOtherCounts(dataframe):
    print("print the non-numeric data types counts")
    
    for column in df:
         if str(df[column].dtype) == 'object':        
             print(df[column].value_counts(), "\n")

def changeValueForAllColumnsToMostCommon(df):
    for column in range(len(df.columns)):
        #this changes the first value to the highest count value in the column
        df.iloc[:,column].replace("?", df.iloc[:,column].value_counts().index[0], inplace=True)



#X_scaled = preprocessing.scale(df)
           
#getNumericCounts(df)
#getOtherCounts(df)
#getColumnDataTypes(df)

changeValueForAllColumnsToMostCommon(df)
getOtherCounts(df)

