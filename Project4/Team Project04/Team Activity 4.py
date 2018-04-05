# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:21:03 2018

@author: Chad
"""

import sys
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

def getOtherCounts(dataframe):
    print("print the non-numeric data types counts")
    
    for column in dataframe:     
        print(dataframe[column].value_counts(), "\n")

def readCSV1():    
    filename = 'C:/Users/Chad/Desktop/450/Project4/Project/iris.data'
    data = pd.read_csv(filename, skipinitialspace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['sep_len', 'sep_wid','sep_len', 'sep_wid', 'targets']
    return df    

def readCSV2():    
    filename = 'C:/Users/Chad/Desktop/450/Project4/Project/lenses.data'
    data = pd.read_csv(filename, delim_whitespace=True)
    df = pd.DataFrame(data = data)
    df.columns = ['worthless','age', 'perscription','astimatic', 'tear_reproduciton_rate','need']
    df.drop(df.columns[0], axis=1, inplace=True)
    return df    

#missing values, comma deliminated
def readCSV3():    
    filename = 'C:/Users/Chad/Desktop/450/Project4/Project/house-votes-84.data'
    data = pd.read_csv(filename)
    df = pd.DataFrame(data = data)  
    df.columns = ['Target','1', '2','3', '4','5', 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    df.iloc[:,2].replace("?", np.NaN, inplace=True)
    df.dropna(inplace=True)
    changeValueForAllColumnsToMostCommon(df)
    return df    

def changeValueForAllColumnsToMostCommon(dataframe):
    for column in range(len(dataframe.columns)):
        #this changes the first value to the highest count value in the column
        dataframe.iloc[:,column].replace("?", dataframe.iloc[:,column].value_counts().index[0], inplace=True)

def changeToDiscrete(dataframe):
    
    
    print(dataframe.iloc[:,0])
    sum = (dataframe.iloc[:,0].sum())
    sample_size = len(dataframe.iloc[:,0]) 
    mean = sum/sample_size
    standard_deviation = dataframe.iloc[:,0].std(axis = 0)
#    Z-Score is =        
    #df.groupby('Hostname')[['CPU Peak', 'Memory Peak']].std()
    #standard_deviation = dataframe.iloc[:,0].std
    print(mean)
    print(standard_deviation)

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

def column(matrix, i):
    #https://stackoverflow.com/questions/903853/how-do-you-extract-a-column-from-a-multi-dimensional-array
    return [row[i] for row in matrix]

class D_Tree:
    def __init__(self):
        pass
    
def calcEntropy(col, targets):
    start = list(zip(col,targets))
    print(start)
    
    return
    
    
def root(data, targets):
    print(data, "\n",targets)
    for dataCol in data:
        print(dataCol)
        calcEntropy(dataCol, targets)
    return
    
#data, targetes
def count(d, t):
    c = table(d, t)
    for pos in range(len(d)):
            c.add(d[pos], t[pos], 1)
    c.show()
    print("Score")
    c.showScore()
    

class table:
    def __init__(self, r, c):
        self.row = r
        self.col = c
        self.r = list(set(r))
        self.c = list(set(c))
        self.sizeR = len(self.r)
        self.sizeC = len(self.c)
        self.array = [[] * self.sizeR for i in range(self.sizeR)]
        self.array = self.fill()
        return
    
    def fill(self):
        out = [[] * self.sizeR for i in range(self.sizeR)]
        for i in range(self.sizeR):
            for j in range(self.sizeC):
                out[i].append(0)
        return out
    
    def row(self, rowVals):
        self.r = list(set(rowVals))
        self.sizeR = len(self.r)
        
    def col(self, colVals):
        self.c = list(set(colVals))
        self.sizeC = len(self.c)
        
    def size(self):
        return self.sizeC, self.sizeR
    
    def getOne(self, row, col):
        for i in range(self.sizeC):
            for j in range(self.sizeR):
                if self.r[j] == row and self.c[i] == col:
                    return self.array[j][i]
                
    def add(self, row, col, val):
        for i in range(self.sizeC):
            for j in range(self.sizeR):
                if self.r[j] == row and self.c[i] == col:
                    self.array[j][i] +=val
    def get(self):
        return self.array
    
    def show(self):
        total = 0
        print ("\t",end='')
        for t in self.c:
            print(t, end='\t')
        print ("")
        for rowPos in range(len(self.array)):
            print(self.r[rowPos],end='')
            for p in self.array[rowPos]:
                total += p
                print("\t",p, end='')
            print("")
        print("Total : ", total)
        return self.array
    
    def weight(self):
        out = self.fill()
        r = 0
        for row in self.array:
            sum = 0
            for pos in row:
                sum += pos
            for pos in range(len(row)):
                out[r][pos] = self.array[r][pos] / sum
            r += 1
        return out
    
    def EScore(self):
        out = []
        w = self.weight()
        rowPos = 0
        for row in w:
            sum = 0
            for pos in row:
                if(pos <= 0):
                    e = 0 
                #print("pos", pos, "row", row)
                else:
                   e = -pos*math.log2(pos)
                #print("e =", e)
                sum += e
            out.append(sum)
            rowPos += 1
        return out
    
    def EScoreTotal(self):
        t = 0
        s = self.EScore()
        for pos in range(len(s)):
            t += s[pos]
        return t
    
    def sum(self):
        out = 0
        for rowPos in range(len(self.array)):
            for p in self.array[rowPos]:
                out += p
        return out
    
    def score(self):
        out = 0
        total = self.sum()
        e = self.EScore()
        for rowPos in range(len(self.array)):
            rowTotal = 0
            for p in self.array[rowPos]:
                rowTotal += p
            out += (rowTotal / total) * e[rowPos]
        return out
    
    def showScore(self):
        total = 0
        s = self.EScore()
        for pos in range(len(s)):
            print(self.r[pos], " : ", s[pos])
            total +=s[pos]
        print ("Total : ", total)
        print ("Weighted Total : ", self.score())
        return self.score()
    
    
    
def main(argv):
    data_train, data_test, targets_train, targets_test = train_test_split(datasets.load_iris().data, datasets.load_iris().target, test_size = .3)
    labels = ['type', 'Plot', 'Star Actors', 'Profit']
    raw   = [['Comedy', 'Deep'   ,'Yes', 'Low' ],
             ['Comedy', 'Shallow','Yes', 'High'],
             ['Drama' , 'Deep'   ,'Yes', 'High'],
             ['Drama' , 'Shallow','No' , 'Low' ],
             ['Comedy', 'Deep'   ,'No' , 'High'],
             ['Comedy', 'Shallow','No' , 'High'],
             ['Drama' , 'Deep'   ,'No' , 'Low' ]]
    
    labels1 = ['Credit_score', 'Income', 'Collateral', 'Should_loan']
 
    loan = [['good'   ,'high','good','yes'],
            ['good'   ,'high','poor','yes'],
            ['good'   ,'low' ,'good','yes'],
            ['good'   ,'low' ,'poor','no' ],
            ['average','high','good','yes'],
            ['average','low' ,'poor','no' ],
            ['average','high','poor','yes'],
            ['average','low' ,'good','no' ],
            ['low'    ,'high','good','yes'],
            ['low'    ,'high','poor','no' ],
            ['low'    ,'low' ,'good','no' ],
            ['low'    ,'low' ,'poor','no' ]]

   
    
    for rowPos in range(len(loan[0]) - 1):
        print("rowPos", rowPos)
        print("\n",labels1[rowPos])
        count(column(loan, rowPos),column(loan, -1))
#    
#    
#    df = readCSV1()    
#    df.replace(df.iloc[:,4] ,categoricalToNumerical(df, 4))
#    df.replace(df.iloc[:,0] ,categoricalToNumerical(df, 0))
#    data = pd.DataFrame.as_matrix(df,columns=None)
#    labels3 = ['sep_len', 'sep_wid','sep_len', 'sep_wid', 'targets']
#    
#    for rowPos in range(len(data[0]) - 1):
#        print(rowPos)
#        print("\n",labels3[rowPos])
#        count(column(data, rowPos),column(data, -1))
           
#    
#    df = readCSV2()
#    data = pd.DataFrame.as_matrix(df,columns=None)    
#    print(data)
#    
#    labels4 = ['age', 'perscription','astimatic', 'tear_reproduciton_rate', "need_contacts"]
#    
#    for rowPos in range(len(data[0]) - 1):
#        print(rowPos)
#        print("\n",labels4[rowPos])
#        count(column(data, rowPos),column(data, -1))    
        
    #df = readCSV3()   
    #print(df)
    #data = pd.DataFrame.as_matrix(df,columns=None)  
    #labels4 = df.columns

    #for rowPos in range(len(data[1])):
        #print(rowPos)
        #print("\n",labels4[rowPos])
        #count(column(data, rowPos+1),column(data, 0))     
    
        

if __name__ == "__main__":
    main(sys.argv)