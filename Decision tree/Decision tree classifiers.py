# -*- coding: utf-8 -*-
"""

Created on Wed Jan 31 14:10:00 2018

@author: Chad
"""



import sys
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


def column(matrix, i):
    #https://stackoverflow.com/questions/903853/how-do-you-extract-a-column-from-a-multi-dimensional-array
    return [row[i] for row in matrix]
    
#data, targetes
def count(d, t):
    c = table(d, t)
    for pos in range(len(d)):
            c.add(d[pos], t[pos], 1)
    c.show()
    print("Score")
    c.showScore()

#assumes last spot is targets
class makeTree:
    def __init__(self, data, labels): #might want to pre remove Targetes
        self.labels = labels
        # get targets NOTE: May CHange based On Data
        self.targets = column(data, -1)
        # clean data
        self.data = data
        self.removeColumn(-1)
        
        self.nodes = []
        
        self.root = self.makeRoot()
        
        self.fillTree(self.root)
        
        return
    
    def fillTree(self, start):
        for child in start.createChildern():
            print ("Name : ",child.getName())
            child.showData()
        
    
    def makeRoot(self):
        bestPos, bestVal = self.getLowist(self.getOptions(self.data, self.targets))        
        data, targets = self.split(self.data, self.targets, bestPos)
        root = node(self.labels[bestPos], '0', data, targets, list(set(column(self.data, bestPos))),self.labels, None)
        self.nodes.append(root)
        return root
    
    def removeColumn(self, pos):
        for row in range(len(self.data)):
            del self.data[row][pos]
        return self.data
    
    def split(self, data, targets, pivot):
        options  = list(set(column(data, pivot)))
        outTargets = [[] * len(options) for i in range(len(options))]
        out = [[] * len(options) for i in range(len(options))]
        for rowPos in range(len(data)):
            for pos in range(len(options)):
                if data[rowPos][pivot] == options[pos]:
                    out[pos].append(data[rowPos])
                    outTargets[pos].append(targets[rowPos])
        return out, outTargets
        
    def getLowist(self, list):
        outVal = list[0]
        outPos = 0
        for pos in range(len(list)):
            if outVal > list[pos]:
                outVal = list[pos]
                outPos = pos
        return outPos, outVal
    
    def getOptions(self, data, targets):
        options = []    
        for colPos in range(len(data[0])):
            col = column(data, colPos)
            tempTable = table(col, targets)
            for pos in range(len(col)):
                tempTable.add(col[pos],targets[pos], 1)
            options.append(tempTable.score())  
        return options
        
class node:
    idTracker = 0
    def __init__(self, name, id, data, targets, splitLabels, labesls, parent):
        self.name = name
        self.id = id
        self.data = data
        self.targets = targets
        self.labels = splitLabels
        self.childern = []
        self.type = 'leaf' # need to make function to find this out
        self.table = table #may be overkill
        #self.createChildern()
        self.parent = parent
        self.numChildern = 0
        self.heads = labesls
        self.pivot = None
        return
    def newTable(self, table):
        self.table = table
    def table(self):
        return self.table
    def changeName(self, name):
        self.name = name
    def name(self):
        return self.name
    def getName(self):
        return self.name
    def addChild(self, name):
        self.numChildern += 1
        self.childern.append(name)
    def getChildern(self):
        return self.childern
    def getNumChildern(self):
        return self.numChildern
    
    def showData(self):
        for dataSet in range(len(self.data)):
            for rowPos in range(len(self.data[dataSet])):
                print(self.name, " : ", rowPos, ' : ' ,self.labels[dataSet]," : ", self.data[dataSet][rowPos], " @\t", self.targets[dataSet][rowPos])
            print ("****")
        return
    
    def nameChildern(self):
        self.idTracker = self.idTracker + 1
        if self.name != 'Root':
            out = self.id + '.' + str(self.idTracker)
            return out
        return self.idTracker
    
    def createChildern(self):
        childern = []
        for dataSetPos in range(len(self.data)):
            bestPos, bestVal = self.getLowist(self.getOptions(self.data[dataSetPos], self.targets[dataSetPos]))
            data, targets = self.split(self.data[dataSetPos], self.targets[dataSetPos], bestPos)
            kid = node(self.labels[dataSetPos], self.nameChildern(), data, targets, list(set(column(self.data[dataSetPos], bestPos))), self.heads, self.name)
            childern.append(kid)
            self.addChild(kid.getName)
        return childern
            
    def getOptions(self, data, targets):
        options = []    
        for colPos in range(len(data[0])):
            col = column(data, colPos)
            tempTable = table(col, targets)
            for pos in range(len(col)):
                tempTable.add(col[pos],targets[pos], 1)
            options.append(tempTable.score())  
        return options
    
    def getLowist(self, list):
        outVal = list[0]
        outPos = 0
        for pos in range(len(list)):
            if outVal > list[pos]:
                outVal = list[pos]
                outPos = pos
        return outPos, outVal
    
    def split(self, data, targets, pivot):
        self.pivot = pivot
        options  = list(set(column(data, pivot)))
        outTargets = [[] * len(options) for i in range(len(options))]
        out = [[] * len(options) for i in range(len(options))]
        for rowPos in range(len(data)):
            for pos in range(len(options)):
                if data[rowPos][pivot] == options[pos]:
                    out[pos].append(data[rowPos])
                    outTargets[pos].append(targets[rowPos])
        return out, outTargets

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
                if pos > 0:
                    e = -pos*math.log2(pos)
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
    #labels = ['type', 'Plot', 'Star Actors', 'Profit']
    raw   = [['Comedy', 'Deep'   ,'Yes', 'Low' ],
             ['Comedy', 'Shallow','Yes', 'High'],
             ['Drama' , 'Deep'   ,'Yes', 'High'],
             ['Drama' , 'Shallow','No' , 'Low' ],
             ['Comedy', 'Deep'   ,'No' , 'High'],
             ['Comedy', 'Shallow','No' , 'High'],
             ['Drama' , 'Deep'   ,'No' , 'Low' ]]
    labels = ['Credit Score', 'Income', 'Collateral']
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
    print(type(loan))
    print(type(datasets.load_iris().data))
    print(type(datasets.load_iris().data.tolist()))
    #tree = makeTree(loan, labels)
    tree2 = makeTree(datasets.load_iris().data.tolist(), labels)
    
    #for rowPos in range(len(loan[0]) - 1):
    #    print("\n",labels[rowPos])
    #    count(column(loan, rowPos),column(loan, -1))
    



if __name__ == "__main__":
    main(sys.argv)
