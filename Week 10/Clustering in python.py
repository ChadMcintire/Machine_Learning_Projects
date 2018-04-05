# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 21:14:42 2018

@author: Chad
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv("C:/Users/Chad/Desktop/450/Week 10/myData.csv",  delimiter=", ", engine='python')
#df = pd.DataFrame(data,[1,2,3,4,5,6,7])

#df = pandas.DataFrame(data, xlab, tlab )

#df.columns.values = "new_name"
#df.rename(columns={'two':'new_name'}, inplace=True)

#df.drop(df.columns[1],axis=1,inplace=True)

print(data.columns)

tlab = [ "1", "error"]    
xlab = ["1/4","1/8", "1/16", "1/32"]

newvals = df.columns.values[2]
#try = newvals.split(",")
print(newvals.split(",")[0])
#df.columns.values = try

#print(df.columns.values)
#print(df.columns)
kmeans = KMeans(n_clusters=4)
#kmeans.fit(df)
