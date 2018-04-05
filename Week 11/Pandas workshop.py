# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 08:07:36 2018

@author: Chad
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

n_features = 500

v = TfidfVectorizer()



#filepath = "C:/Users/Chad/Desktop/450/Week 11/Try3.txt"
#df = pd.read_csv(filepath,delimiter='~', encoding='cp1252', header = None)
#it's not actually  utf-8 it's some windows format
filepath = "C:/Users/Chad/Desktop/450/Week 11/Try3.csv"
df = pd.read_csv(filepath,delimiter=',', encoding='cp1252', header = None)
#drop na's from the 2nd column

v = TfidfVectorizer()
df.dropna(subset=[1])
#print(df[1])
x = v.fit_transform(df[1].values.astype('U'))
#df['tweetsVect']=list(v.fit_transform(df[1]).toarray())
df['tweetsVect']=list(x)




data_samples = df['tweetsVect']#df[:,1]


# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples)