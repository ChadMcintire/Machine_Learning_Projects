# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 22:47:28 2018

@author: Chad
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vocabulary = "a list of words I want to look for in the documents".split()
vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', 
           stop_words='english', vocabulary=vocabulary)

doc = "some string I want to get tf-idf vector for"

vect.fit(vocabulary)
corpus_tf_idf = vect.transform(vocabulary) 
doc_tfidf = vect.transform([doc])

