# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 13:47:51 2018

@author: Chad
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
#%matplotlib inline
from subprocess import check_output