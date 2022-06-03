#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  20 00:00:00 2022

@author: eyobghiday
"""
import pandas as pd 
import numpy as np
import seaborn as sns #for graphing confusion ma
import nltk
nltk.download('stopwords')
import matplotlib.pyplot as plt
from matplotlib import rc

#importing data
df1=pd.read_csv('src/dbf.csv')
df1.head()

plt.figure(figsize=(7,5))
chart=sns.countplot(df1.Sentiment,palette='Set1')
plt.title('Overview of the data')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right') 
#plt.savefig('data.png')
df1=df1.sample(frac=1)
df1.groupby('Sentiment').count()

from nltk.corpus import stopwords
from nltk import word_tokenize 
stop_words = stopwords.words('english')
# I'm aadding cusom list of words to be removed
add_words = [',','``',',,','.','&']
stop_words.extend(add_words)

#removing stop words
def remove_stopwords(rev):
    review_tokenized = word_tokenize(rev)
    rev_new = " ".join([i for i in review_tokenized if i not in stop_words]) 
    return rev_new
    #adding a new column for the processed data
df1['review_processed'] = [remove_stopwords(r) for r in df1['Sentence']]
df1.groupby('Sentiment').count()
  
#for positive and 2 for negative. 
def assign_values(value):
    if value=="positive": 
        return 1
    elif value=="neutral": 
        return 0
    else:
        return 2 #else data the sentntiment is negative
df1['label'] = df1['Sentiment'].apply(assign_values)
#checking to see if my data is processed
df1.head()
# we're good to go now
df1.groupby('Sentiment').count()
#algo for the favoured statistical probability  
class probabilty_calc:
    def nbt(self, x):
        pss = [] #making a list to hold posterior prob 
        for i, j in enumerate(self._classes):
                pr = np.log(self._prs[i])
                ps = np.sum(np.log(self.inst(i, x)))
                ps = pr + ps
                pss.append(ps)
        return self._classes[np.argmax(pss)]

    def base(self, s):
        y_pred = [self.nbt(x) for x in s] 
        return np.array(y_pred)
    
    def inst(self, class_i, x):
        mean = self._mean[class_i]
        variance = self._variance[class_i]
        top = np.exp(-((x - mean) ** 2) / (2 * variance)) 
        bot = np.sqrt(2 * np.pi * variance)
        return top / bot
    def similar_data(self, s, y):
        sam, feat = s.shape
        self._classes = np.unique(y)
        classes = len(self._classes)
        # calculate mean, var, and prior for each class
        self._mean = np.zeros((classes, feat), dtype=np.float64) 
        self._variance = np.zeros((classes, feat), dtype=np.float64) 
        self._prs = np.zeros(classes, dtype=np.float64)
        for i, j in enumerate(self._classes):
            s_i = s[y == j]
            self._mean[i, :] = s_i.mean(axis=0) 
            self._variance[i, :] =s_i.var(axis=0) 
            self._prs[i] = s_i.shape[0] / float(sam)

#splitting data for training and predicting of Naive bayes probability.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
x = df1['review_processed']
y = df1['label']
#i choose 30 - 70 split with a random state of 30
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.3,random_state=30)

# define our vectorisation feature.
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

#define a functionn to get the accuracy interms of flaot number
def class_accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true) 
        return accuracy