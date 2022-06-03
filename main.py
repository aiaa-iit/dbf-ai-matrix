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