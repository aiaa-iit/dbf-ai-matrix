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
