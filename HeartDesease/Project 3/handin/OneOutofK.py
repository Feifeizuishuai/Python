# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:41:16 2019

@author: Afei
"""
import pandas as pd
import xlrd
import numpy as np
from similarity import binarize2


doc = xlrd.open_workbook('C:/Users/A_FEI/Documents/GitHub/ML-DM/HeartDesease/heart.xls').sheet_by_index(0)
X_og = np.empty((303, 14))
for i, col_id in enumerate(range(0, 14)):
    X_og[:, i] = np.asarray(doc.col_values(col_id, 1, 304))
    
attributeNames_og = doc.row_values(0, 0, 14)
attributeNames_og = [name for name in attributeNames_og]

cp = pd.get_dummies(X_og[:,2])
restecg = pd.get_dummies(X_og[:,6])
slope = pd.get_dummies(X_og[:,8])
thal = pd.get_dummies(X_og[:,12])
sex = pd.get_dummies(X_og[:,1])
traget =  pd.get_dummies(X_og[:,13])
doc = xlrd.open_workbook('C:/Users/A_FEI/Documents/GitHub/ML-DM/HeartDesease/heart-binary.xls').sheet_by_index(0)
X_og2 = np.empty((303, 5))
for i, col_id in enumerate(range(0, 5)):
    X_og2[:, i] = np.asarray(doc.col_values(col_id, 1, 304))

attributeNames = doc.row_values(0, 0, 5)
attributeNames = [name for name in attributeNames]


X, attributeNamesBin = binarize2(X_og2, attributeNames)