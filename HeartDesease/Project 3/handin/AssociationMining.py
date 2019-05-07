# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:02:10 2019

@author: Afei
"""
#%%
import numpy as np
from apyori import apriori
import xlrd

# %%

def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T

def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules

# %%
    
#import the data to X_trans matrix
doc = xlrd.open_workbook('C:/Users/A_FEI/Documents/GitHub/ML-DM/HeartDesease/heart-binary.xls').sheet_by_index(0)
X_trans = np.empty((303, 31))
for i, col_id in enumerate(range(0, 31)):
    X_trans[:, i] = np.asarray(doc.col_values(col_id, 1, 304))
    
attributeNames = doc.row_values(0, 0, 31)
attributeNames = [name for name in attributeNames]
attr_length = len(attributeNames)

T = mat2transactions(X_trans,labels=attributeNames)
rules = apriori(T, min_support=.20, min_confidence=.90)

print_apriori_rules(rules)