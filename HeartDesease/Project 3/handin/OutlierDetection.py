# -*- coding: utf-8 -*-
"""
Created on Thu April 29 01:05:44 2019

@author: Afei
"""

import numpy as np
from matplotlib.pyplot import figure, bar, title, plot, show, subplot, hist
from toolbox_02450 import gausKernelDensity
import xlrd
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import NearestNeighbors
from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning)

# %%
# Data Preprocessing
doc = xlrd.open_workbook('C:/Users/A_FEI/Documents/GitHub/ML-DM/HeartDesease/heart.xls').sheet_by_index(0)
attributeNames = doc.row_values(0, 0, 14)
attr_length = len(attributeNames)
#import the data to X_og matrix
#all data
X_og = np.empty((303, 14))
for i, col_id in enumerate(range(0, 14)):
    X_og[:, i] = np.asarray(doc.col_values(col_id, 1, 304))

X_standard = StandardScaler().fit_transform(X_og)
X = X_standard

# %%
# Gaussian KDE with Leave-one-out
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()
   print('Fold {:2d}, w={:f}, value={:f}'.format(i,w,logP[i]))
   
val = logP.max()
ind = logP.argmax()
width=widths[ind]
print('Optimal estimated width is: {0} fold: {1}'.format(width,ind))
density, log_density = gausKernelDensity(X,width)
i = (density.argsort(axis=0)).ravel()
density = density[i].reshape(-1,)

figure(figsize=(12,4))
subplot(121)
plot(range(len(logP)),logP)
plot(ind, logP[ind], 'o')
xlabel = logP
title('Optimal estimated width')
subplot(122)
bar(range(20),density[:20])
title('Density estimate: Outlier score')

    
# %%
# K-nearest neigbor density
K = 4
knn = NearestNeighbors(n_neighbors=K).fit(X)
knn_D, knn_i = knn.kneighbors(X)
knn_density = 1./(knn_D.sum(axis=1)/K)
knn_i = knn_density.argsort()
knn_density = knn_density[knn_i]

figure()
bar(range(20),knn_density[:20])
title('KNN density: Outlier score')


# %%
# K-nearest neigbor average relative density
knn = NearestNeighbors(n_neighbors=K).fit(X)
knn_ard_D, knn_ard_i = knn.kneighbors(X)
knn_density_ = 1./(knn_ard_D.sum(axis=1)/K)
knn_ard_density = knn_density_/(knn_density_[knn_ard_i[:,1:]].sum(axis=1)/K)
knn_ard_i = knn_ard_density.argsort()
knn_ard_density = knn_ard_density[knn_ard_i]

figure()
bar(range(20),knn_ard_density[:20])
title('KNN average relative density: Outlier score')
