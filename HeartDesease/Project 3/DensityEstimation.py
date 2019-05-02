# -*- coding: utf-8 -*-
"""
Created on Thu May  2 01:05:44 2019

@author: Afei
"""

import numpy as np
from matplotlib.pyplot import figure, bar, title, plot, show, subplot, hist
from toolbox_02450 import gausKernelDensity
import xlrd
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import NearestNeighbors
from scipy.stats.kde import gaussian_kde
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
# x-values to evaluate the KDE

xe = np.linspace(-10, 10, 100)
# Compute kernel density estimate
for i in range(attr_length):
    X = X_standard[:,i]
    kde = gaussian_kde(X.ravel())
    # Plot kernel density estimate
    figure(figsize=(8,3))
    subplot(1,2,1)
    hist(X,xe)
    title('Data histogram')
    subplot(1,2,2)
    plot(xe, kde.evaluate(xe))
    title('Kernel density estimate')
    show()
    
    
# %%
K = 60
# K Nearest Neighbour density & K Nearest Neighbour Average Relative Density
    
# Find the k nearest neighbors
for i in range(attr_length):
    X = X_standard[:,i].reshape(-1,1)
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(np.matrix(xe).T)
    
    # Compute the density
    #D, i = knclassifier.kneighbors(np.matrix(xe).T)
    knn_density = 1./(D.sum(axis=1)/K)
    
    # Compute the average relative density
    DX, iX = knn.kneighbors(X)
    knn_densityX = 1./(DX[:,1:].sum(axis=1)/K)
    knn_avg_rel_density = knn_density/(knn_densityX[i[:,1:]].sum(axis=1)/K)
    
    
    # Plot KNN density
    figure(figsize=(12,2))
    subplot(131)
    hist(X,xe)
    title('Data histogram')
    subplot(132)
    plot(xe, knn_density)
    title('KNN density')
    # Plot KNN average relative density
    subplot(133)
    plot(xe, knn_avg_rel_density)
    title('KNN average relative density')

show()
