# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:59:17 2019

@author: Afei
"""

# exercise 11.2.3
import numpy as np
from matplotlib.pyplot import figure, subplot, plot, hist, title, show
from sklearn.neighbors import NearestNeighbors
import xlrd
from sklearn.preprocessing import StandardScaler 


doc = xlrd.open_workbook('C:/Users/A_FEI/Documents/GitHub/ML-DM/HeartDesease/heart.xls').sheet_by_index(0)
attributeNames = doc.row_values(0, 0, 14)
attr_length = len(attributeNames)
#import the data to X_og matrix
#all data
X = np.empty((303, 14))
for i, col_id in enumerate(range(0, 14)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 304))


X_standard = StandardScaler().fit_transform(X)
X1 = X_standard

x = np.linspace(-10, 10, 100)
# Number of neighbors
K = 60
attr_length = len(attributeNames)
for i in range(attr_length):
    X = X1[:,i].ravel().reshape(-1,1)
    
# x-values to evaluate the KNN
    xe = np.linspace(-10, 10, 100)
    
    # Find the k nearest neighbors
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
    subplot(1,3,1)
    hist(X,x)
    title('Data histogram')
    subplot(1,3,2)
    plot(xe, knn_density)
    title('KNN density')
    # Plot KNN average relative density
    subplot(1,3,3)
    plot(xe, knn_avg_rel_density)
    title('KNN average relative density')
    
    show()
    
