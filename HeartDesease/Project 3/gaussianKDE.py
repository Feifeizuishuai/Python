# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:12:39 2019

@author: Afei

"""
import numpy as np
from matplotlib.pyplot import figure, bar, title, plot, show, subplot
from toolbox_02450 import gausKernelDensity
import xlrd
from sklearn.preprocessing import StandardScaler 

doc = xlrd.open_workbook('C:/Users/A_FEI/Documents/GitHub/ML-DM/HeartDesease/heart.xls').sheet_by_index(0)
attributeNames = doc.row_values(0, 0, 14)
attr_length = len(attributeNames)
#import the data to X_og matrix
#all data
X_og = np.empty((303, 14))
for i, col_id in enumerate(range(0, 14)):
    X_og[:, i] = np.asarray(doc.col_values(col_id, 1, 304))


X_standard = StandardScaler().fit_transform(X_og)
#X = X_standard
# Estimate the optimal kernel density width, by leave-one-out cross-validation
for i in range(attr_length):
    X=X_standard[:,i].reshape(-1,1)
    widths = 2.0**np.arange(-10,10)
    logP = np.zeros(np.size(widths))
    for i,w in enumerate(widths):
        f, log_f = gausKernelDensity(X, w)
        logP[i] = log_f.sum()
    val = logP.max()
    ind = logP.argmax()
    
    width=widths[ind]
    print('Optimal estimated width is: {0}'.format(width))
    
    # Estimate density for each observation not including the observation
    # itself in the density estimate
    density, log_density = gausKernelDensity(X, width)
    
    # Sort the densities
    i = (density.argsort(axis=0)).ravel()
    density = density[i]
    
    # Display the index of the lowest density data object
    print('Lowest density: {0} for data object: {1}'.format(density[0],i[0]))
        
    # Plot density estimate of outlier score
    figure(figsize=(8,3))
    subplot(1,2,1)
    bar(range(20),density[:20].reshape(-1,))
    title('Density estimate')
    subplot(1,2,2)
    plot(logP)
    title('Optimal width')
show()


