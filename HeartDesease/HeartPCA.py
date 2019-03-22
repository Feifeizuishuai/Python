# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 20:31:10 2019

@author: Afei
"""

import numpy as np
import xlrd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import matplotlib.pyplot as plt
from scipy.linalg import svd

#import the data set
doc = xlrd.open_workbook('../heartdisease/NewHeart.xlsx').sheet_by_index(0)
attributeNames = doc.row_values(0, 0, 12)

#create the label of each attributes for later use
classLabels = doc.col_values(11, 1, 304)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))
y = np.asarray([classDict[value] for value in classLabels])

#import the data to X matrix
X = np.empty((303, 12))
for i, col_id in enumerate(range(0, 12)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 304))

M = len(attributeNames)
C = len(classNames)

'''
i = 3
j = 4
for c in range(C):
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.8)
legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])
'''

#using PCA to our data
Y = X - np.ones((303,1))*X.mean(axis=0) #subtract the mean
Y=Y/Y.std(axis=0,ddof=1) #divided by standard diviation
U,S,V = svd(Y,full_matrices=False) #svd
rho = (S*S) / (S*S).sum() # calculate the variances

# set a threshold for princpal components
threshold = 0.90

#visualization for the Variance explained by principal components
plt.figure(figsize=(10,6)) # change the size of picture
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()      
plt.show()

#data project on to the lower dimension
V = V.T #Transposition
Z = Y @ V # Projection

# choose the PC1 and PC2
i = 0
j = 1

# data projection on the first 2 dimensions
plt.figure(figsize=(10,6))
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.8)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
plt.show()

'''
import seaborn as sns
plt.figure(figsize=(10,6))
sns.scatterplot(x=V[:,0],y=V[:,1],hue=attributeNames)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(attributeNames)
plt.show()
'''

# the principal directions of the first 3 dimensions by histogram
plt.figure(figsize=(10,6))
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames) # set the axis name
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)# label: PC1, PCA2 and PC3
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()
