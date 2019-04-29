# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:49:18 2019

@author: Afei
"""

import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib.pylab import figure, plot, title, xlabel, ylabel, legend, ylim, show
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler 

#%% 
# ------------------------------data preparation-----------------------------
# ---------------------------------------------------------------------------


#import the data set
doc = xlrd.open_workbook('C:/Users/A_FEI/Documents/GitHub/ML-DM/HeartDesease/heart.xls').sheet_by_index(0)
attributeNames = doc.row_values(0, 0, 14)

#import the data to X_og matrix
#all data
X_og = np.empty((303, 14))
for i, col_id in enumerate(range(0, 14)):
    X_og[:, i] = np.asarray(doc.col_values(col_id, 1, 304))

M = len(attributeNames)
X_index= np.arange(0,M-2)
X = X_og[:,X_index]
y = X_og[:,M-1]
X=X[:,[1,2,7,8,9,11]]
#%%
# ------------------logistic regression with regularization term-------------
#----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.50, random_state=0, stratify=y)

#standradization
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

lambda_interval = np.logspace(-2, 3, 100)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    model = LogisticRegression(penalty='l2', C=1/lambda_interval[k], solver='liblinear')
    model.fit(X_train, y_train)

    y_train_est = model.predict(X_train).T
    y_test_est = model.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = model.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-2, 13.5, "Test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.text(1e-2, 13, "Accuracy : " + str(np.round(100-min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
#plt.ylim([0, 30])
plt.grid()
plt.show()    
'''
plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    

'''















