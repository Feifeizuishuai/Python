# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 07:33:08 2019

@author: Afei
"""

# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score

from toolbox_02450 import rlr_validate

mat_data = loadmat('Heart_mat.mat')
X = mat_data['X']
y = mat_data['Y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

internal_cross_validation = 10    

model = LinearRegression()
parameters = {'C': lambdas}
CVgrid = GridSearchCV(model, parameters, scoring="accuracy", cv=internal_cross_validation)
CVgrid = CVgrid.fit(X_train, y_train)

print(CVgrid.best_params_)
print(CVgrid.best_score_)

scores = cross_val_score(CVgrid, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))










