# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:16:23 2019

@author: Afei
"""
import seaborn as sns
import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib.pylab import figure, plot, title, xlabel, ylabel, legend, ylim, show, boxplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn import model_selection, tree
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
# ----------------------------Classification Tree----------------------------
#----------------------------------------------------------------------------
# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

#standradization
X = StandardScaler().fit_transform(X)

# K-fold crossvalidation
K = 10 
CV = model_selection.KFold(n_splits=K,random_state=42,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train

    k+=1

    
min_error = np.min(Error_test.mean(1))
opt_idx = np.argmin(Error_test.mean(1))
opt = opt_idx+2

f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
plt.title("Test Error with different tree depth")
plot(tc, Error_train.mean(1))
plot(opt, min_error, 'o')
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
    
show()
figure()
cm_lr = confusion_matrix(y_test,y_est_test)
plt.title("Decision Tree Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False)
print(classification_report(y_test,y_est_test))
