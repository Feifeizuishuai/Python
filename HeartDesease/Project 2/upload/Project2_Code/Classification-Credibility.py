# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 07:08:53 2019

@author: Afei
"""

import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib.pylab import figure, plot, title, xlabel, ylabel, legend, ylim, show, boxplot
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree
from sklearn.linear_model import LogisticRegression
from scipy import stats
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
# Create crossvalidation partition for evaluation
K1 = K2 = 10

#StratifiedKflod
CV_outer = model_selection.StratifiedKFold(n_splits=K1,shuffle=True)
CV_inner = model_selection.StratifiedKFold(n_splits=K2,shuffle=True)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)
# Logistic Regression complexity parameter - lambda
lambda_interval = np.logspace(-8, 3, 100)

# Initialize variables
Error_LR = np.empty((K1,K2))
Error_DT = np.empty((K1,len(tc),K2))
Error_BM = np.empty((K1,K2))
opt_lambda_all = np.empty((K1,K2))
optimal_lambda = np.empty((K1,1))
cv_test_error_LR = np.empty((K1,1))
cv_test_error_DT = np.empty((K1,1))

#Error matrix for DecisionTree
Error_train = np.empty((len(tc),K2))
Error_test = np.empty((len(tc),K2))
#--------------------------------------------------------------------------
#-----------------------------------outer----------------------------------
k =0
#X=X[:,[1,2,7,8,9,11]]
for train_index, test_index in CV_outer.split(X,y):
    print('CV-outer-fold {0} of {1}'.format(k+1,K1))
    
    # extract training and test set for current outer CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    #--------------------------------------------------------------------------
    #-----------------------------------inner----------------------------------
    j=0
    for train_index, test_index in CV_inner.split(X_train,y_train):
        print('   CV-inner-fold {0} of {1}'.format(j+1,K2))
        
        # extract training and test set for current inner CV fold
        X_train_inner = X_train[train_index,:]
        y_train_inner = y_train[train_index]
        X_test_inner = X_train[test_index,:]
        y_test_inner = y_train[test_index]
        
        # Standradiztion
        X_train_STD = StandardScaler().fit_transform(X_train_inner)
        X_test_STD = StandardScaler().fit_transform(X_test_inner)
        #%%
        #-------------------- Baseline Model-----------------------------------
        model = LogisticRegression()
        model = model.fit(X_train_STD,y_train_inner)
        
        y_train_est = model.predict(X_train_STD)
        y_test_est = model.predict(X_test_STD)
        
        Error_BM[k,j] = 100*np.sum(y_test_est != y_test_inner) / len(y_test_inner)

        #%%
        #------------ Logistic Regression classifier---------------------------
        #------------ Select best lambda in this fold---------------------------
        train_error_rate = np.zeros(len(lambda_interval))
        test_error_rate = np.zeros(len(lambda_interval))
        for count in range(0, len(lambda_interval)):
            model = LogisticRegression(penalty='l2', C=1/lambda_interval[count], solver='liblinear')
            model.fit(X_train_STD, y_train_inner)
        
            y_train_est = model.predict(X_train_STD)
            y_test_est = model.predict(X_test_STD)
            
            train_error_rate[count] = np.sum(y_train_est != y_train_inner) / len(y_train_inner)
            test_error_rate[count] = np.sum(y_test_est != y_test_inner) / len(y_test_inner)
        

        Error_LR[k,j] = 100*test_error_rate.min()
        opt_lambda_all[k,j] = lambda_interval[np.argmin(test_error_rate)]
        #%%
        #-------------------- Decision Tree classifier-------------------------
        #---------------- Select best depth in this fold-----------------------
        for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=t)
            dtc.fit(X_train_STD,y_train_inner)
            y_est_test = dtc.predict(X_test_STD)
            y_est_train = dtc.predict(X_train_STD)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = np.sum(y_est_test != y_test_inner) / float(len(y_est_test))
            misclass_rate_train = np.sum(y_est_train != y_train_inner) / float(len(y_est_train))
            Error_test[i,j], Error_train[i,j] = misclass_rate_test, misclass_rate_train

            Error_DT[k,i,j] = 100* Error_test[i,j]
        
        #%%
        j+=1
    #--------------------------------------------------------------------------
    #-----------------------------------inner----------------------------------
    Error_DT_2d = Error_DT.min(axis=2)
    
    X_train_outer_STD = StandardScaler().fit_transform(X_train)
    X_test_outer_STD = StandardScaler().fit_transform(X_test)
    
    Optimal_inner_lambda = opt_lambda_all[k,np.argmin(Error_LR[k,:])]
    model_outer = LogisticRegression(penalty='l2', C=1/Optimal_inner_lambda, solver='liblinear')
    model_outer.fit(X_train_outer_STD, y_train)
    y_est_test = model_outer.predict(X_test_outer_STD)
    cv_test_error_LR[k] = 100* np.sum(y_est_test != y_test) / float(len(y_est_test))
    
    Optimal_inner_treedepth = np.argmin(Error_DT_2d[k])
    dtc_outer = tree.DecisionTreeClassifier(criterion='entropy', max_depth=Optimal_inner_treedepth+1)
    dtc_outer.fit(X_train_outer_STD, y_train)
    y_est_test = dtc_outer.predict(X_test_outer_STD)
    cv_test_error_DT[k] = 100* np.sum(y_est_test != y_test) / float(len(y_est_test))
    
    
    k+=1
#--------------------------------------------------------------------------
#-----------------------------------outer----------------------------------

#%%
#-------------------------credibility interval calculate and plot----------------------------------
z = (-cv_test_error_DT.reshape(-1))
#z = (Error_BM.mean(axis=1)-cv_test_error_DT.reshape(-1))
#z = (Error_BM.mean(axis=1)-cv_test_error_LR.reshape(-1))
zb = z.mean()
nu = K1-1
sig =  (z-zb).std()  / np.sqrt(K1-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
x=np.linspace(-15,5,300)
y=stats.norm.pdf(x,zb,sig)
plt.title('Baseline model / Logistic Regression')
plt.plot(x,y,lw=2,color='black')
#plt.vlines(0,0,0.2,linestyles='--')
#plt.text(0,0.18,'0')
plt.vlines(zL,0,0.2,linestyles='--',colors='r')
plt.text(zL+1,0.18, "95% Left",color='r')
plt.vlines(zH,0.2,0,linestyles='--',colors='b')
plt.text(zH+1,0.18, "95% Right",color='b')
xlabel([zL, zH])
plt.grid()
plt.show()
    
#%%
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_BM.mean(axis=1).reshape(K1,-1), Error_LR.mean(axis=1).reshape(K1,-1), Error_DT_2d.mean(axis=1).reshape(K1,-1)),axis=1))
xlabel('Baseline               Logistic Regression        Decision Tree')
ylabel('Cross-validation error [%]')

figure()
X = np.arange(1,4,1)
Y = np.array([Error_BM.mean(),cv_test_error_LR.mean(),cv_test_error_DT.min()])
plot(X,Y,'or-')
xlabel('Baseline                  Logistic Regression           Decision Tree')
ylabel('Cross-validation error [%]')
plt.xticks(X)
plt.ylim([5,20])
show()


#%%
optimal_lambda_index = np.argmin(Error_LR, axis=1)
for i in range(0,K1):
    optimal_lambda[i] = opt_lambda_all[i,optimal_lambda_index[i]]

optimal_lambda=optimal_lambda.reshape(-1)
optimal_tree_depth = np.argmin(Error_DT_2d, axis=1)
optimal_tree_depth = optimal_tree_depth + 1



