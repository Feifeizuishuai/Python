# -*- coding: utf-8 -*-
"""
Created on Mon Apr 1 01:45:38 2019

@author: Afei

"""
import seaborn as sns
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib.pylab import figure, plot, title, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
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


#%%  
# -------------------------------baseline model?-----------------------------
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.80,random_state=42, stratify=y)


model = lm.logistic.LogisticRegression()
model = model.fit(X_train,y_train)

y_est= model.predict(X_test)
y_est_LargestClass = model.predict_proba(X)[:, 1] 

y_train_est = model.predict(X_train)
y_test_est = model.predict(X_test)


Error_Test = 100*np.sum(y_test_est != y_test) / len(y_test)

x_class = model.predict_proba(X_test)

print('- Test error:     {0}'.format(Error_Test))
print('- Accuracy:       {0}'.format(100*accuracy_score(y_test,y_est)))

f = figure();
plt.figure(figsize=[4.5,6])
class0_ids = np.nonzero(y==1)[0].tolist()
plot(class0_ids, y_est_LargestClass[class0_ids], 'ob')
class1_ids = np.nonzero(y==0)[0].tolist()
plot(class1_ids, y_est_LargestClass[class1_ids], 'or')
plt.title("Baseline Model")
xlabel('Data object (Heart disease sample)'); ylabel('Predicted prob. of class Have Heart Disease');
legend(['Have Disease', 'No Disease'])
ylim(-0.01,1.5)

show()

figure()
cm_lr = confusion_matrix(y_test,y_test_est)
plt.title("Baseline Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False)