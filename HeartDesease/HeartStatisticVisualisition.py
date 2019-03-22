# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:25:41 2019

@author: Longfei Lin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
heart = pd.read_excel('../heartdisease/NewHeart.xlsx')

'''
different kind of pictures for each attribute
include scatterplot, heapmap and histogram.
'''
# 1.age
plt.figure(figsize=(12,6))
agePic = sns.countplot(heart.age,hue=heart.target)
plt.legend(["Haven't Disease", "Have Disease"])
plt.title('Distribution of age')
plt.show()
sns.distplot(heart.age,kde=False,bins=50,color='r')
plt.title('Distribution of age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
# 2.sex
sns.countplot(heart.sex,hue=heart.target)
plt.title('Count of Diseases and not about Sex')
plt.xlabel('Sex  (1 = Male; 0 = Female)')
plt.ylabel('Count')
plt.legend(["Haven't Disease", "Have Disease"])
plt.show()

# 3.chest pain
sns.countplot(heart.cp,hue=heart.target)
plt.title('Count of disease about Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Count')
plt.show()

# 4.trestbps
sns.scatterplot(heart.trestbps, heart.age,hue=heart.target)
plt.xlabel('Trestbps: resting blood pressure')
plt.legend(["Haven't Disease", "Have Disease"])
plt.show()
sns.distplot(heart.trestbps,kde=False,bins=50,color='y')
plt.title('Distribution of trestbps')
plt.xlabel('Trestbps: resting blood pressure')
plt.ylabel('Count')
plt.show()

# 5.chol
sns.scatterplot(heart.chol, heart.age,hue=heart.target)
plt.xlabel('Chol: serum cholesterol in mg/dl')
plt.legend(["Haven't Disease", "Have Disease"])
plt.show()
sns.distplot(heart.chol,kde=False,bins=50)
plt.title('Distribution of Chol')
plt.xlabel('Chol: serum cholesterol in mg/dl')
plt.ylabel('Count')
plt.show()

# 6. fbs
sns.countplot(heart.fbs,hue=heart.target)
plt.title('Count of disease about FBS')
plt.xlabel('Fbs: fasting blood sugar > 120mg/dl (1 = true; 0 = false)')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Count')
plt.show()

# 7.restecg
sns.countplot(heart.restecg,hue=heart.target)
plt.title('Count of disease about restecg')
plt.xlabel('Restecg: resting electrocardiographic results ')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Count')
plt.show()

# 8. thalach
#plt.figure(figsize=(10,6))
sns.scatterplot(heart.thalach, heart.age,hue=heart.target)
plt.xlabel('Thalach: maximum heart rate achieved')
plt.legend(["Haven't Disease", "Have Disease"])
plt.show()
sns.distplot(heart.thalach,kde=False,bins=60)
plt.title('Distribution of thalach')
plt.xlabel('Thalach: maximum heart rate achieved')
plt.ylabel('Count')
plt.show()

# 9.Exang: exercise induced angina
sns.countplot(heart.exang,hue=heart.target)
plt.title('Count of disease about exang')
plt.xlabel('Exang: exercise induced angina')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Count')
plt.show()

# 10.oldpeak

sns.scatterplot(heart.oldpeak, heart.age,hue=heart.target)
plt.xlabel('Oldpeak: ST depression induced by exercise relative to rest')
plt.legend(["Haven't Disease", "Have Disease"])
plt.show()

plt.figure(figsize=(15,5))
sns.distplot(heart.oldpeak,kde=False,bins=60,color='g',label='count')
sns.distplot(heart.oldpeak[heart.target==0],kde=False,bins=60,color='r')
plt.title('Distribution of oldpeak')
plt.xlabel('Oldpeak: ST depression induced by exercise relative to rest')
plt.ylabel('Count')
plt.show()

# 11.slope
sns.countplot(heart.slope,hue=heart.target)
plt.title('Count of disease about Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Count')
plt.show()


'''
there are something wrong (data issues) with these 2 attributes,
so we delete these 2 attributes
 
# 12.ca
sns.countplot(heart.ca,hue=heart.target)
plt.title('Count of disease about ca')
plt.xlabel('Ca: number of major vessels (0-3) colored by flouroscopy')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Count')
plt.show()

# 13.Thal
sns.countplot(heart.thal,hue=heart.target)
plt.title('Count of disease about thal')
plt.xlabel('thal')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Count')
plt.show()

#sns.pairplot(heart)
'''

#Heatmap of the correlation coeffieient
f,ax = plt.subplots(figsize=(16,16))
sns.heatmap(heart.corr(), annot=True, linewidths=.8,cmap='YlGnBu', ax=ax)
plt.show()




















