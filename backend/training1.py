import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data0 = pd.read_csv('datall.csv')

data0.head()

dict_ = {1:"summer",2:"spring",3:"autumn",4:"winter"}
data = data0.copy()
data['season']=data['y'].map(lambda x: dict_[x])
data = data.drop(columns="y")
data.head()
listx = ['summer','spring','winter','autumn']
train, test = train_test_split(data, test_size = 0.4, stratify = data['season'], random_state = 42)
train = train[['season','r','g','b','hue','sat']]
X_train = train[['hue','sat']]
Y_train = train.season
X_test = test[['hue','sat']]
Y_test = test.season
'''
from sklearn.tree import DecisionTreeClassifier, plot_tree
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 66)
mod_dt.fit(X_train,Y_train)
prediction=mod_dt.predict(X_test)
print(metrics.classification_report(Y_test, pred, target_names=listx))
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,Y_test)))
'''
train, test = train_test_split(data, test_size = 0.3, stratify = data['season'], random_state = 42)
train = train[['season','l','r','g','b','hue','sat','val']]
X_train = train[['g','b','val','sat']]
Y_train = train.season
X_test = test[['g','b','val','sat']]
Y_test = test.season
clf=SVC()
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)

#print(metrics.classification_report(Y_test, pred, target_names=listx))
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(pred,Y_test)))