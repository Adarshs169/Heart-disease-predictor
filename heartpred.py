# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 04:26:51 2019

@author: adars
"""
import pandas as pd
dataset=pd.read_csv('heart.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.cross_validation import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(criterion='entropy',random_state=0)
clf.fit(x_tr,y_tr)
y_pd=clf.predict(x_ts)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_ts,y_pd)


