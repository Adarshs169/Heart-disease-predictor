# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 05:03:25 2019

@author: adars
"""


import pandas as pd
dataset=pd.read_csv('heart.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.cross_validation import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_tr=sc.fit_transform(x_tr)
from sklearn.svm import SVC
cl=SVC(kernel='linear')
cl.fit(x_tr,y_tr)
y_pd1=cl.predict(x_ts)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_ts,y_pd1)