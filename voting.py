# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 05:32:07 2019

@author: adars
"""

import pandas as pd
dataset=pd.read_csv('heart.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.cross_validation import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import VotingClassifier ,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
rn=RandomForestClassifier()
from sklearn.svm import SVC
sm=SVC()
vot=VotingClassifier(estimators=[('lr',log),('rf',rn),('svc',sm)],voting='hard')
vot.fit(x_tr,y_tr)
y_pd1=vot.predict(x_ts)
from sklearn.metrics import confusion_matrix,accuracy_score
cm1=confusion_matrix(y_ts,y_pd1)
acc=accuracy_score(y_ts,y_pd1)
