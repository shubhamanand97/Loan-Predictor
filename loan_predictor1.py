#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("F:/projects/Loan_Predictor/train.csv") 	#reading data into dataframe
df1 = pd.read_csv("F:/projects/Loan_Predictor/train_u6lujuX_CVtuZ9i.csv")
y=df.Loan_Status
print df1
print df.describe()
fig = plt.figure(figsize=(8,4))

######################(graph--Probability of getting loan by credit history)
temp1 = df1.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print '\nProbility of getting loan for each Credit History class:' 
print temp1
ax1 = temp1.plot(kind= 'bar')
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Probability of getting loan')
ax1.set_title("Probability of getting loan by credit history")

###############################################

df = df[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]

train_features,test_features,train_labels,test_labels = train_test_split(df, y, test_size=0.2, shuffle=False)

#print train_features.shape
#print train_labels.shape
#print test_features.shape
#print test_labels.shape

from sklearn.ensemble import ExtraTreesClassifier
clf=ExtraTreesClassifier(n_estimators=30, min_samples_split=35,random_state=0)
clf.fit(train_features,train_labels)

pred = clf.predict(test_features)

from sklearn.metrics import accuracy_score
print accuracy_score(test_labels,pred)
