# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:10:09 2020

@author: kosea
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Data_file/veriler.csv")

cinsiyet = data.iloc[:,4:5].values
#print(cinsiyet)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

data_1 = data[["boy","kilo","yas"]]

cinsiyet[:,-1] = le.fit_transform(data.iloc[:,-1])
#print(cinsiyet)


 # kolon baslik etiket tasimak one hot encoder
# ohe ile 0 1 ler iki sutun oluyo
ohe = preprocessing.OneHotEncoder()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
#print(cinsiyet)

cinsiyet = pd.DataFrame(data = cinsiyet[:,0:1],index = range(22),columns=[" cinsiyet "])
# artık 1 se erkek 0 sa kadin olmus oldu dummy variable tuzagi gitti



ulke = data[["ulke"]].values

ulke[:,0] = le.fit_transform(ulke[:,0])

#  00 sa amerika 01 tr 10 fransa

ulke = ohe.fit_transform(ulke).toarray()
#print(ulke)

ulke = pd.DataFrame(data = ulke, index = range(22),columns =["fr" , "tr","us"])

data_2 = pd.concat([ulke,data_1], axis = 1)

data_3 = pd.concat([data_2,cinsiyet], axis = 1)

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(data_2,cinsiyet,test_size = 0.33 , random_state =0)

from sklearn.linear_model import LinearRegression

lnr = LinearRegression()

lnr.fit(x_train,y_train)

y_pred_cinsiyet = lnr.predict(x_test)

boy = data_3.iloc[:,3:4].values
sol = data_3.iloc[:,:3]
sag = data_3.iloc[:,4:]
data_4 = pd.concat([sol,sag], axis = 1)

x_train,x_test,y_train,y_test = train_test_split(data_4,boy,test_size = 0.33 , random_state =0)
lnr2 = LinearRegression()
lnr2.fit(x_train,y_train)
y_predict_boy = lnr2.predict(x_test)

# Dogru Degisken sorusu P value BACKWARD  Elimination foward elimination

import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int),values= data_4, axis =1)

X_list = data_4.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list,dtype = float)
model = sm.OLS(boy,X_list).fit() # bulmak istedigim degere gore hesapla
print(model.summary())

"""Burada  p degeri ile Yasin etkisiz oldugunu gorduk ve kaldırdık hesbaı bozuyordu"""
X_list = data_4.iloc[:,[0,1,2,3,5]].values
X_list = np.array(X_list,dtype = float)
model = sm.OLS(boy,X_list).fit() # bulmak istedigim degere gore hesapla
print(model.summary())
"""Burada ise yine p hesaplayınca 0.05 altı olsa da 0.031 cinsiyet etkisini de kaldırabiliriz
X_list = data_4.iloc[:,[0,1,2,3]].values
X_list = np.array(X_list,dtype = float)
model = sm.OLS(boy,X_list).fit() # bulmak istedigim degere gore hesapla
print(model.summary())
"""
lnr3 = LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(X_list,boy,test_size = 0.33 , random_state =0)
lnr3.fit(x_train,y_train)
y_predict_x = lnr3.predict(x_test)
