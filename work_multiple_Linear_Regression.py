# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 03:45:27 2020

@author: kosea
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing

pre = preprocessing.LabelEncoder()

ohe = preprocessing.OneHotEncoder()

data = pd.read_csv("Data_file/odev_tenis.csv")

data_sayi = data[["temperature","humidity"]]


# sunny overcast rainy 

data_hava =   data[["outlook"]].values


data_hava[:,0] = pre.fit_transform(data_hava[:,0]) # satir sutun vermeyince olmuyor

numeric_hava = ohe.fit_transform(data_hava[:,:]).toarray()

# 1. indis  overcast 2. indis rainy 3. indis sunny

data_ruzgar = data[["windy"]].values

data_ruzgar[:,0]= pre.fit_transform(data.iloc[:,3])
 # burda true false olarak bırakti 0 1 yapmadı. ana veriden cekmedigimiz icin yok abi kafasi guzel

numeric_ruzgar = ohe.fit_transform(data_ruzgar[:,:]).toarray()



data_play = data[["play"]].values

data_play[:,0] = pre.fit_transform(data.iloc[:,4]) # 0sa oynama 1 se oyna  

data_play = pd.DataFrame(data=data_play ,index = range(14), columns = ["play"])

# 2. indisi al 0 sa ruzgarsız 1 se ruzgarli

numeric_hava = pd.DataFrame(data = numeric_hava, index=range(14), columns=["overcast","rainy","sunny"])

numeric_ruzgar = pd.DataFrame(data = numeric_ruzgar[:,1], index = range(14), columns = ["windy"])

# numeric ruzgar numeric hava ve data_sayi birlestiricez

data_hava_sayi = pd.concat([numeric_hava,data_sayi],axis =1)

data_hava_sayi_ruzgar = pd.concat([data_hava_sayi,numeric_ruzgar],axis=1)

# data_hava__sayi_ruzgar  bagimsiz data_play bagimli sonuc

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data_hava_sayi_ruzgar,data_play,test_size = 0.33,random_state =0)


from sklearn.linear_model import LinearRegression

lnr = LinearRegression()
lnr.fit(x_train,y_train)

y_predict =  lnr.predict(x_test)

import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int),values= data_hava_sayi_ruzgar, axis =1)
X_list = data_hava_sayi_ruzgar.values
play = data_play.values
X_list = np.array(X_list,dtype = float)
model = sm.OLS(play,X_list).fit() # bulmak istedigim degere gore hesapla
print(model.summary())



