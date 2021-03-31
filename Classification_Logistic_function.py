# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:17:45 2020

@author: kosea
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Karar Agaci 

data = pd.read_csv("Data_file/veriler.csv")

data_s = data[["boy","kilo","yas"]].values

data_cins = data.iloc[:,4].values
# karar agaci basta nereden bolucez entropi hesabı ile yapıyoruz.

# leaf node  ile  agactaki bolge sayisi esit.

# en son yas ortalamalari yapraklara yaziyoruz.  yani boyutu indirgiyoruz. yas tahmin degeri

# uzayda yapraklarda hepsi için aynı degerde  yazılıyor. o yaprak uzayında nerdeki kolay

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data_s,data_cins, random_state = 0)

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(x_train)
X_test = std.transform(x_test)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=0)

log_reg.fit(X_train , y_train)

y_pred = log_reg.predict(X_test)

# Confucin Matris Sınıflandırma  Başarısı nasıl ölçütlenir.Karmaşıklık Matrisi
# TP TN FN FPDerste ogrendiklerimiz. :)
from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(y_test, y_pred)

print(cf_matrix)


