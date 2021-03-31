# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:08:45 2020

@author: kosea
"""

# Polinomal Kategorik veriyi sayısal ve kategorik şekilde çevirme 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as mtb

data = pd.read_csv("Data_file/veriler.csv")

print(data)

ulke = data.iloc[:,0:1].values
print(ulke)
#  1.3 encoder Kategorik verilerden Numerik veri Donusum Islemi 

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(data.iloc[:,0])
print(ulke)


 # kolon baslik etiket tasimak one hot encoder

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


# 1.4 Veri Birlestirme  numpy verileri ( array) DataFrame e pandas cevirme index ekleyerek

sonuc = pd.DataFrame(data = ulke, index= range(22), columns=["fr","tr","us"])
print(sonuc)

yas = data.iloc[:,1:4].values # iloc ile veri okuma satir sutnu isimsiz 
sonuc2 = pd.DataFrame(data = yas , index = range (22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = data.iloc[:,-1].values

sonuc3 = pd.DataFrame(data = cinsiyet , index = range(22) , columns=["cinsiyet"])

print(sonuc3)
print(sonuc2)
# pandas concat ile veri birleştirme 
bagimsiz = pd.concat([sonuc,sonuc2],axis=1)
bagimli = sonuc3

clear_data = pd.concat([sonuc,sonuc2,sonuc3], axis =1)
print(clear_data)


# Veri Test ve train data set bölmek 
# dikey eksende bagimli bagimsiz diye ayırıyoruz. 
#yatay eksende ise test ve train diye ayırıyorz.

#  1.5 Eğitim ve Test için  bolunmesi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(bagimsiz,bagimli,test_size = 0.33 , random_state =0)

# bagimsiz parametrelerin birbirlerine etkilerine bakmak için standart  Scaler
# Oznitelik Olceklendirme 

# 1.6 Verilerin  Olceklendirmesi Normalizasyon ya da Standartlaştırma 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)















                                                             