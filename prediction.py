# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:05:00 2020

@author: kosea
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Data_file/satislar.csv")

# 1 Doğrusal Regresyon Linear Regression
     # y = ax + b y bağımlı değişken x bağımsız değişken a eğim b ne kadar kayacağı
     # hata miktarı doğru ile tahmin arası mesafe
     # 2 kodda Aynı işi yapıyor.
"""
data_satis = data.iloc[:,1:2].values

data_satis = pd.DataFrame(data = data_satis , index = range(30), columns= ["Satislar"])
                          
data_ay = data.iloc[:,0].values

data_ay = pd.DataFrame(data = data_ay , index = range(30), columns= ["Aylar"])
"""
data_satis = data[["Satislar"]]
data_ay = data[["Aylar"]]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data_ay,data_satis,test_size = 0.33 , random_state =0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# 1.1 Doğrusal Regresyon modelin olusturulmasi 

from sklearn.linear_model import LinearRegression

lnr = LinearRegression()
    # lnr.fit(X_train,Y_train)
    #tahmin  =lnr.predict(X_test) # Y_test hic sisteme girmedi. Scale veri yorumlanması farklı zor.
# Scale Edilmemis veri modeli egitimi
lnr.fit(x_train,y_train)
predict = lnr.predict(x_test)


# 1.2 Veri Gorsellestirme ( Basli basina ayrıı bir konu)

    # plt.plot(x_train, y_train) # Aylar sırasız olmasını duzeltmek lazım

x_train_ordered = x_train.sort_index()
y_train_ordered = y_train.sort_index()

plt.plot(x_train_ordered, y_train_ordered)  # %67 lik veriler cizili

x_test_ord = x_test.sort_index()
y_test_ord = y_test.sort_index()
plt.title("Aylara Gore Satis")
plt.xlabel("Aylar")
plt.ylabel("Satislar")
plt.plot(x_test, predict) # % 33 lük veri var
"""  Bir degiskene bagli bir bagimliya bagli veriyi bulmada kullanılıyor.
    birden fazla olarak değiskende ufak hilelerle yapilabilir ama oznitelik sayisi fazla ise 
    bu Coklu Dogrusal Regresyona girer.
"""
