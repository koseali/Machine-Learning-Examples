# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 02:38:45 2020

@author: kosea
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Data_file/veriler.csv")

data_s = data[["boy","kilo","yas"]].values

data_cins = data.iloc[:,4].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data_s,data_cins, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 5 ,p = 2 ,metric = "minkowski")

knn.fit(x_train, y_train)

y_predict = knn.predict(x_test)


from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(y_test, y_predict)

print(cf_matrix)


"""
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(x_train)
X_test = std.transform(x_test)


"""