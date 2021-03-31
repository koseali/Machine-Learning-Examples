# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 03:45:27 2020

@author: kosea
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Data_file/maaslar.csv")


# dilimleme denir
x = data.iloc[:,1:2]
y = data.iloc[:,2:3]
# dataframde den numpy array e çevirilir.
X = x.values
Y= y.values

# Linear Regresyon
from sklearn.linear_model import LinearRegression

lnr = LinearRegression()

lnr.fit(X, Y)
"""
plt.scatter(X, Y,color = "red")
plt.plot(x, lnr.predict(x))
plt.show()"""
# Polynomial Regression

# non linear model Polinomal Regresyon

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)

x_poly = poly_reg.fit_transform(X)

print(x_poly)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)

# Gorsellestirme
plt.scatter(X,Y,color = "red")
plt.plot(X, lin_reg.predict(x_poly))
plt.show()




poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()

lin_reg.fit(x_poly,y)
plt.scatter(X,Y,color = "red")
plt.plot(X, lin_reg.predict(x_poly))
plt.show()

a = lin_reg.predict(poly_reg.fit_transform([[6.6]]))
print(a)
# 3.4 Support Vector Machine Uygulama

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
sc2 = StandardScaler()

x_olcekli = sc1.fit_transform(X)

y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")


svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli)

plt.plot(x_olcekli,svr_reg.predict(x_olcekli))

plt.show()
svr_reg.predict([[5]])


# decision tree

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0 )

r_dt.fit(X,Y)
Z = X +0.5
K = X - 0.4
plt.scatter(X,Y,color = "red")
plt.plot(x,r_dt.predict(Z), color = "red")
plt.plot(x,r_dt.predict(K), color = "yellow")
plt.plot(X, r_dt.predict(X), color= "green")

print(r_dt.predict([[2]]))






