# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 01:42:18 2020

@author: kosea
"""

import pandas as pd 
import numpy as np
data = pd.read_csv("Data_file/nlp.csv",error_bad_lines=False)
"""sparce matris kelime vektoru cogu bos olucak stop wordler"""
# noktalama isaret kaldırma regular express
import re
#yorum = re.sub('[^a-zA-Z]',' ',data['Review'][6]) # öndeki ^ile kalıyor
#♠ donguye koyucaz

# Buyuk kucuk harfler aynı seviyeye getirme 
#yorum = yorum.lower()
#yorum = yorum.split()
# yazılmıs kelimeleri liste cevirme

 # str tipinin methodu

# stop words temizleme yapısı turkce icin se yine text indirip liste çekip


# bu sekilde ayırabiliriz

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer() # kelime koklere ayırıcak

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
""" FEATURE EXTRACTION """
derlem = []
for i in range(716):
    yorum = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    """ corpustan tum stop words alıyor kümeleri verisi stemle """
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] # teker teker geliler
    """ listeden string e cevirmem lazım"""
    yorum = ' '.join(yorum) # yorumu al boslukla stringe cevir.
    derlem.append(yorum)



""" Count vectorizer  ile feature extraction yapılıcak."""

from sklearn.feature_extraction.text import CountVectorizer

countVect = CountVectorizer(max_features=2000) # kelime sayıp en cok gecenleri almak 
X = countVect.fit_transform(derlem).toarray() # bagımsız değişken
y = data[["Liked"]].values # bagımlı değişken 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_predict = gnb.predict(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_predict,y_test)
print(cm)












