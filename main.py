# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 07:55:34 2018

@author: foyaz
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


dataset = pd.read_csv('fake_or_real_news.csv')
X = dataset.iloc[:, 2]  
Y = dataset.iloc[:, -1]


from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
cv_train = cv.fit_transform(X_train)
cv_test = cv.transform(X_test)

clf = MultinomialNB()
clf.fit(cv_train, Y_train)
pred = clf.predict(cv_test)
score = metrics.accuracy_score(Y_test, pred) *100
print('Accuracy with CountVectorizer  :',score)




from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv = TfidfVectorizer(stop_words='english', max_df=0.8)
tfidfv_train = tfidfv.fit_transform(X_train)
tfidfv_test = tfidfv.transform(X_test)

clf = MultinomialNB()
clf.fit(tfidfv_train, Y_train)
pred = clf.predict(tfidfv_test)
score = metrics.accuracy_score(Y_test, pred) *100
print('Accuracy with TfidfVectorizer  :',score)











