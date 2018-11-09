#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 00:36:50 2018

@author: liuchuang

使用决策树进行分类，由于数据少，特征多，不利于分类

引入 PCA 压缩数据，提取特征，---

准确率在 90%


"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.utils import np_utils
import gzip
from sklearn import tree
from sklearn.metrics import classification_report
import Pca


dframe = pd.read_excel("test.xlsx")
test = DataFrame(dframe)
test=test.T
#X = test.iloc[ : , :-1].values 

X= finalData
Y = test.iloc[ : , 302].values 



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=0)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))