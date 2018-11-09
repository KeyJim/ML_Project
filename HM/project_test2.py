#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:18:54 2018

@author: liuchuang

使用SVm 分类器进行 线性分类  结果准确率90%-----训练集太少 特征纬度太多
改变test-size 得到较高的准确率

结论1：样本数目少于特征维度并不一定会导致过拟合，这可以参考余凯老师的这句评论：
“这不是原因啊，呵呵。用RBF kernel, 系统的dimension实际上不超过样本数，与特征维数没有一个trivial的关系
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.utils import np_utils
import gzip
import Pca

dframe = pd.read_excel("test.xlsx")
test = DataFrame(dframe)
test=test.T
#X = test.iloc[ : , :-1].values 
X = finalData
Y = test.iloc[ : , 302].values 



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

"""
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
"""


from sklearn.svm import SVC
clf = SVC(C=1,kernel='linear', probability = True,random_state=0)
clf.fit(x_train, y_train)

#Predicting the Test set results
y_pred = clf.predict(x_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
#print(cm)
print(classification_report(y_test, y_pred))

predictions = [int(a) for a in clf.predict(x_test)]
num_correct = sum(int(a == y) for a, y in zip(predictions, y_test))
print("Baseline classifier using an SVM.")
print(str(num_correct) + " of " + str(len(y_test)) + " values correct.")