#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:57:02 2018

@author: liuchuang

使用Tpot 自动进行参数优化
最好的模型是 LinearSVC -- 准确率 89.15%
自动生成文件--[link](https://github.com/LiuChuang0059/ML_Project/blob/master/HM/tpot_mnist_pipeline.py)
"""

from tpot import TPOTClassifier
from sklearn.datasets import load_digits

import pandas as pd 
from pandas import DataFrame

dframe = pd.read_excel("test.xlsx")
test = DataFrame(dframe)
test=test.T
X = test.iloc[ : , :-1].values 
Y = test.iloc[ : , 302].values 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=30)

"""
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)
"""


tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
