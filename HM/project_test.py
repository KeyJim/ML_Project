#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:41:45 2018

@author: liuchuang

使用神经网络进行处理数据

"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame


dframe = pd.read_excel("test.xlsx")
test = DataFrame(dframe)
test=test.T
X = test.iloc[ : , :-1].values 
Y = test.iloc[ : , 302].values 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=30)


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()

model.add(Dense(100, activation='relu', input_dim=302))
model.add(Dense(30, activation='relu'))

model.add(Dense(5, activation='softmax'))

rms = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])

model.fit(x_train, y_train,validation_split=0.1,
          epochs=300,
          batch_size=10)
score = model.evaluate(x_test, y_test, batch_size=10)
print(model.metrics_names)
print(score)