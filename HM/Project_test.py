
# coding: utf-8

# 基于keras 的三层神经网络
# 输入为 100*302 的矩阵  ： 100组数据，每组302个变量 [数据link](https://github.com/LiuChuang0059/ML_Project/blob/master/Data/test.xlsx)
# Adam加速
# 目前准确率 为 98%



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
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=30)





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

model.fit(x_train, y_train,
          epochs=500,
          batch_size=20)
score = model.evaluate(x_test, y_test, batch_size=10)
print(model.metrics_names)
print(score)

