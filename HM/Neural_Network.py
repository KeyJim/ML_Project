#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:36:43 2018

@author: liuchuang
"""

import numpy as np
import matplotlib.pyplot as plt


N = 100 # number of points per class  每一类的数据数目
D = 2 # dimensionality   数据纬度 
K = 3 # number of classes  数据类别数目
X = np.zeros((N*K,D)) # data matrix (each row = single example) 生成一个N*K行 D列的全0矩阵
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius  每一类N个点之间的距离--均匀
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta 点的角度 在一定的范围内均匀基础上 随机取点
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]  #  确定点的坐标
  y[ix] = j # 0-N是 全是0 ； N-2N是1 ； 2N-3N 是2；
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral) # 画 散点图， X的第一列是横坐标，第二列是纵坐标，y颜色序列选择
# 注意 y不是yellow  而是一个列表
plt.show()


## 训练一个线性分类器
#Train a Linear Classifier
# initialize parameters randomly 初始化权重和 偏置
W = 0.01 * np.random.randn(D,K)  # 权重是 2x3 的矩阵
b = np.zeros((1,K))    # 偏置是 1 X k 的矩阵

# some hyperparameters
step_size = 1e-1   #  步长
reg = 1e-3 # regularization strength 正则化参数 

# gradient descent loop
num_examples = X.shape[0]  ## 表示 X的行数  即是 N*K 所有样本点数
for i in range(200):  # 循环200次
  
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b   #  矩阵乘后 N*K行 K列 ，加b 的时候 用到numpy  的广播性质
  
  # compute the class probabilities
  exp_scores = np.exp(scores)  # 未进行归一化的 概率
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K，k]  
  # 概率进行归一化 [300 ,3]的规模  每一行代表了 一个点对应三种分类的可能性
  
  # compute the loss: average cross-entropy loss and regularization
  #  按照公式 计算样本损失 和正则化损失
  correct_logprobs = -np.log(probs[range(num_examples),y])  # 遍历 所有的概率值
  data_loss = np.sum(correct_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  
  if i % 10 == 0:    # 每10次 打印一下 loss 报告
    print("iteration %d: loss %f" %(i, loss))
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1  #  下降梯度 就是 对应的概率 -1
  dscores /= num_examples  #  计算平均概率值
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient
  
  # perform a parameter update
  W += -step_size * dW   #  梯度更新
  b += -step_size * db
  

# evaluate training set accuracy
# 计算分类的正确率
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

# plot the resulting classifier
# 绘制 分类结果
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))  # 从坐标向量返回坐标矩阵
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8) # 绘制背分类边界
plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap=plt.cm.Spectral) # 绘制散点图
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())





























