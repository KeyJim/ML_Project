# 项目

测试环境 ： Google Colaboratory  Gpu 



##  Model 1 ---[Project-1.ipynb](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/Project-1.ipynb)

*  train 数据--[110ad.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/110ad.xlsx)

*  test 数据---[test_1.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_1.xlsx)

* LSTM  回归预测，使用 fit函数

* 存在问题 ： y是多值输出，需要转换数据类型 ---》 list


------


## Model 2 -----[project_2.ipynb](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/project_2.ipynb)

* train 数据--[110ad-2.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/110ad-2.xlsx)

*  test 数据---
[test_1.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_1.xlsx)

[text_2.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/text_2.xlsx)

[test_3.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_3.xlsx)

* LSTM 回归 train_on_batch

### 1 存在问题

* 过拟合

* 前半部分拟合效果

* 训练速度 慢


### 2 训练结果

<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_result/LSTM/4001epoch-2layer-0.2dropout-50batchsize.png" width="600"/> </div><br>

-------


## Model 3 --- [Project-3]()

* 数据同 Model 2  --- 200组训练数据

* BiLSTM

<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Project/BiLSTM-model.png" width="800"/> </div><br>


### 2 训练结果




<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_result/BiLSTM/1500-50-1.png" width="600"/> </div><br>




<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_result/BiLSTM/3000-50-1.png" width="600"/> </div><br>

