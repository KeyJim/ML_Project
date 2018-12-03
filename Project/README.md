# Project

1. Train environment ： Google Colaboratory  Gpu 

2. Data set：

* train data--[110ad-2.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/110ad-2.xlsx)(200 x 302)

* test data---[test_1.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_1.xlsx) + [text_2.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/text_2.xlsx) + [test_3.xlsx](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_3.xlsx)


* Data preprocessing：

![](https://github.com/LiuChuang0059/large_file/blob/master/1.gif)


## Model 1 -----[project_2--code](https://github.com/LiuChuang0059/ML_Project/blob/master/Project/project_2.ipynb)

### 1. Theory 

**RNN + LSTM**


[RNN+LSTM Theory Detail](https://github.com/LiuChuang0059/ComplexNetwork-DataMining/blob/master/techs/RNN/RNN%E6%A6%82%E8%BF%B0.md#2-%E4%BB%8Ernn%E5%88%B0lstm)


### 2. Training result

<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_result/LSTM/LSTM_result.png" width="800"/> </div><br>


<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_result/LSTM/4001epoch-2layer-0.2dropout-50batchsize.png" width="600"/> </div><br>


### 3. Problems 


* Maybe  Overfitting

* The first half --- Poor fitting effect

------
-------


## Model 2 --- [Project_3--code]()

### 1. Theory 

**RNN + BiLSTM**

<div align="center"> <img src="https://github.com/LiuChuang0059/ComplexNetwork-DataMining/blob/master/techs/Image/%E5%8F%8C%E5%90%91RNN.png" width="800"/> </div><br>


### 2. Training results

<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Project/BiLSTM-model.png" width="800"/> </div><br>


<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_result/BiLSTM/1500-50-1.png" width="600"/> </div><br>



<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Project/test_result/BiLSTM/3000-50-1.png" width="600"/> </div><br>


--------
-------

## 3. Future work

**Seq2Seq** --- [Seq2Seq Theory Detail](https://github.com/LiuChuang0059/ComplexNetwork-DataMining/blob/master/techs/RNN/RNN--Seq2Seq.md)

> The main idea behind this is that it contains an encoder RNN (LSTM) and a decoder rnn. One to ‘understand’ the input sequence and the decoder to ‘decode’ the ‘thought vector’ and construct an output sequence.

🌟 Combine CNN and RNN ;Encodind image into vector




--------

**TCN** --- [TCN theory Detail](https://github.com/LiuChuang0059/ComplexNetwork-DataMining/blob/master/techs/TCN.md#tcn-%E7%BB%93%E6%9E%84)




🌟 Keep all historical information

<div align="center"> <img src="https://github.com/LiuChuang0059/ComplexNetwork-DataMining/blob/master/techs/Image/TCN%E7%BB%93%E6%9E%84.png" width="800"/> </div><br>


