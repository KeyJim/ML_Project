# ML_Project

> 机器学习应用于医学物理方向 

> 项目进程每日记录

# Day1 --论文查找阅读

## 会议期刊

*  美国医学物理年会-----[AAPM](https://aapm.org/pubs/default.asp)

*  神经信息处理系统NIPS---------[Advances in Neural Information Processing Systems 30 ](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-30-2017)


*  medical image analysis-----[MICCAI顶会](https://www.miccai.org/ConferenceTopics)

*  国际机器学习顶会----[ICML](https://icml.cc/Conferences/2017/Tutorials)


--------------------

## 综述文章

* [医学图像分析深度学习方法研究与挑战](http://html.rhhz.net/ZDHXBZWB/html/2018-3-401.htm)----自动化报

* [医学图像分析_赵地](https://github.com/LiuChuang0059/ML_Project/blob/master/Paper/%E5%8C%BB%E5%AD%A6%E5%9B%BE%E5%83%8F%E5%88%86%E6%9E%90_%E8%B5%B5%E5%9C%B0.pdf)


-----------------------

## papers汇总

* [awesome-gan-for-medical-imaging](https://github.com/xinario/awesome-gan-for-medical-imaging)


* [Deep Learning Papers on Medical Image Analysis](https://github.com/albarqouni/Deep-Learning-for-Medical-Applications)


-----------------

## 具体意向项目


* [CAM](https://github.com/metalbubble/CAM)

> CVPR‘16上发表的一个工作：CNN Discriminative Localization and Saliency，代码也在这里metalbubble/CAM。这个工作提出了一个叫CAM (Class Activation Mapping)的方法，可以给任意一个CNN分类网络生成热力图（Class Activation Map），这个热力图可以高亮出图片里面跟CNN最后预测结果最相关的区域，所以这个CAM能解释网络模型到底是基于图片里面哪些部分作为证据进行了这次预测。比如说如下图，我们在caltech256上fine-tune了网络，然后对于每张图生成跟这张图所属类别的Class Activation Map，然后就可以高亮出图片里面跟类别最相关的区域。这里在训练过程中，网络并没有接收到任何bounding box的标定。所以一个在medical image analysis上直接扩展是，CNN分类网络预测出这张图里面有很大概率有cancer，这个高亮出来的区域很可能就是cancer区域，

-------------
------------


# Day2 ----- 基本模型

## 医学图像的分析任务

* 医学图像的分类和识别
> 自动的识别病灶区域和正常器官

* 医学图像的定位与检测
> 确定具体的物理位置

* 医学图像的分割任务
> 识别图像中感兴趣的目标区域(如肿瘤)内部体素及其外轮廓, 它是临床手术图像导航和图像引导肿瘤放疗的关键任务.

## 深度学习模型


### 1.无监督学习模型

#### 1 SAE-- AE组成的多层神经网络

AE： 自动编码机，
![](https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/%E8%87%AA%E5%8A%A8%E7%BC%96%E7%A0%81%E6%9C%BA.jpg)

$$
\begin{equation}
\begin{split}
J= &||{\pmb x}-{\pmb {x}'}||^2=\\
&||{\pmb x}-(W_{{\pmb h}, {\pmb x}'}(\sigma(W_{{\pmb x}, {\pmb
h}}{\pmb x}+{\pmb b}_{{\pmb x}, {\pmb h}}))+{\pmb b}_{{\pmb
h}, {\pmb x}'})||^2
\end{split}
\end{equation}
$$

> AE 通过最小化网络的输入和输出之间的重建误差学习输入数据的  潜在特征或者压缩表示


* 单层的AE是简单的浅层结构，SAE是由多层的AE组成的神经网络，前一层的AE输出作为后一层的AE输入，层与层之间采用全联接形式


*  SAE中间各层添加稀疏性约束，可以构成栈式稀疏自编码器(Stacked sparsely autoencoder, SSAE), 使模型具有一定的抗噪能力, 且模型泛化性更好

* 当输入向量用SSAE表示时, 不同网络层表示不同层次的特征, 即网络的较低层表示简单的模式, 网络的较高层表示输入向量中更复杂抽象的本质模式.




#### 2 受限玻尔兹曼机RBM

![](https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/%E5%8F%97%E9%99%90%E7%8E%BB%E5%B0%94%E5%85%B9%E6%9B%BC%E6%9C%BA.jpg)

* RBM是一个可视层和一个隐层的无向图模型

* 假设可视层和隐层之间是对称连接的，但是内内结点之间不存在任何连接

* 给定输入向量, 可以得到潜在特征表示, 反之亦然.因此, RBM是一个生成模型,可以从训练数据分布中采样生成新数据

* 由于连接的对称性, 可从隐层表示生成输入观察, 因此, RBM本质上是一个AE.模型参数通过最大化观察与输入的相似性进行优化, 通常采用对比散度算法(Contrastive divergence, CD)训练


* DBN 和DBM深度特征学习网络： 

> DBN : 多个RBM堆叠起来，包含一个可视层vv和一系列隐层hh1,⋯,hhL, 靠近可视层的部分隐层使用贝叶斯置信网络, 形成有向生成模型, 这些层的概率分布计算不依赖于高层, 如hh1层仅依赖于可视层vv而不需考虑hh2层, 从而加快了计算速度.而最上面两层仍保持RBM无向生成模型的形式,

> DBM:中所有层保持无向生成模型的形式.如图 , DBM包含输入层和L个隐层, 且只有相邻层结点之间才有连接. DBM中间隐层的条件概率分布计算同时利用了其相邻两层信息,


* 贪婪层次学习：
> 每次只预训练一层网络, 即首先用训练样本数据输入训练第1隐层的参数, 然后用第1隐层的输出作为第2隐层的输入, 训练第2隐层的参数, 以此类推, 第l层的输出作为第l+1层的输入以训练第l+1层参数


**三种深度模型要求网络的输入通常为向量形式, 而对于医学图像, 像素或体素的邻域结构信息是一个重要的信息源, 向量化必然会破坏图像中的邻域结构关系.**



### 2.监督学习模型

#### 1 CNN--卷积神经网络

> 通常, CNN网络参数的训练算法与传统BP算法相似, 通过前向传播计算输出值, 然后将输出值与理想标签值的误差, 通过梯度下降法对最小化误差问题寻优, 再利用反向传播梯度调整CNN的参数.

![](https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/%E5%9F%BA%E4%BA%8ECNN%E7%9A%84%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E6%A1%86%E6%9E%B6.gif)

#### 2RNN---具有反馈连接的循环神经网络

> 其本质属性是网络的状态会随时间演化, 适用于提取数据的时序特征


# 参考
http://html.rhhz.net/ZDHXBZWB/html/2018-3-401.htm

-----------------
-----------------


# Day3  ------ Tensorflow + keras 安装使用

* tensorflow安装过程
https://github.com/LiuChuang0059/Techs/blob/master/tensorflow/README.md


* keras 安装以及 “hello world！”

[数字识别--官方网站](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)

[数字识别--代码详解](http://nooverfit.com/wp/keras-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%85%A5%E9%97%A81-%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98/)

[数字识别--过程可视化](http://nooverfit.com/wp/keras-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%85%A5%E9%97%A81-%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98/
)


<div align="center"> <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Keras_Test_result.png" width="500"/> </div><br>

### pS： 待解决问题： 部署gpu



---------

---------

# Day4 ----- 神经网络学习

## 学习资料


* youtube---3blue1brown---https://www.youtube.com/watch?v=aircAruvnKk&t=710s

* b站----3blue1brown---https://www.bilibili.com/video/av15532370

* YouTube---welch Labs---https://www.youtube.com/watch?v=bxe2T-V8XRs

* 网易云课堂--- 斯坦福cs231n---https://study.163.com/course/courseLearn.htm?courseId=1003223001#/learn/video?lessonId=1003978240&courseId=1003223001

------

* mlp----http://neuralnetworksanddeeplearning.com/chap1.html

* cnn教程---斯坦福CS231n---https://zhuanlan.zhihu.com/p/22038289?refer=intelligentunit  （中文翻译）+  http://cs231n.github.io/convolutional-networks/（英文原版）

* cnn教程-- ---http://nooverfit.com/wp/pycon-2016-tensorflow-%E7%A0%94%E8%AE%A8%E4%BC%9A%E6%80%BB%E7%BB%93-tensorflow-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%85%A5%E9%97%A8-%E7%94%A8%E4%BA%BA%E8%AF%9D%E8%A7%A3%E9%87%8Acnn-%E7%AC%AC%E4%B8%89/


## Note


#### 1 MLP
* example： 手写字符识别

* 像素：28x28 =784  对应784个灰度值不同的神经元


* 目标 通过多层网络（2层） 实现一个复杂函数从784 对应到10

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP33.09.png" width="400"/> </div><br>

* 实现需要 先识别 小的pattern，然后组合拼成大的pattern
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP24.14.png" width="400"/> </div><br>


* 层之间映射的函数
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP35.16.png" width="400"/> </div><br>


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP31.55.png" width="400"/> </div><br>

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP31.14.png" width="400"/> </div><br>


* 整个神经元对应的复杂函数需要确定的参数数量
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP29.21.png" width="500"/> </div><br>


-----

* 参数不可能全部手动配置，引入梯度下降算法。

  * 最开始随机初始化参数的数值
  
  * 定义误差函数
  <div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP37.15.png" width="500"/> </div><br>
  
  
  * 改变权重和偏置值 ---梯度下降法  --多维连续平滑函数----所以神经元的值选取连续值
  
  
* 整体识别过程

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP47.13.png" width="400"/> </div><br>



<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP47.36.png" width="400"/> </div><br>  
  
  


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP45.27.png" width="400"/> </div><br>  
  
  
  
* 实际上神网络并没有像我么预想的一层层分小块识别，而是如下图

* 所以与其说MLP使机器识别了图片，不如是 机器记住了图片----机器只关注cost函数的误差梯度下降

  
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP04.54.png" width="400"/> </div><br>  

--------

**BP算法--反向传播**----计算上一步的梯度下降


* 目标是不但要知道激活值应该变化的方向，还应该知道激活值变化的大小


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP40.00.png" width="400"/> </div><br>

* 如何改变激活值---3个参数


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP44.47.png" width="400"/> </div><br>

* 不但是对于目标数字的变化（变为1），还应该是所有其他的（变为0） 求和平均

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/raw/master/Picture/BP47.33.png" width="400"/> </div><br>


* 对数据集所有数据执行bp 对参数求平均确定最终数值



* 但是所有数据每一次都求，计算速度太慢，，可以使用随机梯度下降法---随机分成minibatch，计算每个minibatch的梯度，直到全部计算完成。
 
 
 ------------
 
 **BP算法的数学表达**----主要为链式法则理解

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP41.19.png" width="400"/> </div><br>


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP58.55.png" width="400"/> </div><br>

* 注意上一个激活值的微分有求和，因为会对每个下一层的激活值都有贡献
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/BP38.59.png" width="400"/> </div><br>


---------
---------

















