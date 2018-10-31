# ML_Project

> 机器学习应用于医学物理方向 

> 项目进程每日记录

 * [ML_Project](#ml_project)
   * [Day1 --论文查找阅读](#day1---论文查找阅读)
      * [会议期刊](#会议期刊)
      * [综述文章](#综述文章)
      * [papers汇总](#papers汇总)
      * [具体意向项目](#具体意向项目)
   * [Day2 ----- 基本模型](#day2-------基本模型)
      * [医学图像的分析任务](#医学图像的分析任务)
      * [深度学习模型](#深度学习模型)
         * [1.无监督学习模型](#1无监督学习模型)
            * [1 SAE-- AE组成的多层神经网络](#1-sae---ae组成的多层神经网络)
            * [2 受限玻尔兹曼机RBM](#2-受限玻尔兹曼机rbm)
         * [2.监督学习模型](#2监督学习模型)
            * [1 CNN--卷积神经网络](#1-cnn--卷积神经网络)
            * [2RNN---具有反馈连接的循环神经网络](#2rnn---具有反馈连接的循环神经网络)
       * [参考](#参考)
   * [Day3  ------ Tensorflow   keras 安装使用](#day3---------tensorflow--keras-安装使用)
      * [pS： 待解决问题： 部署gpu](#ps-待解决问题-部署gpu)
   * [Day4 ----- 神经网络学习](#day4-------神经网络学习)
      * [学习资料](#学习资料)
      * [Note](#note)
           * [1 MLP](#1-mlp)
   * [Day5 --- Neural_Network (python 训练实现一个神经网络)](#day5-----neural_network-python-训练实现一个神经网络)
   * [Day6 ----Using neural nets to recognize handwritten digits](#day6-----using-neural-nets-to-recognize-handwritten-digits)
      * [Perception --- artificial neuron](#perception-----artificial-neuron)
          * [define](#define)
          * [应用1 --- 据测（0 or 1）：](#应用1-----据测0-or-1)
          * [应用2 ---计算逻辑函数 ： AND OR 之类](#应用2----计算逻辑函数--and-or-之类)
      * [Sigmoid neurons](#sigmoid-neurons)
      * [The architecture of neural networks](#the-architecture-of-neural-networks)
      * [A simple network to classify handwritten digits](#a-simple-network-to-classify-handwritten-digits)
         * [To recognize individual digits we will use a three-layer neural network](#to-recognize-individual-digits-we-will-use-a-three-layer-neural-network)
         * [Binary representation](#binary-representation)
      * [Learning with gradient descent](#learning-with-gradient-descent)
      * [Implementing our network to classify digits](#implementing-our-network-to-classify-digits)
         * [1 pre](#1-pre)
         * [2 code](#2-code)
   * [Day 7---实验方向](#day-7---实验方向)
        * [1. AI读取脑波，重建人类思维-----<a href="https://yq.aliyun.com/articles/374287" rel="nofollow">https://yq.aliyun.com/articles/374287</a>](#1-ai读取脑波重建人类思维-----httpsyqaliyuncomarticles374287)
        * [2. GAN之根据文本描述生成图像----<a href="https://blog.csdn.net/stdcoutzyx/article/details/78575240" rel="nofollow">https://blog.csdn.net/stdcoutzyx/article/details/78575240</a>](#2-gan之根据文本描述生成图像----httpsblogcsdnnetstdcoutzyxarticledetails78575240)



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


* youtube---[3blue1brown](https://www.youtube.com/watch?v=aircAruvnKk&t=710s)

* b站----[3blue1brown](https://www.bilibili.com/video/av15532370)

* YouTube---[welch Labs](https://www.youtube.com/watch?v=bxe2T-V8XRs)

* 网易云课堂--- [斯坦福cs231n](https://study.163.com/course/courseLearn.htm?courseId=1003223001#/learn/video?lessonId=1003978240&courseId=1003223001)

------

* [mlp详解](http://neuralnetworksanddeeplearning.com/chap1.html)


* [cnn教程](http://nooverfit.com/wp/pycon-2016-tensorflow-%E7%A0%94%E8%AE%A8%E4%BC%9A%E6%80%BB%E7%BB%93-tensorflow-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%85%A5%E9%97%A8-%E7%94%A8%E4%BA%BA%E8%AF%9D%E8%A7%A3%E9%87%8Acnn-%E7%AC%AC%E4%B8%89/)


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

# Day5 --- Neural_Network (python 训练实现一个神经网络)

[完整实现过程](https://github.com/LiuChuang0059/ML_Project/blob/master/code/Neural_Network_Demystified.ipynb)


## 资料参考
* 参考1 ---YouTube---[Welch labs](https://www.youtube.com/watch?v=S4ZUwgesjS8)

* 参考2 ---[参考一对应GitHub代码](https://www.youtube.com/redirect?v=S4ZUwgesjS8&event=video_description&redir_token=ipOr1avQb7xvqnG4wO3Sft2z_VB8MTUzNzA4NzAxM0AxNTM3MDAwNjEz&q=https%3A%2F%2Fgithub.com%2Fstephencwelch%2FNeural-Networks-Demystified)


* 待解决---还有些问题没有完全解决（train那里优化梯度下降 + 分割训练集和测试集后的处理）


-------
------
# Day6 ----Using neural nets to recognize handwritten digits

* 参考---[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)  By Michael Nielse  

## Perception --- artificial neuron


#### define
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/perception.png" width="400"/> </div><br>


* 给对应值一个权重以及一个 阈值   ； 大于阈值表示为1，小于阈值表示为0.

* a many-layer network of perceptrons can engage in sophisticated decision making.

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/perception_complax.png" width="400"/> </div><br>
**类似于量子力学里面的一维无限深势井 + 迪立科立函数**

#### 应用1 --- 据测（0 or 1）：
> Suppose the weekend is coming up, and you've heard that there's going to be a cheese festival in your city. You like cheese, and are trying to decide whether or not to go to the festival. You might make your decision by weighing up three factors:

> For instance, we'd have x1=1 if the weather is good, and x1=0 if the weather is bad. Similarly, x2=1 if your boyfriend or girlfriend wants to go, and x2=0 if not. And similarly again for x3 and public transit.

根据各个因素影响程度选定权重系数weight


#### 应用2 ---计算逻辑函数 ： AND OR 之类


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/NAND_gate.png" width="400"/> </div><br>

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/NAND_adder.png" width="400"/> </div><br>

----------

## Sigmoid neurons

* similar with Perception

* modified so that small changes in their weights and bias cause only a small change in their output


--------

## The architecture of neural networks

* hidden layer --- it really means nothing more than "not an input or an output"

* feedforward neural network---where the output from one layer is used as input to the next layer



<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/NN_structure.png" width="400"/> </div><br>

* recurrent neural networks----待解决

---------

## A simple network to classify handwritten digits


### To recognize individual digits we will use a three-layer neural network

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Three_layer_NN.png" width="400"/> </div><br>

* The input layer of the network contains neurons encoding the values of the input pixels----784 neurons

* The second layer of the network is a hidden layer. We denote the number of neurons in this hidden layer by n

* The output layer of the network contains 10 neurons.

>  为什么使用10个神经元最后一层，而不是4个，（因为 4个的话 每个代表0或者1； 可以有16种结果）

---------

* 假设隐藏神经元检测是否有下图的样子

* 通过增加该部分的权重，减小其他部分的权重

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/hidden_layer.png" width="100"/> </div><br>

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/hidden_layer2.png" width="200"/> </div><br>

* 据此我们可以基本上推断出 这个image 是 0

> 解释为什么不用4: 基于这种情况： There's no easy way to relate that most significant bit to simple shapes like those shown above.
很难把最高有效位和图像的形状联系起来


### Binary representation

* The extra layer converts the output from the previous layer into a binary representation

* the first 3 layers of neurons are such that the correct output in the third layer (i.e., the old output layer) has activation at least 0.99, and incorrect outputs have activation less than 0.01.

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/binary_out.png" width="400"/> </div><br>

-------------

## Learning with gradient descent

 * 1.数据集---[MNIST data set](http://yann.lecun.com/exdb/mnist/)
 
 * 2.cost函数--- 二次平滑函数，便于求导  +  进行修改改善，根据经验二次函数更好
 
 * 3.梯度下降
 
 * 4.随机梯度下降
 
 > estimate the gradient ∇C by computing ∇Cx for a small sample of randomly chosen training inputs. 
 
   * 随机选取m个作为mini-batch,m足够大，则m的梯度平均值等价于整体n的梯度平均值
   
 * 5. epoch 
 
 >  选取m个计算，之后再选取一个minibatch --直到训练集的数据全部用完
 
 >  使用mini-batch ：梯度估计可能会不准确，会有一些统计波动，但是不影响，因为我们不需要准确的梯度下降，需要大致的下降方向和数值。
 
 <div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Grandient_decend.png" width="400"/> </div><br>
 
 
 ----------
 
 ## Implementing our network to classify digits
 
 
 ### 1 pre
 * the MNIST data  was split into 60,000 training images, and 10,000 test images
 * 60,000 training images = 50,000(train) + 10,000(validation)
 
 ### 2 code 
 
 
 ---------
 
 # Day 7---实验方向
 
###  1. AI读取脑波，重建人类思维-----[paper-link](https://yq.aliyun.com/articles/374287)

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/%E8%84%91%E7%94%B5%E6%B3%A2%E7%94%9F%E6%88%90%E5%9B%BE%E5%83%8F.jpeg" width="500"/> </div><br>


* 模糊
* 

###  2. GAN之根据文本描述生成图像----[paper-link](https://blog.csdn.net/stdcoutzyx/article/details/78575240)

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/gan%E7%94%9F%E6%88%90%E7%BD%91%E7%BB%9C.png" width="500"/> </div><br>


* 提取波的特征-----随机向量---> 图像的对应


###  3.CNN 解决

* 将波形图的数据对应于向量，多个想向量拼接成矩阵---对应于图片的像素矩阵

* 使用cnn提取 波形图--> 图片  + 生成图片的特征

* 特征比对 ，卷积降维。



# Day8 ---CNN 学习（一）

## 学习资料

* cs231课程-cnn---[网易云课堂中文字幕](https://study.163.com/course/courseLearn.htm?courseId=1003223001#/learn/video?lessonId=1004009214&courseId=1003223001)

* [cs231课堂笔记中文翻译](https://zhuanlan.zhihu.com/p/22038289?refer=intelligentunit)

> 很详尽
* [cs231 课堂笔记--官网英文原版](http://cs231n.github.io/convolutional-networks/)
> 动图展示很直观

## CNN结构概述
* 常规的神经网络：
  * 每个神经元与前一层的所有神经元连接，同一个隐层的神经元相互独立，---全连接层 
  * 常规神经网络在大尺寸图像上 参数过多，效率低下，可能导致过拟合

* 卷积神经网络：
  * 三维排列
  * 神经元只对应前一层的一小块区域连接----不是全连接

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/MLP%E5%92%8CCNN%E6%AF%94%E8%BE%83.jpg" width="400"/> </div><br>



## 构建CNN的各种层

### 卷积层
* 卷积层由一些可以学习的滤波器集合构成
  * 滤波器的宽度和高度都比较小，深度和输入数据一致。----滤波器的尺寸选取为什么是奇数
  * 滤波器在输入数据的宽度和高度上滑动---产生一个二维的激活图
  * 将所有的激活映射在深度方向上层叠起来生成了输出数据

* 局部连接
  * 神经元只和输入数据的一个局部区域连接---receptive field
  * 参数数量： [5x5x3]---5x5x3+1(偏置参数)=76
  
  <div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/%E5%8D%B7%E7%A7%AF%E5%B1%82%E5%B1%80%E9%83%A8%E8%BF%9E%E6%8E%A5.jpg" width="400"/> </div><br>


* 空间排列---输出数据体的神经元数量

  * 深度： 深度和使用的滤波器的数量一致；沿着深度方向排列、感受野相同的神经元集合称为深度列
  
  * 步长（stride）： 当步长为1，滤波器每次移动1个像素。当步长为2（或者不常用的3，或者更多，这些在实际中很少使用），滤波器滑动时每次移动2个像素
    
    > （N-F+2P）/s +1  一定要是整数
  
  * 零填充（zero-padding）：控制输出的数据体的空间尺寸： 补充（F-1）/2 --F：尺寸 可以保持原图的大小
   
    > 为什么使用0进行补充，因为0 对卷积输出没贡献,不填充，边缘信息损失


* 参数共享
**每个切片只有一个权重集---每个深度切片中的神经元使用同样的权重和参数**

  * 如果在一个深度切片中的所有权重都使用同一个权重向量，那么卷积层的前向传播在每个深度切片中可以看做是在计算神经元权重和输入数据体的卷积（

> 为什么取同样的参数：
>>> 作一个合理的假设：如果一个特征在计算某个空间位置(x,y)的时候有用，那么它在计算另一个不同位置(x2,y2)的时候也有用。基于这个假设，可以显著地减少参数数量
 
 
**有时候参数共享假设可能没有意义，特别是当卷积神经网络的输入图像是一些明确的中心结构时候。这时候我们就应该期望在图片的不同位置学习到完全不同的特征。一个具体的例子就是输入图像是人脸，人脸一般都处于图片中心。你可能期望不同的特征，比如眼睛特征或者头发特征可能（也应该）会在图片的不同位置被学习。在这个例子中，通常就放松参数共享的限制，将层称为局部连接层**

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/%E5%8D%B7%E7%A7%AF%E8%BF%90%E7%AE%97%E4%BE%8B%E5%AD%90.png" width="400"/> </div><br>



**卷积层运算演示**
[动画演示link](http://cs231n.github.io/convolutional-networks/)
  
  
  
### 汇聚层
* 最大汇聚---较好
* 平均汇聚
* 范式汇聚

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/%E6%B1%87%E8%81%9A%E5%B1%82.jpg" width="400"/> </div><br>


> 不使用汇聚层：很多人不喜欢汇聚操作，认为可以不使用它。比如在Striving for Simplicity: The All Convolutional Net一文中，提出使用一种只有重复的卷积层组成的结构，抛弃汇聚层。通过在卷积层中使用更大的步长来降低数据体的尺寸。有发现认为，在训练一个良好的生成模型时，弃用汇聚层也是很重要的。比如变化自编码器（VAEs：variational autoencoders）和生成性对抗网络（GANs：generative adversarial networks）。现在看起来，未来的卷积网络结构中，可能会很少使用甚至不使用汇聚层


### 全连接层转化为卷积层

> 全连接层转化为卷积层：在两种变换中，将全连接层转化为卷积层在实际运用中更加有用。假设一个卷积神经网络的输入是224x224x3的图像，一系列的卷积层和汇聚层将图像数据变为尺寸为7x7x512的激活数据体（在AlexNet中就是这样，通过使用5个汇聚层来对输入数据进行空间上的降采样，每次尺寸下降一半，所以最终空间尺寸为224/2/2/2/2/2=7）。从这里可以看到，AlexNet使用了两个尺寸为4096的全连接层，最后一个有1000个神经元的全连接层用于计算分类评分。我们可以将这3个全连接层中的任意一个转化为卷积层：

> 针对第一个连接区域是[7x7x512]的全连接层，令其滤波器尺寸为F=7，这样输出数据体就为[1x1x4096]了。
针对第二个全连接层，令其滤波器尺寸为F=1，这样输出数据体为[1x1x4096]。
对最后一个全连接层也做类似的，令其F=1，最终输出为[1x1x1000]
实际操作中，每次这样的变换都需要把全连接层的权重W重塑成卷积层的滤波器。那么这样的转化有什么作用呢？它在下面的情况下可以更高效：让卷积网络在一张更大的输入图片上滑动（即把一张更大的图片的不同区域都分别带入到卷积网络，得到每个区域的得分），得到多个输出，这样的转化可以让我们在单个向前传播的过程中完成上述的操作。

> 举个例子，如果我们想让224x224尺寸的浮窗，以步长为32在384x384的图片上滑动，把每个经停的位置都带入卷积网络，最后得到6x6个位置的类别得分。上述的把全连接层转换成卷积层的做法会更简便。如果224x224的输入图片经过卷积层和汇聚层之后得到了[7x7x512]的数组，那么，384x384的大图片直接经过同样的卷积层和汇聚层之后会得到[12x12x512]的数组（因为途径5个汇聚层，尺寸变为384/2/2/2/2/2 = 12）。然后再经过上面由3个全连接层转化得到的3个卷积层，最终得到[6x6x1000]的输出（因为(12 - 7)/1 + 1 = 6）。这个结果正是浮窗在原图经停的6x6个位置的得分！

> 面对384x384的图像，让（含全连接层）的初始卷积神经网络以32像素的步长独立对图像中的224x224块进行多次评价，其效果和使用把全连接层变换为卷积层后的卷积神经网络进行一次前向传播是一样的。
自然，相较于使用被转化前的原始卷积神经网络对所有36个位置进行迭代计算，使用转化后的卷积神经网络进行一次前向传播计算要高效得多，因为36次计算都在共享计算资源。这一技巧在实践中经常使用，一次来获得更好的结果。比如，通常将一张图像尺寸变得更大，然后使用变换后的卷积神经网络来对空间上很多不同位置进行评价得到分类评分，然后在求这些分值的平均值。

> 最后，如果我们想用步长小于32的浮窗怎么办？用多次的向前传播就可以解决。比如我们想用步长为16的浮窗。那么先使用原图在转化后的卷积网络执行向前传播，然后分别沿宽度，沿高度，最后同时沿宽度和高度，把原始图片分别平移16个像素，然后把这些平移之后的图分别带入卷积网络。



## 层的尺寸设计

* 输入层（包含图像的）应该能被2整除很多次

* 卷积层应该使用小尺寸滤波器（比如3x3或最多5x5），使用步长S=1

> 因为内存限制所做的妥协：在某些案例（尤其是早期的卷积神经网络结构）中，基于前面的各种规则，内存的使用量迅速飙升。例如，使用64个尺寸为3x3的滤波器对224x224x3的图像进行卷积，零填充为1，得到的激活数据体尺寸是[224x224x64]。这个数量就是一千万的激活数据，或者就是72MB的内存（每张图就是这么多，激活函数和梯度都是）。因为GPU通常因为内存导致性能瓶颈，所以做出一些妥协是必须的。在实践中，人们倾向于在网络的第一个卷积层做出妥协。例如，可以妥协可能是在第一个卷积层使用步长为2，尺寸为7x7的滤波器（比如在ZFnet中）。在AlexNet中，滤波器的尺寸的11x11，步长为4。



-------
-------

# Day9 --- CNN学习（二）

## 学习资料

* [Neural Networks and Deep Learning---书籍网页](http://neuralnetworksanddeeplearning.com/index.html)

* [《神经网络和深度学习》系列文章四十四：介绍卷积网络](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650790918&idx=1&sn=4fad7df685991d3dbbd2469fe8c5aedf&scene=21#wechat_redirect)

* [《神经网络和深度学习》系列文章四十五：卷积神经网络在实际中的应用](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650790932&idx=1&sn=10c8c5ee729d1fa200d2dd189f5edc38&chksm=8f4749ffb830c0e9a0efb8a6f944ba49c3419c995d8a07885f30afd3a0743040b266b5751255&scene=21#wechat_redirect)

* [keras 手把手入门#1-MNIST手写数字识别](http://nooverfit.com/wp/keras-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%85%A5%E9%97%A81-%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98/)

* [keras 中文文档](https://keras.io/zh/getting-started/sequential-model-guide/)

* [keras学习之五：Keras中的神经网络层组件简介](https://blog.csdn.net/zzulp/article/details/76590712)

## 共享权重和偏置


* 第一个隐藏层的所有神经元检测完全相同的特征
> 在一个特定的局部感受野的垂直边缘。这种能力在图像的其它位置也很可能是有用的

>>> 用稍微更抽象的术语，卷积网络能很好地适应图像的平移不变性：例如稍稍移动一幅猫的图像，它仍然是一幅猫的图像

* 优点： 大大减少了卷积网络的参数


## coding
* [cnn实现手写识别code+网络层组件学习](https://github.com/LiuChuang0059/ML_Project/blob/master/code/mnist_cnn_keras.ipynb)

> 其中dropout避免过拟合---http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf


-----------
-------------


# Day 10 ---CNN学习（三）

## 学习资料

* [手写数字识别--卷积神经网络CNN原理详解](https://zhuanlan.zhihu.com/p/30665319)---by Charlotte

* [如何用卷积神经网络CNN识别手写数字集？](https://www.cnblogs.com/charlotte77/p/5671136.html)----by Charlotte


## 问题详解
### 对于分类 ，相较于机器学习算法，为什么使用神经网络

* 特征提取高效
  > 特征数目过少，无法精确的分类出来---欠拟合，如果特征数目过多，可能会导致我们在分类过程中过于注重某个特征导致分类错误，即过拟合。
  > 然而神经网络的出现使我们不需要做大量的特征工程，


* 数据简介
  > 需要对数据进行一些处理，譬如量纲的归一化，格式的转化
  > 神经网络不需要过多的处理
  
*  参数数量少


### 卷积层

* 图像本身具有“二维空间特征”，通俗点说就是局部特性。譬如我们看一张猫的图片，可能看到猫的眼镜或者嘴巴就知道这是张猫片，而不需要说每个部分都看完了才知道，啊，原来这个是猫啊。所以如果我们可以用某种方式对一张图片的某个典型特征识别，那么这张图片的类别也就知道了

* filter

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/%E5%8D%B7%E7%A7%AF%E6%93%8D%E4%BD%9C%E5%9B%BE%E8%A7%A3.jpg" width="500"/> </div><br>

> 通过第一个卷积核计算后的feature_map是一个三维数据，在第三列的绝对值最大，说明原始图片上对应的地方有一条垂直方向的特征，即像素数值变化较大；而通过第二个卷积核计算后，第三列的数值为0，第二行的数值绝对值最大，说明原始图片上对应的地方有一条水平方向的特征。

* 对于非正方形的卷积核，需要保证层的输出形状是整数

* 卷积核的个数如何确定
> 由经验确定。通常情况下，靠近输入的卷积层，譬如第一层卷积层，会找出一些共性的特征，如手写数字识别中第一层我们设定卷积核个数为5个，一般是找出诸如"横线"、“竖线”、“斜线”等共性特征，我们称之为basic feature，经过max pooling后，在第二层卷积层，设定卷积核个数为20个，可以找出一些相对复杂的特征，如“横折”、“左半圆”、“右半圆”等特征，越往后，卷积核设定的数目越多，越能体现label的特征就越细致，就越容易分类出来


### MAX pooling

* 进行Max Pooling操作后，提取出的是真正能够识别特征的数值，其余被舍弃的数值，对于我提取特定的特征并没有特别大的帮助

* 减小了feature map的尺寸，从而减少参数，达到减小计算量，缺不损失效果的情况。

* 把卷积后不加Max Pooling的结果与卷积后加了Max Pooling的结果输出对比一下，看看Max Pooling是否对卷积核提取特征起了反效果。

### Flatten 层


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/flatten%E5%B1%82%E8%BF%87%E7%A8%8B.jpg" width="500"/> </div><br>



-----------
------------

# Day 11  --- 神经网络可以计算所有的函数

### Theorems
> A neural network can compute any function
  
  * we can get an approximation .By increasing the number of hidden neurons, we can improve the approximation. We can set a desired accuracy.
  
  * continuous functions better .

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Netural-network.png" width="400"/> </div><br>

*  even function has many inputs,  and many outputs

*  even a  single hidden layer


### 1 : One input and one output

* Single hidden layer : two neurons.  σ(wx+b), where σ(z)≡1/(1+e−z)

* Focus on the top neuron


* Since the w is large enough ,we can get the Step func 
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Step-func.jpg" width="400"/> </div><br>

* We can find  the step is at position s=−b/w
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/step_position.png" width="400"/> </div><br>

* Focus on entire network

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Entire%20network.png" width="400"/> </div><br>


* We can get a bump function ---  set the height :h
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/bump_function.jpg" width="400"/> </div><br>

* We can use our bump-making trick to get two bumps, by gluing two pairs of hidden neurons together into the same network:

* By changing the output weights we're actually designing the function

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/5bump_func.png" width="400"/> </div><br>

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/bump-func2.png" width="400"/> </div><br>

### 2 Many input values

* With w2=0 the input y makes no difference to the output from the neuron

*  The actual location of the step point is sx≡−b/w1.

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/2D-X.png" width="400"/> </div><br>

* we can get from X+Y
<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/2D-XY.png" width="400"/> </div><br>

----------------
----------------

# Day 12 ---  Improving the way neural networks learn


### 1. The cross-entropy cost function

> Artificial neuron has a lot of difficulty learning when it's badly wrong - far more difficulty than when it's just a little wrong

* neuron's output is close to 1, the curve gets very flat, ∂C/∂w and ∂C/∂b get very small. 

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Sigmoid-func.png" width="400"/> </div><br>

*  cross-entropy cost function

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Cross.png" width="400"/> </div><br>
  
  * the cross-entropy is positive
  
  * tends toward zero as the neuron gets better at computing the desired output
  
  * it avoids the problem of learning slowing down: rate at which the weight learns is controlled by σ(z)−y ---- the error in the output


<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Decent.png" width="400"/> </div><br>


### 2 Softmax
* If one activation increases, then the other output activations must decrease by the same total amount, to ensure the sum over all activations remains 1

* the output from the softmax layer can be thought of as a probability distribution.



<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/SoftMax.png" width="400"/> </div><br>


* log-likelihood cost function to solve learning slowdown problem



### 3 Overfitting and regularization   ----- Small

#### 1 Overfit 

* To detect overfitting : we can keep  track of accuracy on the test data as our network trains. 

* Better to test data and the training data both stop improving at the same time.

------
**Early Stopping**

* use the validation_data (different from train_data test_data) to prevenr overfitting

  > Compute the classification accuracy on the validation_data at the end of each epoch. Once the classification accuracy on the validation_data has saturated, we stop trainin

* Why not test_data: we may end up finding hyper-parameters which fit particular peculiarities of the test_data, but where the performance of the network won't generalize to other data sets

* Once we've got the hyper-parameters we want, we do a final evaluation of accuracy using the test_data


#### 2 Regularization---- reduce overfitting and to increase classification accuracies(weight decay)
> The idea of L2 regularization is to add an extra term to the cost function, a term called the regularization term.

<div align="center">  <img src="https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Regulation.png" width="400"/> </div><br>

* add the sum of the squares of all the weights in the network

  *  a way of compromising between finding small weights and minimizing the original cost function
  
  * when λ is small we prefer to minimize the original cost function, but when λ is large we prefer small weights.

------
**Why?**
> smaller weights are, in some sense, lower complexity, and so provide a simpler and more powerful explanation for the data, and should thus be preferred

* higher order really just learning the effects of local noise

*  The smallness of the weights means that the behaviour of the network won't change too much if we change a few random inputs here and there. That makes it difficult for a regularized network to learn the effects of local noise in the data

> sometimes the more complex explanation turns out to be correct.

#### 3 Other techniques for regularization

* L1 regularization


* Dropout




















 
