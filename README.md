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


![](https://github.com/LiuChuang0059/ML_Project/blob/master/Picture/Keras_Test_result.png)

### pS： 待解决问题： 部署gpu



---------

---------

# Day4 ----- 神经网络学习

### 入门了解

* youtube---3blue1brown---https://www.youtube.com/watch?v=aircAruvnKk&t=710s

* b站----3blue1brown---https://www.bilibili.com/video/av15532370
















