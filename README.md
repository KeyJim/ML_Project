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


# Day2 

## 医学图像的分析任务

* 医学图像的分类和识别
> 自动的识别病灶区域和正常器官

* 医学图像的定位与检测
> 确定具体的物理位置

* 医学图像的分割任务
> 识别图像中感兴趣的目标区域(如肿瘤)内部体素及其外轮廓, 它是临床手术图像导航和图像引导肿瘤放疗的关键任务.

## 深度学习模型

#### 1 SAE--

AE： 自动编码机，
![]()

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

> AE 通过最小化网络的输入和输出之间的重建误差学习输入数据的潜在特征或者压缩表示












