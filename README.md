# ML_Project

> 机器学习应用于医学物理方向 

> 项目进程每日记录

# Day1 --论文查找阅读

*  美国医学物理年会AAPM---https://aapm.org/pubs/default.asp

*  NIPS---------[Advances in Neural Information Processing Systems 30 ](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-30-2017)


*  [medical image analysis---MICCAI顶会](https://www.miccai.org/ConferenceTopics)


--------------------

* [医学图像分析深度学习方法研究与挑战](http://html.rhhz.net/ZDHXBZWB/html/2018-3-401.htm)-----综述文章


* [awesome-gan-for-medical-imaging](https://github.com/xinario/awesome-gan-for-medical-imaging)


* [Deep Learning Papers on Medical Image Analysis](https://github.com/albarqouni/Deep-Learning-for-Medical-Applications)


* 具体一个工作----[CAM](https://github.com/metalbubble/CAM)

> CVPR‘16上发表的一个工作：CNN Discriminative Localization and Saliency，代码也在这里metalbubble/CAM。这个工作提出了一个叫CAM (Class Activation Mapping)的方法，可以给任意一个CNN分类网络生成热力图（Class Activation Map），这个热力图可以高亮出图片里面跟CNN最后预测结果最相关的区域，所以这个CAM能解释网络模型到底是基于图片里面哪些部分作为证据进行了这次预测。比如说如下图，我们在caltech256上fine-tune了网络，然后对于每张图生成跟这张图所属类别的Class Activation Map，然后就可以高亮出图片里面跟类别最相关的区域。这里在训练过程中，网络并没有接收到任何bounding box的标定。所以一个在medical image analysis上直接扩展是，CNN分类网络预测出这张图里面有很大概率有cancer，这个高亮出来的区域很可能就是cancer区域，


