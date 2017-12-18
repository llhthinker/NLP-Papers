# 论文阅读：Neural machine translation in linear time

## Pre: Receptive field

*   [ref: 深度神经网络中的感受野 (Receptive Field)](https://zhuanlan.zhihu.com/p/28492837)
*   感受野（Receptive Field）的定义：卷积神经网络的每一层输出的特征图上的像素点在**原**图像上映射的区域大小。

![](http://pic3.zhimg.com/v2-5378f1dfba3e73dedafdc879bbc4c71e_b.png)

*   感受野计算公式：
    *   $r_n = r_{n-1} + (k_n - 1) \prod_{i=1}^{n-1}s$
    *   Conv2的感受野边长 = Conv1的感受野边长 + (卷积核边长 - 1) * 步长累乘
    *   5 = 3 + (2 - 1) * (2 * 1)

## Pre: Dilated convolution

![](https://pic1.zhimg.com/50/v2-3cd4e5ebcae5fa15019c9f4df03bc734_hd.jpg)

*   将“隔若干个像素”的像素点作为输入张量。
*   我自己猜的公式，暂时没有找到来源：$r_n = r_{n-1} + (k_n - 1)  \cdot \prod_{i=1}^{n-1}s \cdot dilate$
    *   对应图二：3 + [(3 -1) * 1 * 1] * 2 = 7
    *   对应图三：7 + [(3 - 1) * 1 * 1 * 1] * 4 = 15



## Introduction

* RNN has potential drawback.
  * serial structure prevents from being run in parallel.
  * Forward and backward signals need traverse the full distance to learn the dependencies between tokens.
* A number of NMT models
  * Encoder-decoder network, attentional pooling, 2-dimensional networks, etc
  * Despite the generally good performance, they:
    * have running time that is **super-linear** in the length of source/target sequences.
    * process the src seq into a const size representation, **burdening the model with a memorization step**.
* **ByteNet**
  * uses **one-dimensional CNN** of **fixed depth** for **both the encoder and the decoder**.
  * The two CNNs use **increasing factors of dilation** to
    rapidly grow the receptive fields

## Neural Translation Model

* Desiderata
  * First, the **running time** of the network should be **linear** in
    the length of the source and target strings.
  * Second, the **size of the source representation** should be **linear** in the length of the source string, i.e. it should be resolution preserving, and **not have constant size**.
  * Third, the **path traversed by forward and backward signals** in the network (between input and ouput tokens) **should be short**. 

## ByteNet

* **Encoder-Decoder Stacking**

  ![](https://pic1.zhimg.com/v2-0d72713562d4015e420f159e703958a8_b.png)

  * 相比其他encode-decode网络的区别在于，不再仅仅通过一个定长的encode vector或者attention vector连接两个网络，而是将source network每一个输出都作为target vector对应位置上的输入。
  * 有点像是dilate=1,2,4,8...后又来了一次dilate=1,2,4,8,...
  * 但是输入和输出又不一样长，怎么直接Stack呢？

* **Dynamic Unfolding**

  * source length $|s|$, target length $|t|$
  * sufficiently tight upper bound $|\hat{t}|=a|s|+b$
  * 德语句子一般比英语长，设a=1.2，b=0 （这些都不重要。。）
  * ![](./dynamic_unfolding.png)
  * 令encoder和decoder这两个一维卷积长度都为$|\hat{t}|$（随着不同source句子长度变化，不固定），source端不足的就补，而target端最终预测的长度不固定，以遇到EOS为准。
    * 如果遇到EOS时的长度<$|\hat{t}|$， 则同时使用encoder的representations和decoder前面时刻的输出。
    * 如果直到超出$|\hat{t}|$都没预测完，那就只使用decoder前面时刻的输出。

* Masked One-dimensional Convolutions

  * 就是保证decoder只和历史相关，不和未来相关。所以decoder在每一个输出时刻只有左边一半。
  * dynamic unfolding是从左往右展开的，右边的不知道，显然没法用。

* **Dilation**

  ![](https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig2-Anim-160908-r01.gif)

  ​

* **Residual Blocks 残差网络**

  * Each layer is wrapped in a residual block that contains
    additional convolutional layers with filters of size 1 × 1

## Model Comparison

![](https://pic2.zhimg.com/v2-e17c18d8a0f70fa5ac08e76eeb4a311d_b.png)

* Time代表时间复杂度，RP代表resolution preserving，Paths代表source网络从输入到输出的路径长度，Patht代表target网络从输入到输出的路径长度。Path越短代表反向传播的层数越少，网络越容易收敛，因为网络越浅越不容易出现梯度扩散。
* ByteNet可以用RNN，也可以用CNN来实现。本文是用CNN实现的，但是也分析了一下RNN的复杂度。
* 其实之前Seq2seq的模型也是线性复杂度的，ByteNet比它好在：
  * Resolution preserving。不再是把encoder representations塞到一个固定长的向量中。
  * Path变短，为常数$c=log d$，d是在翻译中需要走过的依赖长度。
* ByteNet比attention模型好在：
  * 时间复杂度从平方复杂度变成了线性复杂度。
  * Path变短为常数。

## Experiment

* 语言模型任务（字符级）
  * ![](https://pic4.zhimg.com/v2-7942cb10517176e4d562045f98bb4f13_b.png)
  * 达到了SOTA
* 机器翻译任务（字符级）
  * ![](https://pic1.zhimg.com/v2-e81bdedfc7c6e9c6641325d8a0fadfdc_b.png)

## 相关阅读

[ref: 《Neural Machine Translation in Linear Time》阅读笔记](https://zhuanlan.zhihu.com/p/23795111)

