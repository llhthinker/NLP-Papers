## 论文笔记：Neural Machine Translation via Binary Code Prediction

## Intro

* Complexity of the output layer of the Encoder-decoder-attention NMT model: $O(V)$
* What we desire:
  * Memory efficiency
  * Time efficiency (for CPU under actual production envir)
  * Compatibility with parallel computation (for GPU under training)
* In this paper, we propose a method that satisfies all of conditions above.
  * The method works by not predicting a softmax over the entire output vocabulary, 
  * but instead by encoding each vocabulary word as a vector of binary variables, then independently predicting the bits of this binary representation. 
  * $O(V) \rightarrow (log V)$

## Prior Work on Suppressing Complexity of NMT model

* 对softmax的加速问题提出了使用近似计算的方法，近似计算的主要方法如下：

  a) 基于采样的近似方法：重要性采样；噪声对比估计方法；简单负采样；mini-batching负采样

  b) 基于结构的近似方法：基于分类的softmax；基于层次结构的softmax；二进制编码预测

* Hierarchical softmax (Morin and Bengio, 2005)

  * Time Complexity $O(HlogV)$
  * but still requires Space Complexity $O(HV)$

* Differentiated softmax (Chen et al., 2016)

* ​