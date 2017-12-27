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

## Binary Code Prediction Models

- $\mathbf{b}(w) := [b_1(w), b_1(w), ..., b_B(w)] \in \{0,1\}^B$（$\{0,1\}^B$可以看做是$R^n$一类的记号）
- 编码满足：1）每个word必有一个编码。2）不同word编码不同。3）留有冗余，即同一word可以有多个编码。因此$B \geq log_2V$。
- 默认UNK为二进制的0，BOS为二进制的1，EOS为二进制的2，其余的单词按照频率排序得到一个rank(w)，最终表示为二进制的2+rank(w)。
- Loss Function
  - 可以使用平方误差，也可以使用交叉熵。

## More Improvements

- While this idea is simple and intuitive, we found that it alone was not enough to achieve competitive accuracy with real NMT models. Thus we make two improvements:

### 改进1：Hybrid Model



### 改进2：Convolutional error correcting codes with Viterbi decoding