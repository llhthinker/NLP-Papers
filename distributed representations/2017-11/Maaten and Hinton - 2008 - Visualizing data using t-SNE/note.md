
## Introduction

* Visualizes high-dimensional data by giving each data point a location in a 2 or 3-dimensional map. 
* Most of these techniques simply provide tools to display more than two data dimensions, and leave the interpretation of the data to the human observer.
* 传统的**线性技术**主要是想让**不相似的点**在低维表示中**分开**。
  * PCA（Principle Components Analysis，主成分分析）
  * MDS（Multiple Dimensional Scaling，多维缩放）
* 对于处于低维、非线性**流形**上的高维数据而言，更重要的是让**相似的近邻点**在低维表示中**靠近**。**非线性技术**主要保持数据的局部结构。
  * Sammon mapping
  * CCA（Curvilinear Components Analysis）
  * **SNE（Stochastic Neighbor Embedding，随机近邻嵌入）**，**t-SNE是基于SNE的**。
  * Isomap（Isometric Mapping，等度量映射）
  * MVU（Maximum Variance Unfolding）
  * LLE（Locally Linear Embedding，局部线性嵌入）
  * Laplacian Eigenmaps
* In particular, most of the techniques are not capable of retaining both the **local** and the **global structure** of the data in a single map. 
* In this paper, we introduce 
  * a way of converting a high-dimensional data set into a **matrix of pairwise similarities** and,
  * a new technique, called “**t-SNE**”, for visualizing the resulting similarity data. 

## SNE

* Stochastic Neighbor Embedding，随机近邻嵌入

* 一种基于概率的数据降维处理方法。

* Given a set of **high-dimensional** data points $x_1, x_2, ..., x_n$, $p_{i|j}$ is the **conditional probability** that $x_i$ would pick $x_j$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian **centered at $x_i$**.

  $$p_{j|i} = \frac{exp(-||x_i-x_j||^2/2\sigma_i^2)}{\sum_{k\neq i}exp(-||x_i-x_k||^2/2\sigma_i^2)}$$

  注意每不同数据点$x_i$有不同的$\sigma_i$。其计算方式下面说。

* Similarly, define $q_{i|j}$ as **conditional probability** corresponding to **low-dimensional** representations of $y_i$ and $y_j$ (corresponding to $x_i$ and $x_j$). The variance of Gaussian in this case is set to be $1/\sqrt{2}$.

  $$q_{j|i} = \frac{exp(-||y_i-y_j||^2)}{\sum_{k\neq i}exp(-||y_i-y_k||^2)}$$

* If the map points $y_i$ and $y_j$ correctly model the similarity between the high-dimensional data points $x_i$ and $x_j$, the conditional probabilities $p_{j|i}$ and $q_{j|i}$ will be equal.

* SNE aims to find a low-dimensional data representation that **minimizes the mismatch between  $p_{j|i}$ and $q_{j|i}$**. Thus we use the **sum of Kullback-Leibler divergences over all data points** as the **cost function**:

  $$C=\sum_i KL(P_i||Q_i) = \sum_i \sum_j p_{j|i} log\frac{p_{j|i}}{q_{j|i}}$$

  in which $P_i(k=j)=p_{j|i}$ represents the conditional probability **distribution** over **all other data points** given data point $x_i$.  $Q_i(k=j)=q_{j|i}$ too.

* 对$C$进行梯度下降即可以学习到合适的$y_i$

  $$\frac{\partial C}{\partial y_i} = 2 \sum_j (p_{j|i} - q_{j|i} + p_{i|j} - q_{i|j})(y_i - y_j)$$

  t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results. 很难优化，也对初值十分敏感，因此要跑多次选取KLD最小/可视化最好结果。

  （注意这个不是为了泛化，因此选最好的结果即可。）

* Kullback-Leibler divergence is not symmetric. 因为 $KLD=p log(\frac{p}{q})$，p>q时为正，p<q时为负。则如果**高维数据相邻**而**低维数据分开**，则**cost很大**；相反，如果**高维数据分开**而**低维数据相邻**，则**cost很小**。所以t-SNE倾向于保留高维数据的局部结构。

* 如何为每一个$x_i$选取对应的$\sigma_i$？

  * It is not likely that there is a single value of $\sigma_i$ that is optimal for all data points in the data set because the density of the data is likely to vary. **In dense regions, small $\sigma_i$, while in sparse region, large $\sigma_i$.**

  * 让条件分布$P_i$的困惑度等于用户预定义的困惑度即可。

    $Perp(P_i) = 2^{H(P_i)} = 2^{-\sum_j p_{j|i} log_2 p_{j|i}}$

  * The perplexity can be interpreted as a smooth measure of **the effective number of neighbors**. The performance of SNE is fairly robust to changes in the perplexity, and **typical values are between 5 and 50.**

  * 这样，每一个$x_i$的$\sigma_i$就可以通过简单的 binary search (Hinton and Roweis, 2003) 或者鲁棒性非常高的 root-finding method (Vladymyrov and Carreira-Perpinan, 2013) 算法找到。找到的$\sigma_i$即具有如上所述的性质：数据密集的区域小，数据稀疏的区域大。

## t-SNE

- **t-Distributed** Stochastic Neighbor Embedding
- SNE有两个问题：
  - Cost function 难以优化。-> 解决方案：使用Symmetric SNE
  - Crowding Problem -> 解决方案：在低维嵌入上使用Student's t-distribution替代Guassian distribution。

###Symmetric SNE

- We define the **joint probabilities** $p_{ij}$ in the high-dimensional space to be the **symmetrized conditional probabilities**, that is, we set $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$.

- minimizing the sum of KLD between conditional probabilities -> minimizing a single KLD between joint probability distribution.

  $$C = KL(P||Q) = \sum_i \sum_j p_{ij} log \frac{p_{ij}}{q_{ij}}$$

- The main advantage of the symmetric version of SNE is the **simpler form of its gradient**, which is faster to compute. 

- $$\frac{\partial C}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)$$ 

  (注意：这只是Symmetric SNE的梯度公式，t-SNE的梯度公式推导见附录A)

### Crowding Problem

- When we model a high-dimensional dataset in 2 (or 3) dimensions, it is difficult to segregate the nearby datapoints from moderately distant datapoints and gaps can not form between natural clusters. 似乎说的就是维度灾难，在高维空间下，距离差异趋向于小，很难从根据距离区分出cluster。

- One way around this problem is to use UNI-SNE but optimization of the cost function, in that case, is difficult.

### t-SNE

* Instead of Gaussian, use a **heavy-tailed distribution (like Student-t distribution)** to convert distances into probability scores in **low dimensions**. This way **moderate** distance in high-dimensional space can be modeled by **larger** distance in low-dimensional space.

* **Student's t-distribution**

  * [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) has the probability density function given by

    $$f(t)={\frac {\Gamma ({\frac {\nu +1}{2}})}{{\sqrt {\nu \pi }}\,\Gamma ({\frac {\nu }{2}})}}\left(1+{\frac {t^{2}}{\nu }}\right)^{\!-{\frac {\nu +1}{2}}}$$

  * Special cases $\nu = 1$

    $$f(t) =  \frac{1}{\pi (1+t^2)}$$

    Called **Cauchy distribution**. 我们用到是这个简单形式。

  * Special cases $\nu = \infty$

    $$f(t) = \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}}$$

    Called **Guassian/Normal distribution**.

  * ![figure of probability density function](https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Student_t_pdf.svg/325px-Student_t_pdf.svg.png)

* **The joint probabilities $q_{ij}$** are defined as

  $$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k\neq l}(1+||y_k - y_l||^2)^{-1}}$$

  （注意：和SNE的$q_{j|i}$公式相比，分母中求和号中，之前是$k\neq i$，表示仅排除$i$自身项；现在是$k\neq l$，表示排除所有自身项 ）

* The cost function is easy to optimize.

  $$\frac{\partial C}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)(1+ ||y_i - y_j||^2)^{-1}$$

  （注意：是在Symmetric SNE的梯度公式后面加上了$(1+ ||y_i - y_j||^2)^{-1}$一项，推导见论文附录A。）

### Complexity

- **Space and time complexity is quadratic** in the number of datapoints so infeasible to apply on large datasets.
- Random walk
  - Select a random subset of points (called landmark points) to display.
  - for each landmark point, define a random walk starting at a landmark point and terminating at any other landmark point.
  - $p_{i|j}$ is defined as fraction of random walks starting at $x_i$ and finishing at $x_j$ (both these points are landmark points). This way, $p_{i|j}$ is not sensitive to "short-circuits" in the graph (due to noisy data points).
- **[Barnes-Hut approximations (2014)](http://www.jmlr.org/papers/volume15/vandermaaten14a/vandermaaten14a.pdf)**
  - allowing it to be applied on large real-world datasets. We applied it on data sets with up to 30 million examples.

## Advantages of t-SNE

- Gaussian kernel employed by t-SNE (in high-dimensional) defines a soft border between the local and global structure of the data.
- Both nearby and distant pair of datapoints get equal importance in modeling the low-dimensional coordinates.
- The local neighborhood size of each datapoint is determined on the basis of the local density of the data.
- Random walk version of t-SNE takes care of "short-circuit" problem.

## Limitations of t-SNE

- it is unclear how t-SNE performs on general dimensionality reduction tasks, 
- the relatively local nature of t-SNE makes it sensitive to the curse of the intrinsic dimensionality of the data, and 
- t-SNE is not guaranteed to converge to a global optimum of its cost function. 


## Recommended & Related Works

* [ref: How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
* [ref: t-SNE Q&A](http://lvdmaaten.github.io/tsne/)
* [ref: sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

