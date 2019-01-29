# NLP-Papers
- [Distributed Word Representations](#distributed-word-representations)
- [Distributed Sentence Representations](#distributed-sentence-representations)
- [Entity Recognition (Sequence Tagging)](#entity-recognition)
- [Language Model](#language-model)
- [Machine Translation](#machine-translation)
- [Question Answering (Machine Reading Comprehension)](#question-answering)
- [Relation Extraction](#relation-extraction)
- [Sentences Matching (Natural Language Inference/Textual Entailment)](#sentences-matching)
- [Text Classification (Sentiment Classification)](#text-classification)
- [Materials](#materials)

## Papers and Notes
### Distributed Word Representations
- 2017-11
  - Faruqui and Dyer - 2014 - **Improving vector space word representations using multilingual correlation** [[pdf]](http://repository.cmu.edu/lti/31/)  [[note]](./distributed%20representations/2017-11/Faruqui%20and%20Dyer%20-%202014%20-%20Improving%20vector%20space%20word%20representations%20using%20multilingual%20correlation/note.md)
  - Maaten and Hinton - 2008 - **Visualizing data using t-SNE** [[pdf]](http://www.jmlr.org/papers/v9/vandermaaten08a.html) [[pdf (annotated)]](./distributed%20representations/2017-11/Maaten%20and%20Hinton%20-%202008%20-%20Visualizing%20data%20using%20t-SNE/Maaten%20and%20Hinton%20-%202008%20-%20Visualizing%20data%20using%20t-SNE.pdf) [[note]](./distributed%20representations/2017-11/Maaten%20and%20Hinton%20-%202008%20-%20Visualizing%20data%20using%20t-SNE/note.md)
  - Ling et al. - 2015 - **Finding function in form: Compositional character models for open vocabulary word representation**  [[pdf]](https://arxiv.org/abs/1508.02096) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/2017-11/Finding%20Function%20in%20Form%20Compositional%20Character%20Models/Finding%20Function%20in%20Form%20Compositional%20Character%20Models.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/2017-11/Finding%20Function%20in%20Form%20Compositional%20Character%20Models/note.md) 
  - Bojanowski et al. - 2016 - **Enriching word vectors with subword information** [[pdf]](https://arxiv.org/abs/1607.04606) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/2017-11/Enriching%20Word%20Vectors%20with%20Subword%20Information/Enriching%20Word%20Vectors%20with%20Subword%20Information.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/2017-11/Enriching%20Word%20Vectors%20with%20Subword%20Information/note.md) 
- 2017-12
  - Bengio and Senécal - 2003 - **Quick Training of Probabilistic Neural Nets by Importance Sampling** [[pdf]](http://www.iro.umontreal.ca/~lisa/pointeurs/senecal_aistats2003.pdf) [[pdf(annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/2017-12/Quick%20Training%20of%20Probabilistic%20Neural%20Nets%20by%20Importance%20Sampling/Quick%20Training%20of%20Probabilistic%20Neural%20Nets%20by%20Importance%20Sampling.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/2017-12/Quick%20Training%20of%20Probabilistic%20Neural%20Nets%20by%20Importance%20Sampling/note.md)
- 2018-05
  - Peters et al. - 2018- **Deep contextualized word representations** [[pdf]](https://arxiv.org/abs/1802.05365)
- references
  - word Embedding
    - [word2vec(tensorflow)](https://github.com/llhthinker/udacity-deeplearning/blob/master/5_word2vec.ipynb)
    - [subword-based word vector](https://github.com/facebookresearch/fastText)
    - [Chinese Word Vectors 中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)
    - [Tencent AI Lab Embedding Corpus for **Over 8 Million** Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/embedding.html)
  - ELMo
    - [Pre-trained ELMo Representations for Many Languages](https://github.com/HIT-SCIR/ELMoForManyLangs)

### Distributed Sentence Representations
- 2017-11
  - Le and Mikolov - 2014 - **Distributed representations of sentences and documents** [[pdf]](http://proceedings.mlr.press/v32/le14.pdf) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/2017-11/Distributed%20Representations%20of%20Sentences%20and%20Documents/Distributed%20Representations%20of%20Sentences%20and%20Documents.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/2017-11/Distributed%20Representations%20of%20Sentences%20and%20Documents/note.md) 
- 2018-12
  - Li and Hovy - 2014 - **A Model of Coherence Based on Distributed Sentence Representation** [[pdf]](http://www.aclweb.org/anthology/D14-1218) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/A%20Model%20of%20Coherence%20Based%20on%20Distributed%20Sentence%20Representation.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#a-model-of-coherence-based-on-distributed-sentence-representation)
  - Kiros et al. - 2015 - **Skip-Thought Vectors** [[pdf]](http://papers.nips.cc/paper/5950-skip-thought-vectors) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/Skip-Thought%20Vectors.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#skip-thought-vectors)
  - Hill et al. - 2016 - **Learning Distributed Representations of Sentences from Unlabelled Data** [[pdf]](https://arxiv.org/abs/1602.03483) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/Learning%20Distributed%20Representations%20of%20Sentences%20from%20Unlabelled%20Data.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#learning-distributed-representations-of-sentences-from-unlabelled-data)
  - Arora et al. - 2016 - **A simple but tough-to-beat baseline for sentence embeddings** [[pdf]](https://openreview.net/forum?id=SyK00v5xx) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/A%20Simple%20but%20Tough-to-Beat%20Baseline%20for%20Sentence%20Embeddings.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#a-simple-but-tough-to-beat-baseline-for-sentence-embeddings)
  - Pagliardini et al. - 2017 - **Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features (sent2vec)** [[pdf]](https://arxiv.org/abs/1703.02507) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/Unsupervised%20Learning%20of%20Sentence%20Embeddings%20using%20Compositional%20n-Gram%20Features.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#unsupervised-learning-of-sentence-embeddings-using-compositional-n-gram-features)
  - Logeswaran et al. - 2018 - **An efficient framework for learning sentence representations (Quick-Thought Vectors)** [[pdf]](https://arxiv.org/abs/1803.02893) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/An%20efficient%20framework%20for%20learning%20sentence%20representations.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#an-efficient-framework-for-learning-sentence-representations)
- 2019-01
  - Wieting et al. - 2015 - **Towards universal paraphrastic sentence embeddings** [[pdf]](https://arxiv.org/abs/1511.08198) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/Towards%20universal%20paraphrastic%20sentence%20embeddings.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#towards-universal-paraphrastic-sentence-embeddings)
  - Adi et al. - 2016 - **Fine-grained Analysis of Sentence Embeddings Using Auxiliary Prediction Tasks** [[pdf]](https://arxiv.org/abs/1608.04207) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/Fine-grained%20Analysis%20of%20Sentence%20Embeddings%20Using%20Auxiliary%20Prediction%20Tasks.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#fine-grained-analysis-of-sentence-embeddings-using-auxiliary-prediction-tasks)
  - Conneau et al. - 2017 - **Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (InferSent)** [[pdf]](https://arxiv.org/abs/1705.02364) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/Supervised%20Learning%20of%20Universal%20Sentence%20Representations%20from%20Natural%20Language%20Inference%20Data.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#supervised-learning-of-universal-sentence-representations-from-natural-language-inference-data)
  - Cer et al. - 2018 - **Universal Sentence Encoder** [[pdf]](https://arxiv.org/abs/1803.11175) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/Universal%20sentence%20encoder.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/distributed%20representations/sentence-embedding/note.md#universal-sentence-encoder)
- references
  - [SentEval: evaluation toolkit for sentence embeddings](https://github.com/facebookresearch/SentEval)
  - [doc2vec(gensim)](https://github.com/jhlau/doc2vec)
  - [Skip-Thought Vectors](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)
  - [SIF(sentence embedding by Smooth Inverse Frequency weighting scheme)](https://github.com/PrincetonML/SIF)
  - [Quick-Thought Vectors](https://github.com/lajanugen/S2V)
  - [sent2vec](https://github.com/epfml/sent2vec)
  - [InferSent](https://github.com/facebookresearch/InferSent)
  
### Entity Recognition
- 2018-10
  - Lample et al. - 2016 - **Neural Architectures for Named Entity Recognition** [[pdf]](https://arxiv.org/abs/1603.01360)
  - Ma and Hovy - 2016 - **End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF** [[pdf]](https://arxiv.org/abs/1603.01354)
  - Yang et al. - 2017 - **Transfer Learning for Sequence Tagging with Hierarchical Recurrent Networks** [[pdf]](https://arxiv.org/abs/1703.06345)
  - Peters et al. - 2017 - **Semi-supervised sequence tagging with bidirectional language models** [[pdf]](https://arxiv.org/abs/1705.00108)
  - Shang et al. - 2018 - **Learning Named Entity Tagger using Domain-Specific Dictionary** [[pdf]](https://arxiv.org/abs/1809.03599)
- references
  - [ChineseNER (TensorFlow)](https://github.com/zjy-ucas/ChineseNER)
  - [flair (PyTorch)](https://github.com/zalandoresearch/flair)
### Language Model
- 2017-11
  - Bengio et al. - 2003 - **A neural probabilistic language model** [[pdf]](http://www.jmlr.org/papers/v3/bengio03a.html)
  - Press and Wolf - 2016 - **Using the output embedding to improve language model** [[pdf]](https://arxiv.org/abs/1608.05859)
- references
  - [LSTM for Language Model](https://github.com/gaussic/language_model_zh/blob/master/lm_chinese.ipynb)

### Machine Translation
* 2017-12
  * Oda et al. - 2017 - **Neural Machine Translation via Binary Code Predict** [[pdf]](https://arxiv.org/abs/1704.06918) [[note]](./machine%20translation/Oda%20et%20al.%20-%202017%20-%20Neural%20Machine%20Translation%20via%20Binary%20Code%20Prediction/note.md)
  * Kalchbrenner et al. - 2016 - **Neural machine translation in linear time** [[pdf]](https://arxiv.org/abs/1610.10099) [[pdf (annotated)]](./machine%20translation/Kalchbrenner%20et%20al.%20-%202016%20-%20Neural%20machine%20translation%20in%20linear%20time/Kalchbrenner%20et%20al.%20-%202016%20-%20Neural%20machine%20translation%20in%20linear%20time.pdf) [[note]](./machine%20translation/Kalchbrenner%20et%20al.%20-%202016%20-%20Neural%20machine%20translation%20in%20linear%20time/note.md)
* 2018-05
  * Sutskever et al. - 2014 - **Sequence to Sequence Learning with Neural Networks** [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural)
  * Cho et al. - 2014 - **Learning Phrase Representations using RNN Encoder-Decoder for NMT** [[pdf]](https://arxiv.org/abs/1406.1078) 
  * Bahdanau et al. - 2014 - **NMT by Jointly Learning to Align and Translate** [[pdf]](https://arxiv.org/abs/1409.0473)
  * Luong et al. - 2015 - **Effective Approaches to Attention-based NMT** [[pdf]](https://arxiv.org/abs/1508.04025)
* 2018-06
  * Gehring et al. - 2017 - **Convolutional sequence to sequence learning** [[pdf]](https://arxiv.org/abs/1705.03122)
  * Vaswani et al. - 2017 - **Attention is all you need** [[pdf]](https://arxiv.org/abs/1706.03762) [[note1:The Illustrated Transformer]](http://jalammar.github.io/illustrated-transformer/) [[note2:The Annotated Transformer]](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
* references
  - [OpenNMT-py (in PyTorch)](https://github.com/OpenNMT/OpenNMT-py)
  - [nmt (in TensorFlow)](https://github.com/tensorflow/nmt)
  - [MT-Reading-List](https://github.com/THUNLP-MT/MT-Reading-List)

### Question Answering

* 2018-03
  * Wang and Jiang. - 2016 - **Machine Comprehension Using Match-LSTM and Answer Pointer** [[pdf](https://arxiv.org/abs/1608.07905)]
  * Seo et al. - 2016 - **Bidirectional Attention Flow for Machine Comprehension** [[pdf](https://arxiv.org/abs/1611.01603)] 
  * Cui et al. - 2016 - **Attention-over-Attention Neural Networks for Reading Comprehension** [[pdf](https://arxiv.org/abs/1607.04423)]
* 2018-04
  * Clark and Gardner. - 2017 - **Simple and Effective Multi-Paragraph Reading Comprehension** [[pdf](https://arxiv.org/abs/1710.10723)]
  * Wang et al. - 2017 - **Gated Self-Matching Networks for Reading Comprehension and Question Answering** [[pdf](http://www.aclweb.org/anthology/P17-1018)]
  * Yu et al. - 2018 - **QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension** [[pdf](https://arxiv.org/abs/1804.09541)] 
* references
  - [DuReader](https://github.com/baidu/DuReader)
  - [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
  - [MS MARCO](http://www.msmarco.org/leaders.aspx)
  - [深度学习解决机器阅读理解任务的研究进展](https://zhuanlan.zhihu.com/p/22671467)
  - [RCPapers: Must-read papers on Machine Reading Comprehension](https://github.com/thunlp/RCPapers)

### Relation Extraction
* 2018-08 
  * Mintz et al. - 2009 - **Distant supervision for relation extraction without labeled data** [[pdf]](https://dl.acm.org/citation.cfm?id=1690287)
  * Zeng et al. - 2015 - **Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks** [[pdf]](http://www.aclweb.org/anthology/D15-1203)
  * Zhou et al. - 2016 - **Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification** [[pdf]](http://www.aclweb.org/anthology/P16-2034)
  * Lin et al. - 2016 - **Neural Relation Extraction with Selective Attention over Instances** [[pdf]](http://www.aclweb.org/anthology/P16-1200)
* 2018-09
  * Ji et al. - 2017 - **Distant Supervision for Relation Extraction with Sentence-Level Attention and Entity Descriptions** [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14491/14078)
  * Levy et al. - 2017 - **Zero-Shot Relation Extraction via Reading Comprehension** [[pdf]](https://arxiv.org/abs/1706.04115)
* references
  * [OpenNRE](https://github.com/thunlp/OpenNRE)
  * [NREPapers: Must-read papers on neural relation extraction (NRE)](https://github.com/thunlp/NREPapers)
  * [awesome-relation-extraction](https://github.com/roomylee/awesome-relation-extraction)
  
### Sentences Matching

* 2017-12
  * Hu et al. - 2014 - **Convolutional neural network architectures for Matching Natural Language Sentences**  [[pdf]](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/sentences%20matching/2017-12/%20Convolutional%20Matching%20Model%20/Convolutional%20Neural%20Network%20Architectures%20for%20Matching%20Natural%20Language%20Sentences.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/sentences%20matching/2017-12/Convolutional%20Matching%20Model/note.md)
* 2018-07
  * Nie and Bansal - 2017 - **Shortcut-Stacked Sentence Encoders for Multi-Domain Inference** [[pdf]](https://arxiv.org/abs/1708.02312) [[note]](./sentences%20matching/note.md#shortcut-stacked-sentence-encoders-for-multi-domain-inference)
  * Wang et al. - 2017 - **Bilateral Multi-Perspective Matching for Natural Language Sentences** [[pdf]](https://arxiv.org/abs/1702.03814) [[note]](./sentences%20matching/note.md#bilateral-multi-perspective-matching-for-natural-language-sentences)
  * Tay et al. - 2017 - **A Compare-Propagate Architecture with Alignment Factorization for Natural Language Inference** [[pdf]](https://arxiv.org/abs/1801.00102)
  * Chen et al. - 2017 - **Enhanced LSTM for Natural Language Inference** [[pdf]](https://arxiv.org/abs/1609.06038) [[note]](./sentences%20matching/note.md#enhanced-lstm-for-natural-language-inference)
  * Ghaeini et al. - 2018 - **DR-BiLSTM: Dependent Reading Bidirectional LSTM for Natural Language Inference** [[pdf]](https://arxiv.org/abs/1802.05577)
* references
  * [The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)


### Text Classification
- 2017-09
  - Joulin et al. - 2016 - **Bag of tricks for efficient text classification** [[pdf]](https://arxiv.org/abs/1607.01759v3) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-09/Bag%20of%20Tricks%20for%20Efficient%20Text%20Classification/Bag%20of%20Tricks%20for%20Efficient%20Text%20Classification.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-09/Bag%20of%20Tricks%20for%20Efficient%20Text%20Classification/note.md)
- 2017-10
  - Kim - 2014 - **Convolutional neural networks for sentence classification** [[pdf]](https://arxiv.org/abs/1408.5882) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-10/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-10/Convolutional%20Neural%20Networks%20for%20Sentence%20Classification/note.md)
  - Zhang and Wallace - 2015 - **A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification** [[pdf]](https://arxiv.org/abs/1510.03820) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-10/A%20Sensitivity%20Analysis%20of%20(and%20Practitioners%E2%80%99%20Guide%20to)%20Convolutional/A%20Sensitivity%20Analysis%20of%20(and%20Practitioners%E2%80%99%20Guide%20to)%20Convolutional.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-10/A%20Sensitivity%20Analysis%20of%20(and%20Practitioners%E2%80%99%20Guide%20to)%20Convolutional/note.md)
  - Zhang et al. - 2015 - **Character-level convolutional networks for text classification** [[pdf]](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-fo) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-10/Character-level%20Convolutional%20Networks%20for%20Text%20Classification/Character-level%20Convolutional%20Networks%20for%20Text%20Classification.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-10/Character-level%20Convolutional%20Networks%20for%20Text%20Classification/note.md)
  - Lai et al. - 2015 - **Recurrent Convolutional Neural Networks for Text Classification** [[pdf]](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-10/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-10/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification/note.md)
- 2017-11
  - Iyyer et al. - 2015 - **Deep unordered composition rivals syntactic methods for Text Classification** [[pdf]](http://www.aclweb.org/anthology/P15-1162) [[pdf (annotated)]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-11/Deep%20Unordered%20Composition%20Rivals%20Syntactic%20Methods%20for%20Text%20Classification/Deep%20Unordered%20Composition%20Rivals%20Syntactic%20Methods%20for%20Text%20Classification.pdf) [[note]](https://github.com/llhthinker/NLP-Papers/blob/master/text%20classification/2017-11/Deep%20Unordered%20Composition%20Rivals%20Syntactic%20Methods%20for%20Text%20Classification/note.md)
- references
  - [fastText](https://github.com/facebookresearch/fastText)
  - [text_classification](https://github.com/brightmart/text_classification)
  - [PyTorchText(知乎看山杯)](https://github.com/chenyuntc/PyTorchText)

### Materials

- [Neural Networks for NLP (CS11-747 Fall 2017 @ CMU)](http://www.phontron.com/class/nn4nlp2017/schedule.html)

- optimization algorithms

  - [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)

- [NLP-progress](https://github.com/sebastianruder/NLP-progress)

- [Awesome-Chinese-NLP](https://github.com/crownpku/awesome-chinese-nlp)
- [StateOfTheArt.ai](https://www.stateoftheart.ai/)
- [funNLP(从文本中抽取结构化信息等资源汇总)](https://github.com/fighting41love/funNLP)
