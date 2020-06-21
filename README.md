### EmotionCompute
计算新闻文本情感值，采用回归分析利用标注进行监督训练。输入新闻语料，输出attitude_score.

#### 模块描述
##### -> data
存放语料文件，用于训练的处理好的文件。以及目标未标注文件。可以将所有含有english_content字段的excel表放在这里参与词向量训练。  
word2vec_corpus放所有可以用来训练word2vec模型的文件。
processed_data放处理后的训练用的数据。  
unlabel_corpus放需要标注的文件。

##### -> figures
相关数据图表。
##### -> models
训练好的词向量模型以及回归模型。
##### -> utils
基础支持模块，包括停用词的处理等。主要为common_utils.py。
##### -> viz_loss
回归损失的可视化。
##### -> config.py
相关配置的全局变量模块。
##### -> stream_sentence_genenrator.py
新闻语料流式数据生成器。
##### -> train_word2vec.py
利用相关库对新闻的英文翻译训练词向量模型。
##### -> train_regression.py
训练回归神经网络。
##### -> ml_regression.py
利用机器学习库进行评估和对比。

#### -> Necessary Requirement
1. pytorch
2. gensim
3. pandas

#### -> 步骤
1. 将语料文件放至对应目录下
2. 将config更新为想要的配置
3. 运行train_wordvec得到词向量
4. 运行data_generator得到训练用的数据
5. 配置并运行train_regression，挑选并得到回归模型
6. 运行scoring对没有标注的文件进行标注(训练好之后只执行该文件即可)

#### -> 结果及分析
a. 对不同词嵌入大小进行训练后，词嵌入大小为64取得的效果比较好。  
b. TextCNN比LSTM好。  
c. TextCNN中的超参是实验调出来的，相对于当前的数据量来说是最好的。  
对语料中的单词数进行统计,90%左右的文章单词数都小于800.
![Seq-Length](https://github.com/luyu-fan/news-emotion-computation/tree/master/figures/words_length.png)
依然存在的问题：经过很长时间的训练调参之后，模型在验证集合上的损失始终无法下降到2一下，利用sklearn库进行对比实验产生的效果还不如神经网络。
对每一篇文章得到的词嵌入进行降维，得到的结果如下：
![训练数据集](https://github.com/luyu-fan/news-emotion-computation/tree/master/figures/myplot_tsne_train.png)  
![测试数据就](https://github.com/luyu-fan/news-emotion-computation/tree/master/figures/myplot-tsne_valid.png) 
共有6000条用于训练的样本，可以看到几乎各个标注分数区间的样本分布的区域都是一样的，几乎很难区分开，样本分布是混乱的，而不是在一个曲面上。训练过程中的损失如下图所示，如果样本量增加之后或许会得到更好的结果。
![训练损失](https://github.com/luyu-fan/news-emotion-computation/tree/master/figures/regression_loss.png)  
a. Sklearn库训练得到的误差统计
![ML方法](https://github.com/luyu-fan/news-emotion-computation/tree/master/figures/myplot-ml.png)
b. 神经网络回归模型得到的误差统计
![神经网络回归](https://github.com/luyu-fan/news-emotion-computation/tree/master/figures/regression_error.png)

