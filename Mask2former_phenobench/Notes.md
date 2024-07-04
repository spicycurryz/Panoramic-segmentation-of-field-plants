# Mask2former论文学习笔记

***

## 预备知识

由于Mask2fomer论文中提及了许多模式识别课程之外的模型或者知识，在阅读的过程中我遇到了很多困难，于是我先去学习了一下额外的预备知识。

***

### 1.Transformer

Transformer是一个基于注意力机制和Encoder-Decoder的网络主干结构，最早用于机器翻译，后来研究者发现其在自然语言处理和图像分类中取得了优越的性能。其基本形式适用于序列建模任务。

#### Encoder-Decoder

* Encoder-Decoder框架顾名思义也就是编码-解码框架，目前大部分attention模型都是依附于Encoder-Decoder框架进行实现。
* 编码器对于输入的序列<x1,x2,x3…xn>进行编码，使其转化为一个语义编码C，解码器根据输入的语义编码C，然后将其解码成序列数据<y1,y2,y3…yn>，解码方式可以采用RNN/LSTM。
* 普通的Encoder-Decoder结构存在一定的问题：解码器在生成序列数据时，使用的语义编码C是相同的，这意味着不论是生成哪个序列数据，y1,y2还是y3，其实输入序列X中任意数据对生成某个目标数据yi来说影响力都是相同的，没有任何区别。另外一个问题是：如果输入序列一长，只是用一个语义编码C来表示整个序列的信息肯定会损失很多信息，这样将所有信息压缩在一个C里面显然就不合理。

#### Attention model

* Attention model模仿人的注意力机制。包括Query(Q),key(k),value(V)三个部分。V相当于一个知识库，是我们手头已有的资料，K是知识库的钥匙(标志?)，Q是我们待查询的东西。
* 先将Q和K做点积查看两者的相似度关系，接着使用softmax函数进行归一化，得到attention矩阵，将attetion矩阵与V的乘积作为输出。\[O = softmax(Q{K^T})V\]
* Transformer中还对上述注意力机制进行了改进，使用了“多头注意力机制”，多头注意力机制假设输入的特征空间可以分为互不相交的几个子空间，然后我们只需要在几个子空间内单独计算注意力，最后将多个子空间的计算结果拼接即可。

#### Padding Mask

* 简单总结来说就是输入序列长度不一致时需要统一batch，当对较短的序列进行padding时，引入掩膜机制，将序列padding对应的attention权重赋值为0，让attention的计算结果不受padding的影响。

#### position encoding（位置编码）

* 在Transformer中，模型输入其实隐含了前后顺序，然而，从attention的形式看，attention并不具备上述对位置进行编码的能力，这是因为attention对所有输入都一视同仁，因此，为了保留原始输入的顺序关系，我们需要在Transformer的输入中加入表征顺序的特征，进行位置编码。

#### Transformer原理

![transformer](https://raw.githubusercontent.com/XB304/image/main/img/transformer.png)

* 如图所示，transformer由编码器和解码器组成，解码器包含掩膜多注意力模块，归一化模块，前馈全连接神经网络，输出经线性化后经softmax层，输出最为可能的预测值。

***

## 2.Maskformer结构

作为Mask2former模型的前身，在阅读论文前首先对Maskformer进行了一定的了解。

![maskformer](https://raw.githubusercontent.com/XB304/image/main/img/maskformer.png)

在同一分割图像模型提出前，语义分割的策略是把问题简化为单个像素分类的问题，即将图像离散成单个的像素点，对每个像素点进行分类（类别数来自于数据集定义的类别数），把分割问题转化成了分类问题。这样做的缺点是语义分割结果只能输出固定个数的类别数目，很难解决实例分割这样更难的问题。而实例分割的思路是以二元掩膜作为基本单位（binary Mask），对于每一个binary Mask预测一个类别。
MaskFormer模型认为，使用二元掩膜进行分割的mask classfication可以同样应用于语义分割，取代之前逐像素分类的处理方式，统一了语义分割，实例分割和全景分割。MaskFormer提出将全景分割看成是mask分类任务。

模型分析如下：Pixel-level module模块用backbone对图片的H和W进行压缩，提取视觉特征，通过像素解码器得到per-pixel embedding；Transformer module模块使用标准的 Transformer 解码器，输入是图像特征和N个可学习的query，输出是N个mask embedding和N个cls pred；最终Segmentation module模块根据mask embedding和per-pixel embedding相乘得到N个mask prediction，最后cls pred和mask pred相乘，丢弃没有目标的mask，得到最终预测结果。

***

## 3.Mask2Former结构

如下图所示MaskFormer作为统一分割模型，在性能上接近或超越了专门用于语义分割和全景分割的模型，但是在实例分割领域，MaskFormer还是较差。

![mask2fomer指标](https://raw.githubusercontent.com/XB304/image/main/img/mask2fomer指标.png)

Mask2Former的提出大大提高了模型的性能，使得Mask2Former作为统一分割模型，首次在性能上超越了专门用于特定分割的模型。

![mask2former](https://raw.githubusercontent.com/XB304/image/main/img/mask2former.png)

Mask2Former增加了mask-attention机制：在做embeding时， MaskFormer使用每个像素和整张图像做attention，Mask2Former使用像素和自己同一个类别的，同一个qurey的像素做attention。个是相对maskformer最主要的区别，最核心的贡献。另外，pixel decoder输出的图像特征大小分别为原图的1/32, 1/16, 1/8。对于每个分辨率的图片，在给到Transformer decoder之前，会加入sinusoidal positional embedding 和一个可学习scale-level embedding 。Transformer decoder对这种三层Transformer decoder结构重复L次。
