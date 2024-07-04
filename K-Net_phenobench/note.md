# 《K-Net:面向统一图像分割》
## 背景
&emsp;&emsp;K-Net来源于2021年发布的《K-Net: Towards Unified Image Segmentation》，其核心理念是在保持高性能的同时，降低深度学习模型的复杂度。通过采用一种叫做“空间-通道融合” （Spatial-Temporal Fusion）的新方法，K-Net能够更好地处理大规模数据集，并且适用于计算机视觉、自然语言处理等多个领域。针对语义分割、实例分割、全景分割三个任务，K-Net提出了一种基于动态内核的分割模型，为每个任务分配不同的核来实现多任务统一一开始一系列的卷积核都是随机初始化的，然后根据分割目标，即语义类别的语义核和实例标识的实例核，学习这些卷积核。一个简单的语义核和实例核的组合，能够获得全景分割(panoptic segmentation)。
&emsp;&emsp;K-Net动态地更新卷积核，使得它们对于它们对于图像的响应是不一样的。 这样的content-aware机制能够使得每个卷积核精确对应于不同的目标。 通过迭代应用这种自适应核更新策略，K-Net显著提高了核的识别能力，提高了最终分割性能。值得注意的是，该策略普遍适用于所有分割任务的卷积核。


## 先行知识

#### 图像分割目标
&emsp;将图像中不同的具有相似性、一致性的像素聚集在一起
  - 语义分割：将每一个像素映射到一个语义类别（每个pixel代表一个语义类别）
  - 实例分割：将每一个像素映射到一个instance ID （同一个object上）
  - 全境分割：将每一个像素映射到一个instance ID或语义类别{存在stuff（不可数区域）}

## K-Net的本质

&emsp;&emsp;对于所有细分的分割任务，其内核都是对有意义的像素进行分组，譬如语义分割，将不同的类别像素分组。理论上而言，分割任务的分组是有上限的，因此，可以把分组数量认为设置为N，比如，有N个用于语义分割的预定义语义类或图像中最多有N个实例对象，对于全景分割，N是图像中的stuff类和Instance类的总数。
那么，可以使用N个Kernel将图像划分为N组，每个Kernel都负责找到属于其相应组的像素（也就是先前讲的，Kernel与Content实现一对一映射）。具体而言，给定由深神经网络产生的B图像的输入特征映射 $
{F\in R}^{B\times C\times H\times W}
$ 。
$$
\ M=\sigma(K\ast F) 
$$

&emsp;&emsp;使用$\sigma$对结果进行激活，设置对应阈值后即可得到N个二进制掩码Mask（这也是语义分割长期以来的基本思想 。但是，为了实现实例分割，需要对每一个Kernel进行限制，也就是每个Kernel最多只能处理图像中一个对象，通过这种方式，K-NET可以区分实例并同时执行分割，从而在一个特征映射中实现实例分割，而无需额外的步骤（实现了NMS-Free和Box-Free）。为简单起见，我们可以将这些内核称为本文中的语义内核和实例内核，分别用于语义和实例分割。实例内核和语义内核的简单组合即可实现全景分割，该分割将像素分配给实例对象或一类东西。


### Group-Aware Kernels

&emsp;&emsp;虽然使用Kernel来区分语义类别的构想已经成形，但是要区分实例对象仍然十分棘手。因为实例内核需要区分图像内部和跨图像内外部的对象，不像语义类别具有共同和明确的特征，实例内核需要拥有比语义内核更强的判别能力。
&emsp;&emsp; 因此模型的创造团队提出了一个Kernel Update策略，来使每一个内核对应一个像素组。
Kernel Update Head $ f_i $包含三个关键步骤：
group feature assembling、adaptive kernel update 和 kernel interaction。
* group feature assembling：首先，通过Mask $M_{i-1} $来计算聚合出一个组特征映射$F^K$。其中，每一个组group都对应着一个语义类实例对象。
* adaptive kernel update： 这个F^K内的每一个组都将会被用来更新内核$K_{i-1}$。
* kernel interaction：随后，这些内核$K_{i-1}$进行交互，互相能够提供上下文信息以对全局特征图进行建模，获得新的内核$K_i$ 。
&emsp;&emsp; 最后，使用$K_i$对特征图F进行卷积，得到预测结果更加精确的Mask$ K_i$。
用公式表达就是：
$$
 K_i,M_i=f_i(M_{\left(i-1\right)},K_{\left(i-1\right)},F)
$$


#### Group Feature Assembling
&emsp;&emsp; Kernel Update Head首先聚合每个group的特征，随后使用这些特征来实现核的group-aware，也就是每一个核能够对特征图正确感知。由于$K_{i-1}$ 中每个内核的掩码本质上定义了像素是否属于该内核的相关组，可以通过将特征映射F乘$K_{i-1}$ 作为新的组装特征$F^K$：
$$
F^K=\sum_{u}^{H}\sum_{v}^{W}{(M_{\left(i-1\right)}(u,v)}·F(u,v),{F^K\in R}^{B\times N\times C}
$$

#### Adaptive Feature Update
&emsp;&emsp; 经过Group Feature Assembling之后，我们可以使用$F^K$计算得到一组新的Kernels。但作者考虑到mask $M_{i-1}$可能不够准确，可能包含了其他组被误分类进来的噪音，因此设计了一个自适应的内核更新策略，首先在$F^K$和$M_{i-1}$之间执行元素乘法$(\mathrm{\Phi}_1、Φ2 $为线性变换)：
$$
F^G=\mathrm{\Phi}_1\left(F^K\right)\ \otimes\mathrm{\Phi}_2K_{i-1},{F^G\in R}^{B\times N\times C}
$$
&emsp;&emsp; 随后，计算两个门控gates,$ G^F$和$G^K$：
$$
G^K=\sigma\left(\mathrm{\Phi}_1\left(F^G\right)\right)\ 、G^F=\sigma\left(\mathrm{\Phi}_2\left(F^G\right)\right)
$$ 
&emsp;&emsp; 再由这两个gates计算出一组新的kernels $\widetilde{K}$：
$$
\widetilde{K}=G^F\ \otimes\mathrm{\Phi}_3\left(F^K\right)+G^K\ \otimes\mathrm{\Phi}_4\left(K_{i-1}\right)$$
&emsp;&emsp; 其中，$ \mathrm{\Phi}_n $函数均为Fully connected layers（全连接层）+ Layer Norm（层归一化）。计算结果$\widetilde{K}$则将用于Kernel Interaction中。
&emsp;&emsp; 计算出来的两个gate，$G^F$和 $G^K$，可以看做是Transformer中的self-attention机制，本质上是计算一个权重对特征$F^K$和核$K_{i-1}$做一个加权求和。
 

#### Kernel Interaction
&emsp;&emsp; Kernel Interaction可以使不同的kernel之间互相信息流通，也就是能够提供上下文信息，这些信息允许kernel隐式利用图像group之间的关系。为了从上诉计算出来的$\widetilde{K}$中计算出一组新的kernels$K_i$，作者这里采用了Multi-Head Self-Attention与Feed-Forward Neural Network结合的形式。也就是用$ MSA-MLP$的形式来输出一组新的$K_i$。$K_i$将用来计算新的Mask：$$M_i=g_i(K_i)\ast F$$
&emsp;&emsp; 这里的$g_i$为$FC-LN-ReLU$操作。
### 全景分割

&emsp;&emsp; 在Kernel Update Head的基础上，通过一个BackBone和Neck（作者这里使用了FPN）来生成一组特征图F。由于语义分割和实例分割所要求的特征图有所差异，所以作者通过两个独立的卷积分支对F进行处理生成$F^{ins}$和$F^{seg}$，当然，这里使用的卷积核为初始化的$K_0^{ins}$和$K_0^{seg}$，这样就生成了一组新的Mask：$M_0^{ins}$、$M_0^{seg}$。由于$M_0^{seg}$中自然而然的包括了全景分割中所需求的things和stuff，那么只需要从$M_0^{seg}$中将包含stuff的部分提取出来，再和$M_0^{ins}$直接进行通道相加，即可得到全景分割所需要的Mask：$M_0$。同理，对于卷积核Kernel，也只需要提取对应的$K_0^{ins}$和$K_0^{seg}$组合成新的$K_0$。而对于特征图F，将$F^{ins}$和$F^{seg}$简单通道相加即可。对于新得到的$M_0$、$K_0$和$F$，经过S次Kernel Update Head处理，可以得到最终的Output：$M_S$。

### 实例分割（instance segmentation）
&emsp;&emsp; 与上述全景分割类似，在实例分割中，只需要删除内核和掩模的串联过程即可执行实例分割。在这一步并不需要删除语义分割分支，因为语义信息仍然是互补的，毕竟实例信息可以从语义信息中提取。不过，需要注意的是，在这种情况下，语义分割分支不使用额外的真实标签，语义结果的标签是通过将实例掩码转换为相应的类标签来构建的，在一定程度上起了一个监督的作用。

### 语义分割（Semantic segmentation）
&emsp;&emsp; 对于语义分割的流程而言，结构是最为简单的，只需要简单地将Kernel Update Head附加到任何依赖语义内核的语义分割方法中即可执行语义分割。
