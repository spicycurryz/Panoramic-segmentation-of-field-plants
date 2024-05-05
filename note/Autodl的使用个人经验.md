# Autodl的使用
由于没有本地的算力，我们需要借助一些云端服务器来完成一些模型训练工作，进而实现代码复现。我的原本计划是使用百度提供的aistudio平台，该平台我有少量的使用经验且相对熟悉，但很遗憾的是aistudio平台现在已经不再支持pytorch框架(在notebook里完全无法使用，在终端虽然能安装pytorch，但每次启动环境都需要重新配置，非常的麻烦，且cuda版本对不上)，因此我放弃了使用该平台。在综合比较之后选择了autodl。该平台使用流程大致如下：
## 1. 注册账号，创建实例
首先登陆autodl官网创建账号，创建完成之后来到“容器实例”页面
![autodl1](https://raw.githubusercontent.com/spicycurryz/My_img/main/img/autodl1.png)

选择“租用新实例”，之后选择地区和要租用的型号，地区一般选资源较多的，型号我这里担心显存不足，选用的是Tesla V100
![autodl2](https://raw.githubusercontent.com/spicycurryz/My_img/main/img/autodl2.png)

选完型号之后要根据自己要复现的项目中提到的环境配置要求选择框架。以我学习的oneformer为例，根据要求选用基础镜像中的"pytorch1.10.0 python 3.8 cuda 11.3".
![oneformer1](https://raw.githubusercontent.com/spicycurryz/My_img/main/img/oneformer1.png)

![autodl3](https://raw.githubusercontent.com/spicycurryz/My_img/main/img/autodl3.png)

这里在选择时如果基础镜像里没有符合要求的配置，可以选择miniconda，之后启动环境再进行配置
![autodl4](https://raw.githubusercontent.com/spicycurryz/My_img/main/img/autodl4.png)

创建实例成功后，在“容器实例”界面即可看到已经创建的实例，选择jupyterlab即可进入环境

## 2. 上传数据
autodl提供了一个200G的云空间供我们上传数据，且在关机后不会清除数据，这是很有帮助的功能。</br>
首先选择文件存储界面，选择刚刚创建的实例服务器所在地区，点击上传按键后选择文件即可上传
![autodl5](https://raw.githubusercontent.com/spicycurryz/My_img/main/img/autodl5.png)

但这种方式只能上传单个文件，如果想要上传文件夹，可以使用Xftp软件，具体可以参考这篇CSDN文档：[Autodl的使用](https://blog.csdn.net/LWD19981223/article/details/127085811 "AutoDL使用教程")</br>

我用这种方式上传了要用的数据集Phenobench.zip,数据集压缩包本身就有7G多，上传速度还是比较慢的。上传成功后在实例里即可使用

## 3. 进入实例，配置环境
选择jupyterlab进入之后，会看到如下界面：
![autodl6](https://raw.githubusercontent.com/spicycurryz/My_img/main/img/autodl6.png)

如果之前选用的是基础镜像，则不需要用conda额外配置一个环境，直接点击笔记本下方的pythonkernel即可进入jupyter notebook界面。

之后就可以根据要求在终端或者是jupyter notebook里进行环境的配置和代码的执行了。