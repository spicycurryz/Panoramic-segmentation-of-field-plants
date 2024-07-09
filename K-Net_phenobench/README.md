# K-Net@phenobench

## 1.安装 Installation

OpenMMLab 软件包如下:
版本需求
* MIM >= 0.1.5
* MMCV-full >= v1.3.14
* MMDetection >= v2.17.0
* MMSegmentation >= v0.18.0
* scipy
* panopticapi


```bash
pip install openmim scipy mmdet mmsegmentation
pip install git+https://github.com/cocodataset/panopticapi.git
mim install mmcv-full
```

## 2.数据准备 Data preparation
根据[MMDetection](https://github.com/open-mmlab/mmdetection)and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)准备数据集。数据集结构如下:

```plaintext
data/
├── coco
│   ├── annotations
│   │   ├── panoptic_{train,val}.json
│   │   ├── instance_{train,val}.json
│   │   ├── panoptic_{train,val}/ # panoptic png annotations
│   │   ├── image_info_test-dev.json # for test-dev submissions
│   ├── train
│   ├── val
│   ├── test
```

## 3.Training and testing
对于训练和测试，可以直接使用 MIM 来训练和测试模型


```bash
# 训练语义分割模型
python train.py --work-dir work_dir1 configs/det/knet/knet_s3_r101_dcn-c3-c5_fpn_ms-3x_coco-panoptic.py 
#环境配制建议：
mmcv                                     2.1.0
mmdet                                    3.3.0
mmengine                                 0.10.4 

# 训练实例分割模型
python train.py --work-dir work_dir3 configs/det/knet/knet_s3_r50_fpn_1x_coco-panoptic.py
#环境配制建议：
mmcv-full                                1.5.0
mmengine                                 0.10.4 

```
* PARTITION: 正在使用的 slurm 分区
* WORK_DIR: 保存配置、日志和检查点的工作目录
* CONFIG: 目录下的配置文件
* JOB_NAME: slurm所需的工作名称

## 4. OpenMMLab
[MMEngine](https://github.com/open-mmlab/mmengine): 用于训练深度学习模型的 OpenMMLab 基础库
[MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
[MMDetection](https://github.com/open-mmlab/mmdetection):OpenMMLab 检测工具箱和基准测试
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱和基准测试
[MIM](https://github.com/open-mmlab/mim): MIM 安装 OpenMMLab 软件包

