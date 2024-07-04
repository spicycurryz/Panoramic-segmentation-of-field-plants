# Panoptic-DeepLab@phenobench

## 1.Installation

按照说明安装[Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

可以在detectron2的内置项目中找到[Panoptic-DeepLab模型](https://github.com/facebookresearch/detectron2/tree/main/projects/Panoptic-DeepLab)

## 2.数据集准备

参照以下格式准备coco数据集

```plaintext
coco/
  annotations/
    instances_{train,val}.json
    panoptic_{train,val}.json
  {train,val,test}/
  panoptic_{train,val}/  
  panoptic_semseg_{train,val}/
```

## 3.训练

* 使用2个GPU进行训练

```bash
cd /path/to/detectron2/projects/Panoptic-DeepLab
python train_net.py --config-file configs/COCO-PanopticSegmentation/panoptic_deeplab.yaml --num-gpus 2
```

* 断点处开始训练

```bash
cd /path/to/detectron2/projects/Panoptic-DeepLab
python train_net.py --config-file configs/COCO-PanopticSegmentation/panoptic_deeplab.yaml --num-gpus 2 --resume
```

## 4.评估

* 运行如下指令评估模型

```bash
cd /path/to/detectron2/projects/Panoptic-DeepLab
python train_net.py --config-file configs/COCO-PanopticSegmentation/panoptic_deeplab.yaml  --eval-only MODEL.WEIGHTS/YOUR-MODEL-WEIGH
```

MODEL.WEIGHTS部分输入训练模型权重的路径

## 5.运行demo实现可视化

* 使用detectron2内置的demo.py进行可视化查看
* 运行如下的指令指定test集的任意一张图片

```bash
cd demo
python demo.py  --config-file  ../projects/Panoptic-DeepLab/configs/COCO-PanopticSegmentation/panoptic_deeplab.yaml\
 --input  xxx.png\
 --output  OUTPUT_DIR\
 --opts MODEL.WEIGHTS /YOUR-MODEL-WEIGH
```
