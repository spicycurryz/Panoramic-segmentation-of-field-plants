# MP-Former@phenobench

## 1.Installation

环境的配置参考[MP-Former环境配置](https://github.com/IDEA-Research/MP-Former/blob/main/README.md)
配置示例如下所示

```bash
conda create --name MP-Former python=3.8 -y
conda activate MP-Former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone git@github.com:IDEA-Research/MP-Former.git
cd MP-Former
pip install -r requirements.txt
cd MP-Former/modeling/pixel_decoder/ops
sh make.sh
```

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

* 使用1张V100显卡训练

```bash
python train_net.py --num-gpus 1\
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml
```

* 断点再训练

```bash
python train_net.py --num-gpus 1 \
--resume --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml
```

## 4.评估

运行以下指令进行评估

```bash
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --eval-only MODEL.WEIGHTS YOUR-MODEL-WEIGHT
```

其中MODEL.WEIGHTS部分输入训练模型权重的路径

## 5.运行demo实现可视化

运行以下指令指定test集的某一张图片进行可视化查看

```bash
cd demo
python demo.py --config-file ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml\
--input xxx.png\
--output OUTPUT_DIR\
--opts MODEL.WEIGHTS YOUR-MODEL-WEIGHT
```
