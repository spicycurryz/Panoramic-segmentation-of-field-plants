# K-Net@phenobench

## 1.Installation

It following OpenMMLab packages:
requires the 
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

## 2.Data preparation
Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection)and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). The data structure looks like below:

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
For training and testing, you can directly use mim to train and test the model


```bash
# train semantic segmentation models
python train.py --work-dir work_dir1 configs/det/knet/knet_s3_r101_dcn-c3-c5_fpn_ms-3x_coco-panoptic.py 
#环境配制建议：
mmcv                                     2.1.0
mmdet                                    3.3.0
mmengine                                 0.10.4 

# train  instance segmentation models
python train.py --work-dir work_dir3 configs/det/knet/knet_s3_r50_fpn_1x_coco-panoptic.py
#环境配制建议：
mmcv-full                                1.5.0
mmengine                                 0.10.4 

```
* PARTITION: the slurm partition you are using
* CHECKPOINT: the path of the checkpoint downloaded from our model zoo or trained by yourself
* WORK_DIR: the working directory to save configs, logs, and checkpoints
* CONFIG: the config files under the directory configs/
* JOB_NAME: the name of the job that are necessary for slurm

## 4.Projects in OpenMMLab
[MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
[MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
[MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
[MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.

