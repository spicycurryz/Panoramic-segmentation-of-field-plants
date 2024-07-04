# MODEL_ZOO

* 这个文件包含了我们在phebobench数据集上训练完成的模型，所有模型都使用了NVIDIA V100-32G显卡训练了50000次,写于下面表格中方便下载

* 除OneFormer外所有模型使用的Backbone都是[Resnet-50](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl),OneFormer使用Swin_Transformer作为Backbone，其预训练模型可在OneFormer@PhenoBench文件夹下找到

| 模型                | mIoU    | PQ      | SQ      | RQ      |
|---------------------|---------|---------|---------|---------|
| [Mask2Former](https://pan.baidu.com/s/1ulhyH3n35MTfRh70dq9y9Q)         | 82.9086 | 75.157  | 89.124  | 83.854  |
| [MP-Former](https://pan.baidu.com/s/13P_DofZFBzir851u3tXefA)           | 86.4577 | 70.636  | 84.356  | 83.199  |
| [OneFormer](https://pan.baidu.com/s/1D4lsKFpuGl6B5TGyBYlkCA?pwd=wsac)           | 87.841 | 78.260   | 87.039  | 89.636  |
| [K-net](https://pan.baidu.com/s/18-P3JCXrKGZi8-0JRY2lfw)               | 79.5675 | 67.4573 | 80.5825 | 79.3683 |
| [Panoptic-DeepLab](https://pan.baidu.com/s/1P6sOEf9fws3Nr467RxfzmA)    | 86.5241 | 82.025  | 90.762  | 90.189  |
