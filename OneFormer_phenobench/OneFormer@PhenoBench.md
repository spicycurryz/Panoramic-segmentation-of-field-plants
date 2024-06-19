# OneFormer@PhenoBench
## 1.Installation
项目环境的配置和有关库的安装参考[OneFormer环境配置教程](https://github.com/SHI-Labs/OneFormer/blob/main/INSTALL.md).其中WANDB需要提前注册账号
## 2.Dataset Preparation
官方项目中提供了ADE20K, Cityscapes and COCO 2017三个数据集的配置，这里给出PhenoBench数据集Plants部分的准备方法。
- 切换目录到data_processing
- 运行下面的代码生成数据集：
   ```python
   python prepare_dataset_plants.py --dataset-folder PHENOBENCH_PATH
   python prepare_coco_semantic_annos_from_panoptic_annos_plants.py 
   ```
  其中`PHENOBENCH_PATH`需要换成自定义的数据集。</br>
- 利用生成的文件，运行`register_phenorob_plants_panoptic_annos_semseg.py`注册数据集

## 3.Training
- 鉴于训练资源有限，本项目采用1张V100训练，原项目采用的是8张A100，可根据自身算力对部分参数做适当调整
- 启动wandb
```python
pip install wandb
wandb login
```
- 默认情况下，训练过程中采用的评估都是evalution模式
- 在训练前首先注册数据集，然后运行下面的指令，以使用：
```python
python train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 1 \
    --config-file <path/to/your/config-file> \
    OUTPUT_DIR <path/to/output> WANDB.NAME <name>
```

## 4.Evaluation
- 设置task的值,从['semantic','instance','panoptic']中选择
- 使用下面的指令完成评估：
```python
python train_net.py --dist-url 'tcp://127.0.0.1:50164' \
    --num-gpus 1 \
    --config-file <path/to/your/config-file> \
    --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-checkpoint> \
    MODEL.TEST.TASK <task>
```

## 5.Inference Demo
- 第一步是选择模式：
```python
export task=panoptic
```
- 第二步是进行推理
```python
python demo.py --config-file <path/to/your/config-file> \
  --input <path-to-images> \
  --output <output-path> \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS <path-to-checkpoint>
```

## 6. 部分预训练模型下载地址：
可以在这里找到部分预训练模型：[model_zoo](https://github.com/SHI-Labs/OneFormer/tree/main?tab=readme-ov-file#results)