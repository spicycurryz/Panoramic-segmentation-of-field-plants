# 为Mask2former准保coco格式的数据集

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

其中，train，val和test文件夹负责存放原始图片。

运行以下代码生成panoptic_{train,val}和panoptic_{train,val}.json,将PHENOBENCH_PATH替换为phenobench数据集的位置。

```bash
python datasets/prepare_dataset_plants.py --dataset-folder PHENOBENCH_PATH --output-folder PHENOBENCH_PATH
```

运行以下代码生成panoptic_semseg_{train,val}用于语义评估

```bash
python datasets/prepare_coco_semantic_annos_from_panoptic_annos_plants.py
```

运行以下代码生成instances_{train,val}.json用于实例评估。

```bash
python datasets/panoptic2detection_coco_format
```
