# Copyright (c) OpenMMLab. All rights reserved.
# from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from os.path import osp

@DATASETS.register_module()
class PlaneDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    METAINFO = dict(
        classes=("0",'1', '2', '3', '4'),
        palette=[[0,0,0],[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50]
                 ])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
