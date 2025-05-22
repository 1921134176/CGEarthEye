# _*_ coding:utf-8 _*_
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

cropland = dict(classes=('background', 'cropland'),palette=[[255, 255, 255], [255, 0, 0]])

@DATASETS.register_module()
class RsimageDataset(BaseSegDataset):
    METAINFO = cropland
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)