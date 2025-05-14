# _*_ coding:utf-8 _*_
from typing import List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .custom import CustomDataset

RESISC45 = ('airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
            'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
            'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
            'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection',
            'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park',
            'mountain', 'overpass', 'palace', 'parking_lot', 'railway',
            'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway',
            'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium',
            'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland')

AID = ('Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center',
       'Church', 'Commercial', 'DenseResidential', 'Desert', 'Farmland', 'Forest',
       'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking',
       'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River',
       'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct')

@DATASETS.register_module()
class Resisc45(CustomDataset):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': RESISC45}


@DATASETS.register_module()
class AID(CustomDataset):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': AID}
