# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseTextDetHead
from .db_head import DBHead
from .drrg_head import DRRGHead
from .fce_head import FCEHead
from .pan_head import PANHead
from .pse_head import PSEHead
from .textsnake_head import TextSnakeHead
from .fewnet_head import FewNetHead, FeatureGrouping, FeatureSampling

__all__ = [
    'PSEHead', 'PANHead', 'DBHead', 'FCEHead', 'TextSnakeHead', 'DRRGHead',
    "FewNetHead", "FeatureGrouping", "FeatureSampling", 
    'BaseTextDetHead', 
]
