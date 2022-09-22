# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class FewNet(SingleStageTextDetector):
    """
    This class for implementing FewNet text detector: Few Could Be 
    Better Than All: Feature Sampling and Grouping for Scene Text Detection

    [http://arxiv.org/abs/2203.15221]
    """
