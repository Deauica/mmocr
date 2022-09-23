# Copyright (c) OpenMMLab. All rights reserved.

from .base import BaseTextDetPostProcessor
from mmocr.registry import MODELS


@MODELS.register_module()
class FewNetPostprocessor(object):
    pass
