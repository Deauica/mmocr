# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np
import copy
import functools
from shapely.geometry import Polygon as plg
from mmocr.utils.fewnet_utils import (HungarianMatcher, GDLoss)
from mmocr.utils.typing import DetSampleList

def cost_logits_func(out_prob: Tensor, tgt_labels: Tensor):
    """Currently, this utilizes bce_loss directly for debug
    
    Args:
        out_prob (torch.Tensor): .shape == [bs * num_queries, 1]
        tgt_labels (torch.Tensor): .shape == [num_tgt_boxes_batch]
        
    Returns:
        cost_logits (torch.Tensor): .shape == [bs * num_queries, num_tgt_boxes_batch]
        
    Notes:
        这里后面从 [bs * num_queries, 1] 到 [bs * num_queries, num_tgt_boxes_batch] 的扩展
        是 直接复制的结果。
    """
    if len(out_prob.shape) < 2:
        out_prob = out_prob.unsqueeze(dim=-1)
    cost_logits = F.binary_cross_entropy(out_prob, torch.ones_like(out_prob), reduction="none")
    return cost_logits.tile([1, len(tgt_labels)])  # [bs * num_queries, num_tgt_boxes_batch]

def cost_rbox_func(out_boxes: Tensor, tgt_boxes: Tensor, **kwargs):
    """Currently, this function utilize GWD Loss as matcher metrics directly for debug
    
    Args:
        out_boxes (Tensor): a Tensor with shape (bs * num_queries, 5)
        tgt_boxes (Tensor): a Tensor with shape (sum_num_tgt_boxes, 5)
        
    Returns:
        cost_mat (Tensor): a two-dimensional Tensor of shape (bs * num_queries, sum_num_tgt_boxes)
        
    Notes:
        - Default loss_rbox_func should be gwd_loss. However, you can specified a callable object
            for loss_rbox_func from key-value parameter.
    """
    assert out_boxes.shape[-1] == tgt_boxes.shape[-1] == 5, (
        "shape of out_boxes and tgt_boxes: {}, {}".format(out_boxes.shape, tgt_boxes.shape)
    )
    tiled_H, tiled_W = out_boxes.shape[0], tgt_boxes.shape[0]
    tiled_out_boxes = torch.tile(
        out_boxes.unsqueeze(dim=1), dims=(1, tiled_W, 1)
    )  # ensure the element in each row be same
    tiled_tgt_boxes = torch.tile(
        tgt_boxes.unsqueeze(dim=0), dims=(tiled_H, 1, 1)
    )  # ensure the element in each col be same
    
    if "loss_rbox_func" not in kwargs:
        loss_rbox_func = GDLoss(
            loss_type=kwargs.get("rbox_loss_type", "gwd"),
            fun=kwargs.get("rbox_fun", "log1p"),
            tau=kwargs.get("rbox_tau", 1.0),
            reduction="none"  # 这里仍然采用 reduction
        )
    else:
        loss_rbox_func = kwargs["loss_rbox_func"]
    
    cost_mat = loss_rbox_func(
        tiled_out_boxes.flatten(0, 1), tiled_tgt_boxes.flatten(0, 1),
        reduction_override="none"
    ).reshape(tiled_H, tiled_W)  #
    return cost_mat

class FewNetModuleLoss(nn.Module):
    def __init__(
        self, 
        weight_cost_logits=1.0, weight_cost_boxes=1.0,
        weight_loss_score_map=1.0, weight_loss_logits=1.0, weight_loss_rbox=1.0,
        max_target_num=100, angle_version="le135",
        rbox_loss_type="gwd", rbox_fun="log1p", rbox_tau=3.0,
        need_norm_boxes=True
    ) -> None:
        super(FewNetModuleLoss, self).__init__()
        self.weight_cost_logits, self.weight_cost_boxes = (
            weight_cost_logits, weight_cost_boxes
        )
        self.weight_loss_score_map, self.weight_loss_logits, self.weight_loss_rbox = (
            weight_loss_score_map, weight_loss_logits, weight_loss_rbox
        )
        # specify the loss func, loss_logits_func, loss_rbox_func
        self.loss_logits_func = self.loss_logits
        self.need_norm_boxes = need_norm_boxes  # whether need normalized boxes coordinates ?
        self.rbox_fun, self.rbox_loss_type, self.rbox_tau, self.rbox_reduction = (
            rbox_fun, rbox_loss_type, rbox_tau, "sum"
        )

        self.loss_rbox_func = GDLoss(
                loss_type=self.rbox_loss_type, fun=self.rbox_fun, tau=self.rbox_tau,
                reduction="sum"  # 这里仍然采用 reduction
        )
        # specify the cost
        self.cost_logits_func, self.cost_boxes_func = (
            cost_logits_func,
            functools.partial(cost_rbox_func, loss_rbox_func=self.loss_rbox_func)
        )

        # During matching, for simply consider cost_boxes while ignoring cost_logits
        # for scene text detection.
        self.matcher = HungarianMatcher(
            self.weight_cost_boxes, self.cost_boxes_func,
            self.weight_cost_logits, self.cost_logits_func,
        )

        self.max_target_num = max_target_num
        self.angle_version = angle_version
        self.angle_minmax = torch.as_tensor(dict(
            oc=(0, np.pi / 2), le135=(-np.pi / 4, np.pi * 3 / 4),
            le90=(-np.pi / 2, np.pi / 2)
        )[self.angle_version])

    def forward(self, outs, targets, *args, **kwargs):
        """ Calculate the loss_dict baesd on outs and targets. 

        Args: 
            outs (OrderedDict): an dict object with these entries:
                - score_map: List of Tensor with shape [B, Hi, Wi], Hi, Wi is the height and 
                    width for i-th level; 
                - logits: Tensor of dim [bs, num_selected_features, 1] with the classification 
                    logits
                - boxes: Tensor of dim [bs, num_selected_features, 4] with the normalized
                    boxes coordinates (cx, cy, w, h)
                - angle: Tensor of dim [bs, num_selected_features, 1] with the angle for each 
                    corresponding boxes, format should be `le135`
            targets (DetSampleList): 

        Returns: 
            loss_dict (OrderedDict): a dict with these entries:
              - loss_score_map: loss for score_map 
              - loss_logits: loss for logits 
              - loss_rbox: loss for rbox 

        Note: 
            - Since label is not necessary for text detection, so we ignore the "label" key 
                in `targets`.
        """
        pass 

    def get_targets(self, targets: DetSampleList, *args, **kwargs):
        """ Generate targets to calculate fewnet loss based on raw targets

        Args: 
            targets (DetSampleList): List of DetSample object with these attributes: 
                - gt_polygons: List[ndarray(2k, )]
                - gt_ignored: torch.Tensor(N, )

        Returns: 
            results (OrderedDict): an OrderedDict object with these entries:
                - score_map: 
                - score_mask: 
                - boxes:
                - angle: 
                - num_tgt_boxes: 
        """
        pass 
