# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np
import functools
from shapely.geometry import Polygon as plg

from mmocr.utils.fewnet_utils import (
    HungarianMatcher, GDLoss, poly2obb_np, gen_gaussian_target, obb2poly
)
from mmocr.utils.typing import DetSampleList, List, Dict, Tuple
from mmocr.registry import MODELS
from mmengine.structures.instance_data import InstanceData
from mmocr.utils.polygon_utils import rescale_polygons, poly_iou
from .db_module_loss import DBModuleLoss

import cv2


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


@MODELS.register_module()
class FewNetModuleLoss(nn.Module):
    """
    Loss for FewNet Module.
    
    Args:
        weight_cost_{logits, boxes}: cost for the weight of HungarianMatcher.
        weight_loss_{score_map, logits, rbox}: loss for the weight of loss.
        
        max_target_num: utilized for gen_target_matched. intermediate variable.
        angle_version: version for current angle.
        rbox_loss_type, rbox_fun, rbox_tau: gwd_loss arguments.
        need_scaled_gwd: whether to utilized scaled gwd loss.
        
        strides: strides for the feature map. utilized in target generation.
        need_norm_boxes: True for norm boxes in targets and false otherwise.
        min_radius_limit, coef_gaussian, max_num_gau_center: hyper-parameter for gaussian targets.
        fg_value, bg_value: foreground and background default value.
    """
    def __init__(
            self,
            weight_cost_logits=1.0, weight_cost_boxes=1.0,
            weight_loss_score_map=1.0, weight_loss_logits=1.0, weight_loss_rbox=1.0,
            max_target_num=100, angle_version="le135",
            rbox_loss_type="gwd", rbox_fun="log1p", rbox_tau=3.0,
            need_norm_boxes=False, need_scaled_gwd=False,
            strides=(8, 16, 32),
            min_radius_limit=1, coef_gaussian=1, max_num_gau_center=100,
            fg_value=0.7, bg_value=0
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
        self.need_scaled_gwd = need_scaled_gwd
        
        self.loss_rbox_func = GDLoss(
            loss_type=self.rbox_loss_type, fun=self.rbox_fun, tau=self.rbox_tau,
            need_scaled_gwd=self.need_scaled_gwd, reduction="sum"
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

        self.max_target_num = max_target_num  # utilized for gen_targets_matched
        self.angle_version = angle_version
        self.angle_minmax = torch.as_tensor(dict(
            oc=(0, np.pi / 2), le135=(-np.pi / 4, np.pi * 3 / 4),
            le90=(-np.pi / 2, np.pi / 2)
        )[self.angle_version])
        
        # targets generator
        self.get_targets = MakeFewNetTargets(
            angle_version=angle_version, need_norm_boxes=need_norm_boxes,
            bg_value=bg_value, fg_value=fg_value,
            min_radius_limit=min_radius_limit, coef_gaussian=coef_gaussian,
            max_num_gau_center=max_num_gau_center, strides=strides
        )

    def forward(self, outs: OrderedDict, targets: DetSampleList, *args, **kwargs) -> Dict:
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
                    
            targets (DetSampleList): List of DetSample object with these attributes:
                - data_fields: gt_instances (InstanceData) currently with these attributes:
                  - ignored: torch.Tensor with shape (N,) representing whether current text instance
                             should be ignored.
                  - polygons: np.ndarray with shape [N, 2k], representing the polygonal
                             boundary for each text instance.
                  - bboxes: torch.Tensor with shape [N, 4], representing the bbox annotation for
                             each text instance.

                - meta_fields: meta-information currently contains these attributes:
                  - img_shape: shape of current image.
                  - orig-shape: shape of original image.
                  - img_path: path to this image.
                  - batch_input_shape: all image shape for this batch.

        Returns: 
            loss_dict (OrderedDict): a dict with these entries:
              - loss_score_map: loss for score_map 
              - loss_logits: loss for logits 
              - loss_rbox: loss for rbox 

        Note: 
            - Since label is not necessary for text detection, so we ignore the "label" key 
                in `targets`.
        """
        loss_dict = OrderedDict()
        
        # step 0. prepare targets and outputs
        img_shape = targets[0].img_shape
        targets = self.get_targets(targets)
        outs = self.prepare_outs(outs, img_shape=img_shape)
        
        # step 1. loss for score_map
        out_score_maps, tgt_score_maps = outs.pop("score_map"), targets.pop("score_map")
        tgt_score_masks = targets.pop("score_mask", None)
        loss_score_map = self.loss_score_map(out_score_maps, tgt_score_maps, tgt_score_masks)
        loss_dict.update(
            loss_score_map=self.weight_loss_score_map * loss_score_map)

        # step 2. matching between outputs and targets
        B, num_selected_features = outs["boxes"].shape[:2]
        indices, cost_matrix = self.matcher(outs, targets)
        outputs_matched, _ = self.gen_output_matched(    # [str, [num_tgt_boxes, ...]]
            outs, indices, num_selected_features=num_selected_features)
        targets_matched = self.gen_target_matched(targets, indices)
        
        # step 3. loss for rotated boxes -- pay attention to the self.need_scaled_gwd
        N_r = outputs_matched["boxes"].shape[0]
        outputs_rbox = torch.cat(  # [num_tgt_boxes, 5]
            [outputs_matched["boxes"], outputs_matched["angle"]], dim=-1
        )
        tgt_rbox = torch.cat(  # [num_tgt_boxes, 5]
            [targets_matched["boxes"], targets_matched["angle"]], dim=-1
        )
        loss_rbox = self.loss_rbox_func(outputs_rbox, tgt_rbox,
                                        reduction_override=self.rbox_reduction)
        loss_dict.update(
            loss_rbox=self.weight_loss_rbox * loss_rbox / N_r)
        
        # step 4. loss for logits
        loss_logits = self.loss_logits_func(
            outputs_matched["logits"], outputs_rbox, tgt_rbox,
            poly_iou_threshold=0.3
        )
        N = outs["logits"].shape[0] * outs["logits"].shape[1]  # B * num_selected_features
        loss_dict.update(
            loss_logits=self.weight_loss_logits * loss_logits / N)
        
        return loss_dict
    
    def prepare_outs(self, outs: OrderedDict, img_shape: Tuple[int, int],
                     *args, **kwargs) -> OrderedDict:
        r""" Prepare outputs for the loss calculation. Currently, two things will be performed:
        1. scale the normalized scale to proper scope;
        2. scale boxes to proper scope.
        """
        def _scale(src, amin, amax):
            return amin + src * (amax - amin)

        imgH, imgW = img_shape
        scaleH, scaleW = (1, 1) if self.need_norm_boxes else (imgH, imgW)
        _coef = torch.ones_like(outs["boxes"])
        _coef[:, :, 0:-1:2] = _coef[:, :, 0:-1:2] * scaleW
        _coef[:, :, 1::2] = _coef[:, :, 1::2] * scaleH
        _out_boxes = outs["boxes"] * _coef
        _out_angle = _scale(outs["angle"], self.angle_minmax[0], self.angle_minmax[1])
        outs.update(
            boxes=_out_boxes, angle=_out_angle
        )
        return outs

    def gen_output_matched(self, outputs, indices, num_selected_features, ratio=1):
        """
        Returns:
            matched_t (Dict[str, Tensor]): a dict containing at least "bbox", "logits", "angle".
              dim of Tensor is [num_tgt_boxes, ...]

            unmatched_t (Dict[str, Tensor]): a dict containing the same key as matched_t, but for
              unmatched elements.

        Notes:
            - Current the ratio is pos : neg = 1 : 3
        """
        assert "score_map" not in outputs, (
            "Call this function after self.matcher please"
        )
        sizes = [len(elem[0]) for elem in indices]
        batch_idx = torch.cat([torch.full((s,), i) for i, s in enumerate(sizes)])
        src_idx = torch.cat([src_indice for (src_indice, _) in indices])
        
        # unmatched
        t = torch.ones(num_selected_features)
        src_idx_unmatched_list = [
            torch.nonzero(
                torch.scatter(t.to(src_indice.device), 0, src_indice, 0)
            ).flatten()[: sizes[i] * ratio]
            for i, (src_indice, _) in enumerate(indices)
        ]
        src_idx_unmatched = torch.cat(src_idx_unmatched_list)
        unmatched_sizes = [len(t) for t in src_idx_unmatched_list]
        batch_idx_unmatched = torch.cat(
            [torch.full((s,), i) for i, s in enumerate(unmatched_sizes)]
        )
        
        matched_t = OrderedDict()  # [num_tgt_boxes, ...]
        unmatched_t = OrderedDict()
        for k in outputs.keys():
            matched_t[k] = outputs[k][batch_idx, src_idx]
            unmatched_t[k] = outputs[k][batch_idx_unmatched, src_idx_unmatched]
        return matched_t, unmatched_t

    def gen_target_matched(self, targets, indices, keys=("boxes", "angle")):
        """Generate matched targets based on `targets` and `indices`.
        Args:
            targets (Dict[str, Any]): source targets.
            indices (List[Tuple]): returned value of self.matcher.
            keys (Tuple[str]): containing the keys to be padded and matched.

        Returns:
            t (Dict[str, Tensor]):  a dict containing at least "bbox", "angle".
              dim of Tensor is [num_tgt_boxes, ...]
        """
        assert "score_map" not in targets, (
            "Call this function after self.matcher please"
        )
        _targets = OrderedDict()
        for k in targets.keys():
            if k not in keys:  # no operation is needed
                continue
        
            _targets[k] = torch.stack([
                F.pad(input=target, pad=(0, 0, 0, self.max_target_num - target.shape[0]))
                for target in targets[k]
            ])
    
        # Following code is similar to `self.gen_output_matched`
        sizes = [len(elem[1]) for elem in indices]
        batch_idx = torch.cat([torch.full((s,), i) for i, s in enumerate(sizes)])
        tgt_idx = torch.cat([tgt_indice for (_, tgt_indice) in indices])
    
        t = OrderedDict()  # [num_tgt_boxes, ...]
        for k in _targets.keys():
            try:
                t[k] = _targets[k][batch_idx, tgt_idx]
            except Exception as e:
                import pdb
                pdb.set_trace()
            
        return t

    def loss_logits(self, out_matched_logits, out_matched_boxes, tgt_matched_boxes,
                    poly_iou_threshold=0.3):
        """ First version for loss_logits

        Args:
            out_matched_logits (Tensor): tensor with dim [bs * num_queries, 1]
            out_matched_boxes (Tensor): tensor with dim [bs * num_queries, 5]
            tgt_matched_boxes (Tenesor): tensor with dim [bs * num_queries, 5]
            poly_iou_threshold (float): threshold used to check whether current box should be positive.
        """
        assert len(out_matched_logits) == len(out_matched_boxes) == len(tgt_matched_boxes), (
            "len of out_matched_logits, out_matched_boxes, out_matched_boxes should be the same",
            "However, your results: {}".format(
                [len(out_matched_logits), len(out_matched_boxes), len(tgt_matched_boxes)]
            )
        )
    
        target_labels = self.gen_target_labels(
            out_matched_boxes, tgt_matched_boxes, poly_iou_threshold=poly_iou_threshold
        ).to(out_matched_logits.device)
        return F.binary_cross_entropy(
            out_matched_logits.flatten(), target_labels, reduction="sum"
        )

    @torch.no_grad()
    def gen_target_labels(self,
                          out_matched_boxes, tgt_matched_boxes, poly_iou_threshold=0.3):
        r""" Generate target labels utilized for loss_logits

        Args:
            out_matched_boxes (torch.Tensor): tensor of dim [bs * num_tgt_boxes, 8]
            tgt_matched_boxes (torch.Tensor): tensor of dim [bs * num_tgt_boxes, 8]
            poly_iou_threshold (float): float threshold
        """
    
        def _to_np(t):
            return t.reshape(-1, 2).cpu().numpy()
    
        if out_matched_boxes.shape[-1] == tgt_matched_boxes.shape[-1] == 5:
            """
            transform (cx, cy, w, h, theta) to quad version,
            """
            out_matched_boxes = obb2poly(out_matched_boxes, self.angle_version)
            tgt_matched_boxes = obb2poly(tgt_matched_boxes, self.angle_version)
    
        assert out_matched_boxes.shape[-1] >= 8 and tgt_matched_boxes.shape[-1] >= 8, (
            "your out_matched_boxes.shape and tgt_matched_boxes.shape are: {}, {}".format(
                out_matched_boxes.shape, tgt_matched_boxes.shape
            )
        )
    
        target_labels = [
            1 if poly_iou(
                plg(_to_np(out_matched_box)), plg(_to_np(tgt_matched_box))
            ) > poly_iou_threshold else 0
            for out_matched_box, tgt_matched_box in zip(out_matched_boxes, tgt_matched_boxes)
        ]
        target_labels = torch.as_tensor(target_labels,
                                        dtype=torch.float, device=out_matched_boxes.device)
        return target_labels

    def loss_score_map(self,
                       out_score_maps: List[Tensor], tgt_score_maps: List[Tensor],
                       tgt_score_masks: List[Tensor] = None,
                       *args, **kwargs):
        """ Calculate loss for score_map
        Args:
            out_score_maps (Tensor): List of out_score_map with shape [B, Hi, Wi].
            tgt_score_maps (Tensor): List of tgt_score_map with shape [B, Hi, Wi].
            tgt_score_masks(Tensor): List of tgt_score_mask with shape [B, Hi, Wi] if not None.

        Returns:
            loss (Tensor): Scalar that represents the loss for score_map.

        Notes:
            - Simply smooth l1 loss can not be directly used due to the extreme ratio of neg : pos.
            - In order to calculate proper ratio, we only choose pos : neg = 1 : 3.
            - N_f should be also changed.
            - Currently, no smooth l1 loss is utilized.
        """
        assert tgt_score_masks is not None, (
            "Please check your .get_targets "
            "since tgt_score_masks should not be None"
        )
        N_f, loss_sum = 0, 0
        ratio = 3  # pos : neg = 1 : ratio
    
        for _, (out_score_map, tgt_score_map, tgt_score_mask) in enumerate(
                zip(out_score_maps, tgt_score_maps, tgt_score_masks)):
            tgt_score_mask = tgt_score_mask.float()
            positive_mask, negative_mask = tgt_score_mask, 1 - tgt_score_mask
            positive_count = int(positive_mask.sum())
            negative_count = positive_count * ratio
            # loss = F.smooth_l1_loss(  # [B, Hi, Wi]
            #     out_score_map, tgt_score_map, reduction="none")
            loss = F.binary_cross_entropy(out_score_map, tgt_score_map, reduction="none")
            positive_loss, negative_loss = loss * positive_mask, loss * negative_mask
            negative_loss, _ = torch.topk(
                negative_loss.flatten(), min(torch.numel(negative_loss), negative_count)
            )
        
            N_f += positive_count + negative_count
            loss_sum += positive_loss.sum() + negative_loss.sum()
    
        return loss_sum / N_f
    

class MakeFewNetTargets(object):
    def __init__(
            self,
            angle_version="le135", need_norm_boxes=True,
            bg_value=0, fg_value=0.7,
            min_radius_limit=0, coef_gaussian=1, max_num_gau_center=50,  # gaussian related
            strides=(8, 16, 32)
    ):
        self.angle_version = angle_version
        self.need_norm_boxes = need_norm_boxes
        self.bg_value, self.fg_value = bg_value, fg_value
        self.strides = strides
        self.min_radius_limit, self.coef_gaussian, self.max_num_gau_center = (
            min_radius_limit, coef_gaussian, max_num_gau_center
        )
        
        # initialize the corresponding resizer for different strides
        # resizer can be a paritial methods for mmocr.utils.polygon_utils.rescale_polygon
        self.resizers = [
            functools.partial(rescale_polygons, scale_factor=(stride, stride), mode="div")
            for stride in self.strides
        ]
    
    def __call__(self, targets: DetSampleList, *args, **kwargs):
        return self.get_targets(targets)

    def get_targets(self, targets: DetSampleList, *args, **kwargs):
        """ Generate targets to calculate fewnet loss based on raw targets

        Args:
            targets (DetSampleList): same as `targets` in FewNetModuleLoss.forward.

        Returns:
            results (OrderedDict): an OrderedDict object with these entries:
                - score_map: List[torch.Tensor] with each Tensor's shape should be [B, Hi, Wi].
                             Each point in score_map represents the significance for this point.
                             The length for score_map will the number of feature levels.
                - score_mask: List[torch.Tensor]. score_mask is only utilized to obtain the
                             loss for predicted significance map.
                - boxes: List[torch.Tensor] with each element's shape should be [N, 4]. Each element
                             represents the boxes of the rotated boxes for one sample.
                - angle: List[torch.Tensor] with each element's shape should be [N, 1]. each element
                             represents the angle of the rotated boxes for one sample.

        Notes:
            - len(score_map) != len(boxes) by default due to the distinctive meaning for each element.
            - Currently, only the sample with `ignored == False` will be considered.
        """
        results = OrderedDict()
    
        # step 0. obtaining gt_instances for simplicity in the following code snippet.
        gt_instances = [target.gt_instances for target in targets]
    
        # step 1. collect single targets
        boxes, angles, _score_maps, _score_masks = [], [], [], []
        for gt_instance in gt_instances:
            results_single = self.get_single_sample_target(
                gt_instance, img_shape=targets[0].img_shape
            )
            boxes.append(results_single["boxes_single"])
            angles.append(results_single["angle_single"])
            _score_maps.append(results_single["score_maps_single"])
            _score_masks.append(results_single["score_masks_single"])
            
        # step 2. perform transformation for score_maps and score_masks
        score_maps, score_masks = [], []
        for score_map_per_level in zip(*_score_maps):
            score_maps.append(torch.stack(score_map_per_level, dim=0))
        for score_mask_per_level in zip(*_score_masks):
            score_masks.append(torch.stack(score_mask_per_level, dim=0))
        
        results.update(
            boxes=boxes, angle=angles, score_map=score_maps,
            score_mask=score_masks
        )
        return results

    def get_single_sample_target(
            self,
            gt_instances: InstanceData, img_shape: Tuple[int, int],
            *args, **kwargs
    ) -> OrderedDict:
        """ Generate target for single sample.

        Returns:
             score_maps_single (List[torch.Tensor]): Each Tensor object corresponds to the feature
               map for each level. Shape for the element of i-th level should be [Hi, Wi].
             score_masks_single (List[torch.Tensor]): Each Tensor object corresponds to the mask
               for the score_map.
             boxes_single (torch.Tensor): shape should be [N, 4].
             angle_single (torch.Tensor): shape should be [N, 1].
        """
        def _to_tensor(t):
            return torch.from_numpy(t).to(gt_instances["bboxes"].device)
        
        results_single = OrderedDict()
    
        # step 0. generate rotated boxes for single data sample.
        rboxes = []
        poly_valid_pos = torch.full([len(gt_instances)], True)
        
        for i, polygon in enumerate(gt_instances["polygons"]):
            rect = cv2.minAreaRect(polygon.reshape(-1, 2).astype(np.int32))
            polygon = cv2.boxPoints(rect)
            polygon = np.reshape(polygon, [8])
            rbox = poly2obb_np(polygon, version=self.angle_version)
            if rbox is None:
                poly_valid_pos[i] = False
            else:
                rboxes.append(rbox)
        rboxes = np.array(rboxes, dtype=np.float32).reshape([-1, 5])  # [cx, cy, w, h, theta]
        if self.need_norm_boxes:
            max_H, max_W = img_shape
            rboxes[:, 0:-1:2] = rboxes[:, 0:-1:2] / max_W  # increase robustness of indexing
            rboxes[:, 1::2] = rboxes[:, 1::2] / max_H
        boxes, angles = rboxes[:, :4], rboxes[:, -1:]
        boxes, angles = _to_tensor(boxes), _to_tensor(angles)
        
        # update gt_instances to filter out too small text instances
        gt_instances = gt_instances[poly_valid_pos]  # valid_pos, bool type
        assert len(gt_instances) == len(boxes)
        
        # step 1. score_map and score_mask for single data sample
        assert all(length % self.strides[-1] == 0 for length in img_shape), (
            "strides: {}, and your gt_instances.img_shape: {}".format(
                self.strides, img_shape
            ))
        score_maps, score_masks = [], []
        imgH, imgW = img_shape
        for resizer, stride in zip(self.resizers, self.strides):
            aug_canvas = np.full([imgH//stride, imgW//stride], self.bg_value, dtype=np.float32)
            aug_polys = resizer(gt_instances.polygons)
            score_map = self.gen_single_score_map(aug_canvas, aug_polys)
            
            score_map = _to_tensor(score_map)
            score_mask = score_map > self.bg_value  # greater than self.bg_value
            score_maps.append(score_map)
            score_masks.append(score_mask)
        
        results_single.update(
            score_maps_single=score_maps, score_masks_single=score_masks,
            boxes_single=boxes, angle_single=angles
        )
        return results_single

    def gen_single_score_map(self, canvas, polys):
        """
        Args:
             canvas (np.ndarray): ndarray object with shape [Hi, Wi].
             polys (Sequence[ndarray]): each ndarray's shape should be [-1, 2].
             
        Returns:
            score_map (np.ndarray): score map for this data sample.
        """
        # step 1. fill foreground with pre-defined fg_value
        polys = [
            poly.astype(np.int32).reshape(-1, 2) for poly in polys
        ]
        cv2.fillPoly(canvas, polys, self.fg_value)
    
        # step 2. generate gaussian candidate -- list(((x, y), radius))
        gaussian_candidates = []
        for poly in polys:  # 分别在 每一个 poly 中单独获取 随机点
            (x_min, y_min), (x_max, y_max) = np.min(poly, axis=0), np.max(poly, axis=0)
            poly[:, 0] = poly[:, 0] - x_min
            poly[:, 1] = poly[:, 1] - y_min
        
            # generate positive_mask -- tiny_coordinate
            tiny_H, tiny_W = y_max - y_min + 1, x_max - x_min + 1
            tiny_mask = np.zeros([tiny_H, tiny_W])
            cv2.fillPoly(tiny_mask, [poly], 1)  # 1 for positive, 0 for negative
            t_mask = np.random.binomial(
                n=1, p=0.5, size=tiny_mask.shape
            )
            positive_mask = t_mask * tiny_mask
        
            # generate distance array
            # [num_lines, dist_from_line_to_point(2D)] -> [2D]
            dist_point_lines = np.zeros([len(poly), *tiny_mask.shape])
            xs = np.arange(start=0, stop=tiny_W)  # x, ... width
            ys = np.arange(start=0, stop=tiny_H)  # y, ... height
            xs, ys = np.meshgrid(xs, ys, indexing="xy")  # meshgrid, xs - width, ys - height
            for i in range(len(poly)):
                j = (i + 1) % len(poly)
                point_1, point_2 = poly[i], poly[j]
                dist_point_line = self.point2line(xs, ys, point_1, point_2)
                dist_point_lines[i] = dist_point_line
            dist_point_lines = dist_point_lines.min(axis=0)  # [tiny_H, tiny_W]
            # Though the coordinates is different, However,
            # the value in dist_point_lines can also reflect the distance
            # from this point to line.
        
            # obtain first-step xs, ys, max_radius
            positive_point_lines = dist_point_lines * positive_mask  # [tiny_H, tiny_W]
            nk = min(np.sum(positive_mask).astype(np.int32), self.max_num_gau_center)
            nk_ind, nk_val = self.topk_by_partition(
                positive_point_lines.flatten(), nk, axis=0, ascending=False,
            )
            nk_xs, nk_ys = nk_ind % tiny_W, nk_ind // tiny_W  # nk_xs: width, nk_ys: height
            t_mask = nk_val > self.min_radius_limit
            nk_val = nk_val[t_mask]  # now nk_val only contain value greater than min_radius_limit
            nk_xs, nk_ys = nk_xs[t_mask], nk_ys[t_mask]  # check
        
            # generate center and radius
            # t_scale = 0.5 + np.random.rand(*nk_val.shape) * 0.5
            t_scale = np.ones_like(nk_val)
            poly_radius = np.ceil(
                self.min_radius_limit + t_scale * (nk_val - self.min_radius_limit)
            )
            poly_gau_candinates = [
                ((xs + x_min, ys + y_min), radius)  # radius should be integer
                for xs, ys, radius in zip(nk_xs, nk_ys, poly_radius.astype(np.int32))
            ]
            gaussian_candidates.extend(poly_gau_candinates)  # extend gaussian_candidates
        
        # step 3. generate single score map
        for (x, y), radius in gaussian_candidates:
            canvas = gen_gaussian_target(
                torch.as_tensor(canvas), (x, y), radius).cpu().numpy()
    
        return canvas

    @staticmethod
    def topk_by_partition(input, k, axis=None, ascending=True):
        """
        Inherited from: https://hippocampus-garden.com/numpy_topk/
        """
        if not ascending:
            input *= -1
        ind = np.argpartition(input, k - 1, axis=axis)  # use (k - 1) instead of k for extreme situation
        ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
        input = np.take_along_axis(input, ind, axis=axis)  # k non-sorted values
    
        # sort within k elements
        ind_part = np.argsort(input, axis=axis)
        ind = np.take_along_axis(ind, ind_part, axis=axis)
        if not ascending:
            input *= -1
        val = np.take_along_axis(input, ind_part, axis=axis)
        return ind, val
    
    # set the points2line
    point2line = staticmethod(DBModuleLoss._dist_points2line)


# TODO: Define hook to update FewNetModuleLoss.weight_loss_{XXX} per epoch
