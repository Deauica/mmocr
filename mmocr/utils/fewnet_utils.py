# utils for fewnet
from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import copy
import functools
from shapely.geometry import Polygon as plg
from copy import deepcopy
import cv2

from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(self,
                 weight_boxes, cost_boxes_func, weight_logits, cost_logits_func):
        """Creates the matcher. In this class definition, cost_boxes can may contain the cost
        calculation for various box type, such as, bbox, rbox or bezier box.
        """
        super(HungarianMatcher, self).__init__()
        self.weight_boxes, self.cost_boxes_func = weight_boxes, cost_boxes_func
        self.weight_logits, self.cost_logits_func = weight_logits, cost_logits_func

    @torch.no_grad()
    def forward(self, outputs, targets, **kwargs):
        """ Performs the matching
        
        Args:
            outputs (Dict[str, torch.Tensor]):
               包括三个核心的 key, 分别是 boxes, angle, logits:
               "boxes": torch.Tensor, .shape == [batch_size, num_queries, 4], 表示 (x, y, w, h);
               "angle": torch.Tensor, .shape == [batch_size, num_queries, 1], 表示角度信息，采用 弧度制;
               "logits": torch.Tensor, .shape == [batch_size, num_queries, 1], 表示 对应 box 为 positive
               的 probability.
               
            targets (Dict[str, List[Union[Tensor, List[Tensor]]]]):
               包括 两个核心的 key, 分别是 boxes, angle, 也可能包括 labels:
               "boxes": torch.Tensor 类型，维度信息为 [num_target_boxes, 4], 表示 (x, y, w, h);
               "angle": torch.Tensor 类型，维度信息为 [num_target_boxes], 表示每一个 box 的角度信息,
                         采用的单位是 弧度制;
               "labels": Optional, torch.Tensor 类型，维度信息为 [num_target_boxes]。对于 二分类的
                         任务来看，这里可以不生成。

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
                
        Notes:
            - raw angle of outputs is (0, 1) scope, so we need proper pre-process.
        """
        # outputs' order is same as targets
        if kwargs.get("debug_same", False):
            indices = []
            for tgt_boxes in targets["boxes"]:
                indices.append(
                    (range(len(tgt_boxes)), range(len(tgt_boxes)))
                )
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices]
        
        # pre-process for outputs["angle"]
        bs, num_queries = outputs["boxes"].shape[:2]
        
        # step 1. obtain out_{boxes, angle, logits}
        out_logits = outputs["logits"].flatten(0, 1).sigmoid()  # [bs * num_queries, 1]
        
        out_boxes = outputs["boxes"].flatten(0, 1)  # [bs * num_queries, 4], 4 for only simple bbox
        out_angle = outputs["angle"].flatten(0, 1)  # [bs * num_queries, 1], 1 for only angle
        out_boxes = torch.cat([out_boxes, out_angle], dim=1)  # [bs * num_queries, 5]
        
        # step 2. obtain tgt_{boxes, angle, logits}
        tgt_boxes = torch.cat(targets["boxes"], dim=0)  # [num_tgt_boxes_batch, 4] -- normalized
        tgt_angle = torch.cat(targets["angle"], dim=0)  # [num_tgt_boxes_batch, 1]
        tgt_boxes = torch.cat([tgt_boxes, tgt_angle], dim=1)  # [num_tgt_boxes_batch, 5]
        
        tgt_labels = (
            torch.cat(targets["label"], dim=0) if "label" in targets else
            torch.full_like(tgt_angle, 1)
        )
        
        # step 3. generate cost matrix
        cost_logits = self.cost_logits_func(out_logits, tgt_labels)
        cost_boxes = self.cost_boxes_func(out_boxes, tgt_boxes)  # [bs * num_queries, num_tgt_boxes_batch]
        cost_matrix = (  # [bs * num_queries, num_tgt_boxes_batch]
                self.weight_logits * cost_logits + self.weight_boxes * cost_boxes)
        cost_matrix = cost_matrix.reshape([bs, num_queries, -1]).cpu()
        
        # step 4. perform hungarian algo
        sizes = [len(_) for _ in targets["boxes"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices], cost_matrix.split(sizes, -1)


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.
    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).
    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def postprocess(distance, fun='log1p', tau=1.0, reduction="none"):
    """Convert distance to loss.
    Args:
        distance (torch.Tensor)
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        reduction (str, optional): Defaults to none, added by mahy
    Returns:
        loss (torch.Tensor)
    """
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance.clamp(1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        distance = 1 - 1 / (tau + distance)
    else:
        pass
    
    if reduction == "none":
        return distance
    elif reduction == "sum":
        return torch.sum(distance)
    elif reduction == "mean":
        return torch.mean(distance)
    else:
        raise ValueError("Your reduction in gwd_loss.post_process is: {}".format(reduction))


def gwd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True,
             reduction="none"):
    """Gaussian Wasserstein distance loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.
        
    Returns:
        loss (torch.Tensor)
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau, reduction=reduction)


def scaled_preprocess(preds, targets):
    assert preds.shape == targets.shape and preds.shape[-1] == 5, (
        "Your preds.shape and targets.shape: {}, {}".format(preds.shape, targets.shape)
    )
    target_areas = targets[:, 2] * targets[:, 3]  # [N]
    target_areas = torch.sqrt(target_areas)  # acquire the sqrt of area
    targets[:, 2] = targets[:, 2] / target_areas
    targets[:, 3] = targets[:, 3] / target_areas
    preds[:, 2] = preds[:, 2] / target_areas
    preds[:, 3] = preds[:, 3] / target_areas
    return preds, targets


class GDLoss(nn.Module):
    """Gaussian based loss.
    Args:
        loss_type (str):  Type of loss.
        representation (str, optional): Coordinate System.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        alpha (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    Returns:
        loss (torch.Tensor)
    """
    BAG_GD_LOSS = { 'gwd': gwd_loss, }
    BAG_PREP = { 'xy_wh_r': xy_wh_r_2_xy_sigma }

    def __init__(self,
                 loss_type,
                 representation='xy_wh_r',
                 fun='log1p',
                 tau=0.0,
                 alpha=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 need_scaled_gwd=False,   # False for scaled due to the convergence issue
                 **kwargs):
        super(GDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'none', 'sqrt']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = self.BAG_PREP[representation]
        self.fun, self.tau, self.alpha = fun, tau, alpha
        self.reduction, self.loss_weight = reduction, loss_weight 
        self.need_scaled_gwd = need_scaled_gwd
        self.kwargs = kwargs

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)
        
        if self.need_scaled_gwd:
            pred, target = scaled_preprocess(pred, target)
        
        pred = self.preprocess(pred)
        target = self.preprocess(target)

        return self.loss(
            pred, target,
            fun=self.fun, tau=self.tau, alpha=self.alpha,
            reduction=reduction, **_kwargs) * self.loss_weight


# for box utils
def poly2obb_np(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_np_oc(polys)
    elif version == 'le135':
        results = poly2obb_np_le135(polys)
    elif version == 'le90':
        results = poly2obb_np_le90(polys)
    else:
        raise NotImplementedError
    return results


def poly2obb_np_le135(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))
    if edge1 < 2 or edge2 < 2:
        return
    
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))
    angle = norm_angle(angle, 'le135')
    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return x_ctr, y_ctr, width, height, angle


def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a

def poly2obb_np_oc(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
    if w < 2 or h < 2:
        return
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    return x, y, w, h, a


def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')


def obb2poly(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly_oc(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[:, 0]
    y = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    a = rboxes[:, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)


def obb2poly_le135(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons with Tensor.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


# for gaussian target
def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.
    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.
    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)
    
    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()
    
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.
    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.
    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.

    Notes:
        When two gaussian kernel meet, max value will be preserved.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(
        radius, sigma=diameter, dtype=heatmap.dtype, device=heatmap.device)
    
    x, y = center
    
    height, width = heatmap.shape[:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                      radius - left:radius + right]
    out_heatmap = heatmap
    out_heatmap[y - top:y + bottom, x - left:x + right] = (
        torch.max(masked_heatmap, masked_gaussian * k)
    )
    return out_heatmap
