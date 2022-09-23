# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

from mmocr.models.textdet.heads import BaseTextDetHead
from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.utils.typing import DetSampleList

import torch 
from torch import nn
from collections import OrderedDict
import warnings


class CoordConv(nn.Module):
    pass 


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 max_pos_len=3000, pos_dim=None, num_dims=2,
                 embed_type="sin_cos", **kwargs):
        super(PositionalEmbedding, self).__init__()
        
        self.max_pos_len, self.pos_dim = max_pos_len, pos_dim
        self.num_dims = num_dims
        self.embed_type = embed_type
        self.kwargs = kwargs
        self.model_dim = self.kwargs.get("model_dim", self.pos_dim)
        self.embed_tables = self.build_embed_table()
        self.static = None  # whether pe_table will be updated by bp
    
    def forward(self, coords):
        """传入 坐标 的矩阵，并且 返回对应的 positional embedding.
        
        Args:
            coords (torch.Tensor): [num_dims, ...]. coords[i] corresponds to coordinates from the view of
              i-th dims.
              
        Returns:
            out (torch.Tensor): [..., pos_dim], positional embedding tensor.
        """
        assert coords.shape[0] == self.num_dims, (
            "Please check your coords, since the shape of coords: {}, "
            "However, the num_dims: {}".format(coords.shape, self.num_dims)
        )
        
        out = torch.zeros([*coords.shape[1:], self.pos_dim], device=coords.device)
        for i in range(coords.shape[0]):
            out += self.embed_tables[str(i)](coords[i])
        return out
        
    def build_embed_table(self):
        embed_tables = OrderedDict()
        d_model, position_bias = self.model_dim, 0
        if self.embed_type == "sin_cos":
            self.static = True  # 使用静态的 embedding table
            for k in range(self.num_dims):
                i_mat = torch.arange(start=0, end=self.pos_dim, step=2) + position_bias
                i_mat = torch.pow(10000., i_mat / d_model).reshape([1, -1])
                pos_mat = torch.arange(start=0, end=self.max_pos_len, step=1).reshape([-1, 1])
                
                table_weight = torch.zeros(
                    [self.max_pos_len, self.pos_dim], requires_grad=False,
                )
                table_weight[:, 0::2] = torch.sin(pos_mat / i_mat)  # 自动 broadcast
                table_weight[:, 1::2] = torch.cos(pos_mat / i_mat)
                embed_tables.update({
                    str(k): nn.Embedding.from_pretrained(table_weight)
                })
                position_bias += self.max_pos_len  # 对 position_bias 加以更新
        else:
            raise ValueError("Please check your embed type: {}".foramt(self.embed_type))
        
        return nn.ModuleDict(embed_tables)


class FeatureGrouping(nn.Module):
    """
    4 transformer encoder layers.
    """
    def __init__(self,
                 c=256,
                 num_encoder_layer=4, model_dim=512, nhead=8,
                 pe_type="sin_cos", num_dims=3,
                 *args, **kwargs):
        super(FeatureGrouping, self).__init__()
        self.num_encoder_layer, self.model_dim, self.nhead = (
            num_encoder_layer, model_dim, nhead
        )
        self.C = c
        self.args, self.kwargs = args, kwargs
        
        self.input_proj = nn.Linear(in_features=self.C, out_features=self.model_dim, bias=True)
        
        self.pe_type, self.num_dims = pe_type, num_dims
        self.pe_table = PositionalEmbedding(
            pos_dim=self.model_dim, num_dims=self.num_dims, embed_type=self.pe_type)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.nhead)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=self.num_encoder_layer,
            norm=nn.LayerNorm(self.model_dim)
        )
        
    def forward(self, descriptors, coordinates, *args, **kwargs):
        """ Perform the Feature Grouping.
        
        Args:
             descriptors (torch.Tensor): [B, \\sum_i {Nk_i}, C]
             coordinates (torch.Tensor): [B, \\sum_i {Nk_i}, 3]
             
        Returns:
            descriptors:
        """
        # generate positional embedding
        coords_pe = self.pe_table(coordinates.permute(2, 0, 1).contiguous())  # [B, \sum_i nk, model_dim]
        descriptors = self.input_proj(descriptors)  # [B, \sum_i nk, model_dim]
        descriptors = (descriptors + coords_pe).permute(1, 0, 2)  # [Nk, B, model_dim]
        
        # pass descriptors through transformer encoder
        descriptors = self.encoder(descriptors)  # [Nk, B, model_dim], Nk = \sum_i {nk_i}
        return descriptors.permute(1, 0, 2).contiguous()  # [B, Nk, model_dim]


class FeatureSampling(nn.Module):
    """
    1. coord_conv,
    2. (constrained) deformable conv, pooling,
    3. topk
    ->
    List[N_k_i, 256],
    smooth l1 loss. need sigmoid in forward.
    
    整体的网络，就是
    一个 coord conv + 一个 contrained deformable pooling + 1 mlp.
    
    当前，暂时没有实现： coord conv, constrained deformable pooling.
    分别使用 conv2d 和 maxpool2d 来代替。
    """
    
    def __init__(self,
                 c=256,  # channel number for the output of FPN
                 coord_conv_module=None, nk=(256, 128, 64),
                 constrained_deform_pool_module=None,
                 *args, **kwargs):
        super(FeatureSampling, self).__init__()
        
        self.C = c
        self.coord_conv_module = (
            coord_conv_module if coord_conv_module else
            nn.Sequential(
                nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1)
            )
        )  # self.C to self.C
        if constrained_deform_pool_module:
            self.constrained_deform_pool_module = constrained_deform_pool_module
        else:
            self.constrained_deform_pool_module = nn.MaxPool2d(
                kernel_size=2, stride=2
            )  # (H, W) --> (H/2, W/2)
        
        self.mlp_module = nn.Conv2d(
            in_channels=self.C, out_channels=1, kernel_size=1, stride=1)
        self.Nk = nk
        self.args, self.kwargs = args, kwargs
        
    def forward(self, features, *args, **kwargs):
        r""" Generate Significance map for the input features.
        
        Args:
            features (List[torch.Tensor]): a list with each element corresponding to the feature map
               of a specified level. Each feature map's shape should be [batch_size, C, Hi, Wi].
               
        Returns:
            score_maps (List[torch.Tensor]): A list with each element corresponding to the feature map
              of a specified level. Each feature map's shape should be [batch_size, 1, Hi/2, Wi/2]
            descriptors (torch.Tensor): a tensor with shape of [B, \sum_i Nk_i, C], which will be the input
              of `Feature Grouping Network` after permutation.
            coordinates (torch.Tensor): a tensor with shape of [B, \sum_i Nk_i, 3] representing the coordinates
              for the corresponding feature vector. Form of coordinate should be (feature_level, r, c).
        """
        outputs = []
        descriptors, coordinates = None, None
        for i, feature in enumerate(features):
            B, C, Hi, Wi = feature.shape
            if self.coord_conv_module:
                feature = self.coord_conv_module(feature)  # [B, C, Hi, Wi]
            if self.constrained_deform_pool_module:
                feature = self.constrained_deform_pool_module(feature)  # (B, C, Hi/2, Wi/2)
                
            # generate significance map
            significance_map = self.mlp_module(feature).sigmoid().squeeze(dim=1)  # (B, Hi/2, Wi/2)
            outputs.append(significance_map)
            
            # feature sampling
            nk = self.Nk[i]
            _, topk_indices = torch.topk(significance_map.flatten(-2, -1), nk, dim=-1)  # [B, nk]
            # topk_feats = feature.flatten(-2, -1).permute(0, 2, 1)[topk_indices]  # [B, nk, C]
            topk_feats = torch.gather(
                input=feature.flatten(-2, -1).permute(0, 2, 1),
                dim=1, index=torch.tile(topk_indices.unsqueeze(dim=-1), (C, ))
            )
            topk_indices_r, topk_indices_c = (
                torch.div(topk_indices, significance_map.shape[-1], rounding_mode="floor"),
                torch.remainder(topk_indices, significance_map.shape[-1])
            )
            topk_coords = torch.cat([
                torch.full([B, nk, 1], i, device=topk_indices_r.device),  # self.device
                topk_indices_r.unsqueeze(dim=-1), topk_indices_c.unsqueeze(dim=-1)],
                dim=-1
            )  # [B, nk, 3], (feature_level, r, c)
            if descriptors is None:
                descriptors = topk_feats
                coordinates = topk_coords
            else:
                descriptors = torch.cat(  # [B, \sum_i{nk_i}, C]
                    [descriptors, topk_feats], dim=1)
                coordinates = torch.cat(  # [B, \sum_i{nk_i} 3]
                    [coordinates, topk_coords], dim=1)
        return outputs, descriptors, coordinates


@MODELS.register_module()
class FewNetHead(BaseTextDetHead):
    def __init__(
        self, 
        feature_sampling: Optional[Dict] = None, 
        feature_grouping: Optional[Dict] = None, 
        target_mode="rbox", is_coord_norm=True, inner_channels=256, model_dim=256,

        module_loss: Dict = dict(
            type='FewNetModuleLoss'), 
        postprocessor: Dict = dict(
            type="FewNetpostprocessor"), 
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Kaiming', layer=['Conv', 'Linear']),
            dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4),
            dict(type="TruncNormal", layer='TransformerEncoderLayer')  # for Transformer
        ]
    ) -> None:
        super().__init__(module_loss, postprocessor, init_cfg)

        assert (isinstance(feature_sampling, (nn.Module, Dict)) and 
                isinstance(feature_grouping, (nn.Module, Dict))), (
            "Please check the parameter for feature_sampling and feature_grouping"
            "Due to your feature_sampling and feature_grouping  are: {}, {}".format(
                feature_sampling, feature_grouping))
        self.feature_sampling = ( 
            FeatureSampling(**feature_sampling) if isinstance(feature_sampling, Dict) else
            feature_sampling 
        )
        self.feature_grouping = (
            FeatureGrouping(**feature_grouping) if isinstance(feature_grouping, Dict) else
            feature_grouping
        )
        self.target_mode = target_mode
        self.is_coord_norm = is_coord_norm  # 对于 detection 来说，坐标是否是 normalized ?
        self.C = inner_channels  # out_channels for feature pyramid network
        self.model_dim = model_dim  # number of channels for output of feature grouping module
        
        if self.target_mode.lower() == "rbox":  # head should be combined
            if not self.is_coord_norm:
                warnings.warn("the xywh will be normalized all the time,"
                              " Though your self.is_coord_norm is: {}".format(self.is_coord_norm))
                
            self.cls_head = nn.Sequential(
                nn.Linear(self.model_dim, 1),  # cls_logits
                nn.Sigmoid()
            )
            self.xywh_head = nn.Sequential(
                nn.Linear(self.model_dim, 4),
                nn.Sigmoid()
            )
            self.angle_head = nn.Sequential(
                nn.Linear(self.model_dim, 1), nn.Sigmoid()
            )
        else:  # currently no other representation
            pass
        
    def forward(self, features : Tuple[torch.Tensor], *args, **kwargs):
        """NO Loss calcualtion in forward
        """
        out = OrderedDict()
        p2, p3, p4, _ = features 
        features = (p2, p3, p4)
        # pas features through feature sampling network
        score_maps, descriptors, coordinates = self.feature_sampling(features)  # [B, C, H, W], [B, Nk, C]
        # pass features through feature grouping network
        descriptors = self.feature_grouping(descriptors, coordinates)  # [B, Nk, model_dim]
        # pass descriptors through head
        logits = self.cls_head(descriptors)
        boxes = self.xywh_head(descriptors)
        angle = self.angle_head(descriptors)
        out.update(  # update the output here 
            score_map=score_maps,  # score_maps --> score_map
            logits=logits, boxes=boxes, angle=angle
        )
        return out 

    def loss(self,
             features: Tuple[torch.Tensor], batch_data_samples: DetSampleList
    ) -> Dict:
        outs = self(features, batch_data_samples, mode="loss")
        loss_dict = self.module_loss(outs, batch_data_samples)
        return loss_dict 

    def loss_and_predict(
        self, 
        features: Tuple[torch.Tensor], 
        batch_data_samples: DetSampleList
    ) -> Tuple[Dict, DetSampleList]:
        outs = self(features, batch_data_samples, mode="both")
        loss_dict = self.module_loss(outs, batch_data_samples)
        preds = self.post_processor(outs, batch_data_samples)
        return loss_dict, preds

    def predict(
        self, 
        features: Tuple[torch.Tensor], batch_data_samples: DetSampleList
    ) -> DetSampleList:
        outs = self(features, batch_data_samples, mode='predict')
        preds = self.postprocessor(outs, batch_data_samples)
        return preds
