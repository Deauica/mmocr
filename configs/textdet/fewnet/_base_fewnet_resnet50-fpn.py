# In this file,
# define model, train_pipeline, test_pipeline

inner_channels = 256
model_dim = 512

model = dict(
    type='FewNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    
    neck=dict(
        type='mmdet.FPN', in_channels=[256, 512, 1024, 2048],
        out_channels=inner_channels, num_outs=4
    ),  # Adapted from PANet
    
    det_head=dict(  # Head, ModuleLoss, PostProcessor should be defined here
        type='FewNetHead',  # User-defined
        target_mode="rbox", is_coord_norm=True,
        inner_channels=inner_channels, model_dim=model_dim,

        feature_sampling=dict(
            c=inner_channels, nk=(256, 128, 64)
            # coord_conv, constrained_deformable_module is None currently
        ),  # c == inner_channels
        feature_grouping=dict(
            c=inner_channels, num_encoder_layer=4, model_dim=model_dim, nhead=8,
            pe_type="sin_cos", num_dims=3
        ),

        module_loss=dict(
            type='FewNetModuleLoss',  # other parameter for fewnet_moduleloss 
        ),  # For BaseTextDetHead
        postprocessor=dict(
            type='FewNetPostprocessor',  # other parameter for fewnet_postprocessor
        ),
        init_cfg= [
            dict(type='Kaiming', layer=['Conv', 'Linear']),
            dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4),
            dict(type="TruncNormal", layer='TransformerEncoderLayer')  # for Transformer
        ]
    ),
    data_preprocessor=dict(  # need no change 
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))

file_client_args = dict(backend="disk")

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='color_ignore_orientation',
        file_client_args=file_client_args),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(  # RandomRotate
        type='RandomRotate',  # 更改 10 到15， 但是无所谓
        max_angle=15,
        use_canvas=True),
    dict(
        type='RandomFlip',  # 保持默认就可以
        prob=[0.3, 0.5],
        direction=["horizontal", "vertical"]),
    dict(
        type='ImgAugWrapper',
        args=[
            dict(cls="PerspectiveTransform", scale=(0.01, 0.15))  # default
        ]),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5,
        contrast=0.5),
    dict(
        type="RandomCrop",
        min_side_ratio=0.4  # default value,这里不采用``TextDetRandomCrop``
    ),
    dict(
        type="Resize",
        scale=[1280, 1280], keep_ratio=False  # keep_ratio is True
    ),
    dict(
        type='PackTextDetInputs',  # PackTextDetInputs has no batch op 
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]
test_pipeline = [

]