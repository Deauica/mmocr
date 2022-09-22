# step 0: Modified from _base_
data_type = "toy"

# petrel_url = "s3://openmmlab/datasets/ocr/"  # useless
data_root = ("s3://openmmlab/datasets/ocr/det/icdar2015/"
             if data_type == "petrel" else "tests/data/det_toy_dataset")

file_client_args = dict(backend="petrel") if data_type == "petrel" else dict(
    backend="disk")

train_anno_path = ('instances_training.json'
                   if data_type == "petrel" else "instances_test.json")

train_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file=train_anno_path,
    data_prefix=dict(img_path='imgs/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

train_list = [train_dataset]

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
        scale=[640, 640], keep_ratio=True  # keep_ratio is True
    ),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset', datasets=train_list, pipeline=train_pipeline))

# Visualizer
visualizer = dict(
    type='TextDetLocalVisualizer', name='visualizer', save_dir='imgs')
