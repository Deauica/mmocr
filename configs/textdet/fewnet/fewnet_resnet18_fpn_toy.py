_base_ = [
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
    '_base_fewnet_resnet18-fpn.py',
]

# basic runtime setting
randomness = dict(seed=20)

# model -- simplified version
model = dict(
    neck=dict(out_channels=32),
    det_head=dict(
        inner_channels=32, model_dim=16,
        
        feature_sampling=dict(
            c=32, nk=(32, 16, 8)
            # coord_conv, constrained_deformable_module is None currently
        ),  # c == inner_channels
        feature_grouping=dict(
            c=32, num_encoder_layer=1, model_dim=16, nhead=2,
            pe_type="sin_cos", num_dims=3
        ),
    )
)

file_client_args = dict(backend='disk')
# for train
ic15_det_train = _base_.ic15_det_train
ic15_det_train.pipeline = _base_.train_pipeline  # pipeline update

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_det_train)

# for evaluation and test
val_dataloader = None
val_cfg = None
val_evaluator = None

test_dataloader = None
test_cfg = None
test_evaluator = None
