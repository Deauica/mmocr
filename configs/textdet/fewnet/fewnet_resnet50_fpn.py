_base_ = [
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
    '_base_fewnet_resnet50-fpn.py',
]

# seed
randomness = dict(seed=20)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-4),
    custom_keys={
        "feature_grouping": dict(lr=5e-4)
    }
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400, val_interval=20)
param_scheduler = [
    dict(type='PolyLR', power=0.9, end=400),
]

# model -- simplified version
model = dict(
    det_head=dict(
        module_loss=dict(
            weight_cost_logits=0, weight_cost_boxes=1,
            weight_loss_logits=0, weight_loss_rbox=1,
            need_norm_boxes=False,  # no need to normalize boxes currently
            need_scaled_gwd=False   # no need to utilize scaled gwd loss
        )
    )
)

file_client_args = dict(backend='disk')
# for train
ic15_det_train = _base_.ic15_det_train
ic15_det_train.pipeline = _base_.train_pipeline  # pipeline update

train_dataloader = dict(
    batch_size=32,
    num_workers=32,
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
