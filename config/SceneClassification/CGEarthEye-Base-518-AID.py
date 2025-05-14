############数据集配置############
dataset_type = 'AID'
num_classes = 30
data_root='./datasets/SceneClassification'
data_prefix='AID'
train_ann_file='AID/val_20per.txt'
val_ann_file='AID/train_80per.txt'
train_batch_size = 32
val_batch_size = 64
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=518),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=532, edge='short'),
    dict(type='CenterCrop', crop_size=518),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=data_prefix,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=data_prefix,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = val_dataloader
test_evaluator = val_evaluator
############模型配置############
img_size = 518 # 模型预训练所用图像大小
frozen_stages = 12 # 冻结骨干
# frozen_stages = -1 # 全量微调
checkpoint = './model_zoo/CGEarthEye-Base-518' #预训练权重
data_preprocessor = dict(
    num_classes=num_classes ,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=img_size,
        patch_size=14,
        layer_scale_init_value=1e-5,
        frozen_stages=frozen_stages,
        init_cfg=dict(type='Pretrained',
                      checkpoint=checkpoint, 
                      prefix='backbone')
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=num_classes,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
    ))
############优化器配置############
learn_rate = 1e-4
epoch = 100
warmup_LinearLR = 5
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=learn_rate,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=True,
        begin=0,
        end=warmup_LinearLR,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=epoch-warmup_LinearLR,
        by_epoch=True,
        begin=warmup_LinearLR,
        end=epoch,
    )
]
############训练配置############
work_dir = './work_dirs/SceneClassification/CGEarthEye-Base-518-AID'
val_interval = 1
auto_scale_lr = dict(base_batch_size=128)
train_cfg = dict(by_epoch=True, max_epochs=epoch, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()
# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=50),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best="auto"),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)
# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)
# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)
# set log level
log_level = 'INFO'
# load from which checkpoint
load_from = None
# whether to resume training from the loaded checkpoint
resume = False
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

