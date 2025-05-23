############数据集配置############
dataset_type = 'iSAIDDataset'
dataset_root = './datasets/SemanticSegmentation/ISAID/'
image_size = (518,518)
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(0.5,2.0,),
        scale=(1600,400,),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=image_size, type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackSegInputs'),
]
val_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root=dataset_root,
        pipeline=train_pipeline,
        type=dataset_type),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root=dataset_root,
        pipeline=val_pipeline,
        type=dataset_type),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        data_root=dataset_root,
        pipeline=test_pipeline,
        type=dataset_type),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
test_evaluator = val_evaluator
############模型配置############
checkpoint = './model_zoo/CGEarthEye-Giant-518' #预训练权重
img_size = 518 # 模型预训练所用图像大小
frozen_stages = 40 # 冻结骨干
num_classes = 16
data_preprocessor = dict(
        bgr_to_rgb=True,
        mean=[123.675,116.28,103.53,],
        pad_val=0,
        seg_pad_val=255,
        size=img_size,
        std=[58.395,57.12,57.375,],
        type='SegDataPreProcessor')
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1536,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        arch='vitadapter-g',
        frozen_stages=40,
        img_size=518,
        init_cfg=dict(
            checkpoint=checkpoint,
            prefix='backbone.',
            type='Pretrained'),
        layer_cfgs=dict(ffn_type='swiglu_fused'),
        layer_scale_init_value=1e-05,
        out_indices=[9,19,29,39],
        out_type='featmap',
        patch_size=14,
        type='mmpretrain.VisionTransformer'),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[1536,1536,1536,1536,],
        in_index=[0,1,2,3,],
        loss_decode=[
            dict(loss_weight=1.0, type='CrossEntropyLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        pool_scales=(1,2,3,6,),
        type='UPerHead'),
    neck=dict(
        in_channels=[1536,1536,1536,1536,],
        out_channels=1536,
        scales=[4,2,1,0.5,],
        type='MultiLevelNeck'),
    test_cfg=dict(crop_size=img_size, mode='slide', stride=(10,10,)),
    train_cfg=dict(),
    type='EncoderDecoder')
############优化器配置############
max_iters = 160000
warmup_LinearLR = 3000
start_lr = 6e-05
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=start_lr, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=warmup_LinearLR, start_factor=1e-06,type='LinearLR'),
    dict(begin=warmup_LinearLR,by_epoch=False,end=max_iters,eta_min=0.0,power=1.0,type='PolyLR'),]
############训练配置############
work_dir = './work_dirs/SemanticSegmentation/CGEarthEye-Giant-518-ISAID'
val_interval = 1600
train_cfg = dict(max_iters=max_iters, type='IterBasedTrainLoop', val_interval=1600)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# defaults to use registries in mmdet
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=5000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
# configure environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(name='visualizer',type='SegLocalVisualizer',vis_backends=[dict(type='LocalVisBackend'),])
# set log
log_processor = dict(by_epoch=False)
log_level = 'INFO'
# load from which checkpoint
load_from = None
# whether to resume training from the loaded checkpoint
resume = False
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmpretrain.models',
    ])
launcher = 'none'
norm_cfg = dict(requires_grad=True, type='SyncBN')





