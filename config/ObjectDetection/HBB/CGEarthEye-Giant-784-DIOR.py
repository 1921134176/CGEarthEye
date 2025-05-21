############数据集配置############
dataset_type = 'CocoDataset'
dataset_root = './datasets/ObjectDetection/DIOR/'
image_size = (784,784)
backend_args = None
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='RandomResize', scale=image_size, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_type='absolute_range', crop_size=image_size, recompute_bbox=True, allow_negative_crop=True),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=image_size, type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor',
        ),
        type='PackDetInputs'),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        backend_args=None,
        ann_file='Annotations/trainval.json',
        data_prefix=dict(img='images/trainval'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
        ),
)
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=dataset_root,
        ann_file='Annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images/test'),
        pipeline=test_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file= dataset_root + 'Annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_evaluator = val_dataloader
############模型配置############
checkpoint = './model_zoo/CGEarthEye-Giant-518' #预训练权重
img_size = 518 # 模型预训练所用图像大小
frozen_stages = 40 # 冻结骨干
num_classes = 20
model = dict(
    type='DINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        arch='dinov2-giant',
        frozen_stages=frozen_stages,
        img_size=img_size,
        init_cfg=dict(
            checkpoint=checkpoint,
            type='Pretrained',
            prefix='backbone'
        ),
        layer_cfgs=dict(ffn_type='swiglu_fused'),
        layer_scale_init_value=1e-05,
        out_indices=(
            13,
            26,
            39,
        ),
        out_type='featmap',
        drop_path_rate=0.3,
        patch_size=14,
        type='mmpretrain.VisionTransformer',
        with_cls_token=False),
    neck=dict(
        type='MLChannelMapper',
        in_channels=[1536, 1536, 1536],
        kernel_size=1,
        out_channels=256,
        ml_out_channels=1536,
        scales=[2,1,0.5],
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR
############优化器配置############
max_epochs = 60
warmup_LinearLR = 2
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001,
        betas=(0.9, 0.999)
        ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1),
                     'pos_embed': dict(decay_mult=0.),
                     'cls_token': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.),
                    })
)
param_scheduler = [
    dict(begin=0, by_epoch=True, convert_to_iter_based=True, end=warmup_LinearLR, start_factor=0.001, type='LinearLR'),
    dict(T_max=max_epochs-warmup_LinearLR, begin=warmup_LinearLR, by_epoch=True, end=max_epochs, type='CosineAnnealingLR'),
]
############训练配置############
work_dir = './work_dirs/ObjectDetection/CGEarthEye-Giant-784-DIOR'
val_interval = 3
auto_scale_lr = dict(enable=False)
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# defaults to use registries in mmdet
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='coco/bbox_mAP_50', type='CheckpointHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
# configure environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# set log
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
# load from which checkpoint
load_from = None
# whether to resume training from the loaded checkpoint
resume = False

