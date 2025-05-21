############数据集配置############
dataset_type = 'DIORRDataset'
dataset_root = './datasets/ObjectDetection/DIORR/'
num_classes = 20
train_batch_size = 2
val_batch_size = 8
image_size = (784,784)
train_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True,ratio_range=(0.5,2.0), resize_type='mmdet.Resize',scale=image_size,type='mmdet.RandomResize'),
    dict(crop_size=image_size, type='mmdet.RandomCrop'),
    dict(direction=['horizontal','vertical','diagonal'], prob=0.75, type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=image_size, type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'), type='mmdet.PackDetInputs'),
]
train_dataloader = dict(
    batch_sampler=None,
    batch_size=train_batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'),
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['DATASET_TYPE',],
        datasets=[
            dict(
                type='DIORDataset',
                ann_file='ImageSets/Main/train.txt',
                data_prefix=dict(img_path='images'),
                data_root=dataset_root,
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline),
            dict(
                type='DIORDataset',
                ann_file='ImageSets/Main/val.txt',
                data_prefix=dict(img_path='images'),
                data_root=dataset_root,
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline),
        ],
    )
)
val_dataloader = dict(
    batch_size=4,
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        type='DIORDataset',
        ann_file='ImageSets/Main/test.txt',
        data_prefix=dict(img_path='images'),
        data_root=dataset_root,
        pipeline=test_pipeline,
        test_mode=True,
    )
)
val_evaluator = dict(metric='mAP', type='DOTAMetric')
test_dataloader = val_dataloader
test_evaluator = val_evaluator
############模型配置############
angle_cfg = dict(start_angle=0, width_longer=True)
checkpoint = './model_zoo/CGEarthEye-Giant-518' #预训练权重
frozen_stages = 40 # 冻结骨干
img_size = 518 # 模型预训练所用图像大小
costs = [
    dict(type='mmdet.FocalLossCost', weight=2.0),
    dict(box_format='xywha', type='HausdorffCost', weight=5.0),
    dict(fun='log1p',loss_type='kld',sqrt=False,tau=1,type='GDCost',weight=5.0),
]
model = dict(
    type='RHINO',
    version='v2',
    with_box_refine=True,
    as_two_stage=True,
    num_queries=900,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675,116.28,103.53],
        std=[58.395,57.12,57.375],
        boxtype2tensor=False,
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        arch='dinov2-giant',
        drop_path_rate=0.3,
        frozen_stages=frozen_stages,
        img_size=img_size,
        init_cfg=dict(checkpoint=checkpoint,prefix='backbone',type='Pretrained'),
        layer_cfgs=dict(ffn_type='swiglu_fused'),
        layer_scale_init_value=1e-05,
        out_indices=(13,26,39,),
        out_type='featmap',
        patch_size=14,
        type='mmpretrain.VisionTransformer',
        with_cls_token=False),
    neck=dict(
        act_cfg=None,
        in_channels=[1536,1536,1536],
        kernel_size=1,
        ml_out_channels=1536,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        scales=[2,1,0.5],
        type='MLChannelMapper'),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    bbox_head=dict(
        angle_cfg=dict(start_angle=0, width_longer=True),
        loss_bbox=dict(loss_weight=5.0, type='mmdet.L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(
            fun='log1p',
            loss_type='kld',
            loss_weight=5.0,
            sqrt=False,
            tau=1,
            type='GDLoss'),
        num_classes=20,
        sync_cls_avg_factor=True,
        type='RHINOPositiveHungarianClassificationHead'),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True,max_num_groups=30,num_dn_queries=100,num_groups=None),
        label_noise_scale=0.5),
    train_cfg=dict(
        assigner=dict(
            match_costs=costs,
            type='mmdet.HungarianAssigner'),
        dn_assigner=dict(
            match_costs=costs,
            type='DNGroupHungarianAssigner')),

    test_cfg=dict(max_per_img=500),
)
############优化器配置############
max_epochs = 60
warmup_LinearLR = 2
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0))),
    )
param_scheduler = [
    dict(begin=0,by_epoch=True,convert_to_iter_based=True,end=warmup_LinearLR,start_factor=0.001,type='LinearLR'),
    dict(T_max=max_epochs-warmup_LinearLR, begin=warmup_LinearLR, by_epoch=True, end=max_epochs, type='CosineAnnealingLR'),
]
############训练配置############
work_dir = './work_dirs/ObjectDetection/CGEarthEye-Giant-784-DIORR'
val_interval = 3
auto_scale_lr = dict(enable=False)
backend_args = None
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=6)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_scope = 'mmrotate'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=1, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
vis_backends = [dict(type='LocalVisBackend'),]
visualizer = dict(name='visualizer',type='RotLocalVisualizer',vis_backends=vis_backends)
launcher = 'pytorch'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
load_from = None
resume = False

