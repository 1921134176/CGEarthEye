dataset_root = r'./datasets/ChangeDetection/SYSU-CD'
dataset_type = 'LEVIR_CD_Dataset'
backbone_checkpoint = r"./model_zoo/CGEarthEye-Giant-518.pth"
batch_crop_size = (224, 224)
train_batch_size = 8
val_batch_size = 8
work_dir = r'./res/SYSU-CD'

default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='CDLocalVisBackend')]
visualizer = dict(
    type='CDLocalVisualizer', 
    vis_backends=vis_backends, name='visualizer', alpha=1.0)
log_processor = dict(by_epoch=False)
log_level = 'INFO'


train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=batch_crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=batch_crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]
img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='MultiImgLoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='MultiImgResize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
                dict(type='MultiImgRandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='MultiImgLoadAnnotations')],
            [dict(type='MultiImgPackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=batch_crop_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True, 
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='train/Label',
            img_path_from='train/Image1',
            img_path_to='train/Image2'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=batch_crop_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='val/Label',
            img_path_from='val/Image1',
            img_path_to='val/Image2'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=batch_crop_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='test/Label',
            img_path_from='test/Image1',
            img_path_to='test/Image2'),
        pipeline=test_pipeline))

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(
    type='mmseg.IoUMetric',
    iou_metrics=['mFscore', 'mIoU'])

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=0,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(
        size_divisor=0
    ))
model = dict(
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
            type='mmpretrain.VisionTransformer',
            arch='dinov2-giant',
            img_size=batch_crop_size,
            patch_size=14,
            out_type='featmap',
            out_indices=(9, 19, 29, 39),
            layer_scale_init_value=1e-5,
            layer_cfgs=dict(ffn_type='swiglu_fused'),
            frozen_stages=frozen_stages,
            init_cfg=dict(type='Pretrained',
                          checkpoint=backbone_checkpoint,
                          prefix='backbone'
                          ),
    ),
    neck=dict(type='MultiLevelFeatureFusionNeck',
              multilevel_in_channels=[1536, 1536, 1536, 1536],
              multilevel_out_channels=1536,
              multilevel_scales=[4, 2, 1, 0.5],
              featurefusion_policy='concat'),
    decode_head=dict(
        type='mmseg.UPerHead',
        in_channels=[v * 2 for v in [1536, 1536, 1536, 1536]],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256, 
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='mmseg.FCNHead',
        in_channels=1536 * 2,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))   

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=warmup_LinearLR,
        convert_to_iter_based=True
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

# training schedule for 100 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, save_best='mIoU', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1,
                       img_shape=(256, 256, 3)))

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
