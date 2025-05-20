_base_ = './default_runtime.py'
dataset_root = r'E:\cgeartheye-main\cgeartheye\datasets\ChangeDetection\CDD'
dataset_type = 'SVCD_Dataset'
model_path = r"E:\cgeartheye-main\cgeartheye\model_zoo\20240724_dinov2_vitg-p14-droppathrate0.4_20240711_205W_8xb56_449999i_224x224_mmseg.pth"
batch_crop_size = (224, 224)
train_batch_size = 16
val_batch_size = 16
work_dir = r'E:\CDD'

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
    dict(
        cat_max_ratio=0.75, crop_size=batch_crop_size, type='MultiImgRandomCrop'),
    dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
    dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        brightness_delta=10,
        contrast_range=(
            0.8,
            1.2,
        ),
        hue_delta=10,
        saturation_range=(
            0.8,
            1.2,
        ),
        type='MultiImgPhotoMetricDistortion'),
    dict(type='MultiImgPackSegInputs'),
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='MultiImgResize'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
img_ratios = [
    0.75,
    1.0,
    1.25,
]

tta_pipeline = [
    dict(backend_args=None, type='MultiImgLoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True, scale_factor=0.75, type='MultiImgResize'),
                dict(keep_ratio=True, scale_factor=1.0, type='MultiImgResize'),
                dict(
                    keep_ratio=True, scale_factor=1.25, type='MultiImgResize'),
            ],
            [
                dict(
                    direction='horizontal',
                    prob=0.0,
                    type='MultiImgRandomFlip'),
                dict(
                    direction='horizontal',
                    prob=1.0,
                    type='MultiImgRandomFlip'),
            ],
            [
                dict(type='MultiImgLoadAnnotations'),
            ],
            [
                dict(type='MultiImgPackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(
        data_prefix=dict(
            img_path_from='train/Image1',
            img_path_to='train/Image2',
            seg_map_path='train/Label'),
        data_root=dataset_root,
        pipeline=train_pipeline,
        type='SVCD_Dataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))

val_dataloader = dict(
    batch_size=val_batch_size,
    dataset=dict(
        data_prefix=dict(
            img_path_from='val/Image1',
            img_path_to='val/Image2',
            seg_map_path='val/Label'),
        data_root=dataset_root,
        pipeline=test_pipeline,
        type='SVCD_Dataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_dataloader = dict(
    batch_size=val_batch_size,
    dataset=dict(
        data_prefix=dict(
            img_path_from='test/Image1',
            img_path_to='test/Image2',
            seg_map_path='test/Label'),
        data_root=dataset_root,
        pipeline=test_pipeline,
        type='SVCD_Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')

test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')

# model settings
norm_cfg = dict(requires_grad=True, type='SyncBN')
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53] * 2,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=0,  
    std=[58.395, 57.12, 57.375] * 2,
    test_cfg=dict(size_divisor=0), 
    type='DualInputSegDataPreProcessor')

model = dict(
    asymetric_input=True,
    data_preprocessor=data_preprocessor,
    image_encoder=dict(
        type='mmpretrain.VisionTransformer',
        arch='dinov2-giant',
        img_size=(224, 224),
        patch_size=14,
        out_type='featmap',
        out_indices=(9, 19, 29, 39),
        layer_scale_init_value=1e-5,
        layer_cfgs=dict(ffn_type='swiglu_fused'),
        frozen_stages=40,
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth',
                      prefix='backbone'
                      ),
    ),
    decode_head=dict(
        ban_cfg=dict(
            clip_channels=1536,
            fusion_index=[
                0,   # new add
                1,
                2,
                3,  
            ],
            side_enc_cfg=dict(
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                drop_rate=0.0,
                embed_dims=64,
                in_channels=3,
                init_cfg=dict(
                    checkpoint=head_path,
                    type='Pretrained'),
                mlp_ratio=4,
                num_heads=[
                    1,
                    2,
                    5,
                    8,
                ],
                num_layers=[
                    3,
                    4,
                    6,
                    3,
                ],
                num_stages=4,
                out_indices=(
                    0,
                    1,
                    2,
                    3,
                ),
                patch_sizes=[
                    7,
                    3,
                    3,
                    3,
                ],
                qkv_bias=True,
                sr_ratios=[
                    8,
                    4,
                    2,
                    1,
                ],
                type='mmseg.MixVisionTransformer')),
        ban_dec_cfg=dict(
            type='BAN_MLPDecoder',
            in_channels=[64, 128, 320, 512],
            channels=128,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False),
        loss_decode=dict(
            loss_weight=1.0, type='mmseg.CrossEntropyLoss', use_sigmoid=False),
        type='BitemporalAdapterHead'),
    encoder_resolution=dict(mode='bilinear', size=(
        224,
        224,
    )),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='BAN')

tta_model = dict(type='mmseg.SegTTAModel')

# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            img_encoder=dict(decay_mult=1.0, lr_mult=0.1),
            mask_decoder=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')

param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=80000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]

train_cfg = dict(max_iters=80000, type='IterBasedTrainLoop', val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=2000, save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        img_shape=(
            256,
            256,
            3,
        ), interval=1, type='CDVisualizationHook'))

log_processor = dict(by_epoch=False)
