# Copyright (c) OpenMMLab.
#
# GLH-Occ / FlashOcc M3 variant:
#   Swin-B + BEV Stereo + 512x1408 + MS-BEV + BEV auxiliary supervision.
#
# IMPORTANT:
# This configuration requires a registered detector class named
# `BEVStereo4DOCCMSBEVBEVAux`. The class should inherit from
# `BEVStereo4DOCC` and port the MS-BEV / BEV-Aux logic from
# `BEVDetOCCMSBEVBEVAux`.
#
# The P8/P32 auxiliary branches are applied to the KEY FRAME only.
# The original P16 main branch keeps the M3 BEV-Stereo transformation.

_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (512, 1408),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 80
multi_adj_frame_id_cfg = (1, 2, 1)

model = dict(
    # This class must be implemented and registered in the detector registry.
    type='BEVStereo4DOCCMSBEVBEVAux',

    # Original M3 temporal / stereo settings.
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),

    # BEV auxiliary supervision.
    bev_aux_loss_weight=0.05,
    bev_aux_in_channels=256,
    bev_aux_hidden_channels=256,
    bev_aux_num_classes=17,
    bev_aux_ignore_idx=255,
    bev_aux_pos_weight_max=3.0,

    # Multi-scale BEV branches.
    # Swin-B raw stages: P8=256, P16=512, P32=1024 channels.
    # P16 remains the original BEV-Stereo main branch.
    # P8/P32 use lightweight key-frame LSS branches.
    ms_view_transformer_p8=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=8,
    ),
    ms_view_transformer_p32=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=32,
    ),
    ms_img_channels=(256, 512, 1024),
    ms_proj_channels=256,
    ms_bev_channels=numC_Trans,
    ms_fusion_channels=256,
    ms_use_p8=True,
    ms_use_p16=True,
    ms_use_p32=True,

    # M3 image encoder. out_indices is extended from (2, 3) to
    # (1, 2, 3) so the MS-BEV module can access P8/P16/P32.
    img_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        return_stereo_feat=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False,
        with_cp=True,
    ),

    # After removing the P4 stereo feature, the feature list is
    # [P8, P16, P32]. The original M3 neck uses P16 and P32.
    img_neck=dict(
        type='FPN_LSS',
        in_channels=512 + 1024,
        out_channels=512,
        extra_upsample=None,
        input_feature_index=(1, 2),
        scale_factor=2,
    ),

    # Original M3 P16 BEV-Stereo view transformer.
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(
            use_dcn=False,
            aspp_mid_channels=96,
            stereo=True,
            bias=5.0,
        ),
        downsample=16,
    ),

    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        with_cp=True,
        numC_input=numC_Trans *
        (len(range(*multi_adj_frame_id_cfg)) + 1),
        num_channels=[
            numC_Trans * 2,
            numC_Trans * 4,
            numC_Trans * 8,
        ],
    ),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256,
    ),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[1],
        num_channels=[numC_Trans],
        stride=[1],
        backbone_output_ids=[0],
    ),

    occ_head=dict(
        type='BEVOCCHead2D',
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0,
        ),
    ),
)

# Data.
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True,
    ),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True,
    ),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=grid_config,
    ),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs',
            'gt_depth',
            'voxel_semantics',
            'mask_lidar',
            'mask_camera',
        ],
    ),
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=data_config,
        sequential=True,
    ),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False,
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False,
            ),
            dict(type='Collect3D', keys=['points', 'img_inputs']),
        ],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl',
)

# Practical setting for 2 x RTX 3090.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
    ),
    val=test_data_config,
    test=test_data_config,
)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimization.
# 1e-4 is retained for stability with the small 2-GPU batch.
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[20],
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]

# Evaluate saved checkpoints separately.
evaluation = dict(interval=999, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

# Initialize from the official trained FlashOcc M3 checkpoint, matching the validated M1 plug-in workflow.
load_from = 'ckpts/flashocc-stbase-4d-stereo-512x1408.pth'
resume_from = None

# Strongly recommended for 512x1408 on 24-GB GPUs.
fp16 = dict(loss_scale='dynamic')
work_dir = './work_dirs/flashocc-stbase-4d-stereo-512x1408-msbev-bevaux-m3-24e'
