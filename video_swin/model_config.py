"""Standalone MMAction2 Swin-B model definition for binary scratching detection.

Resolves the source configuration's Swin-Tiny base inheritance and Swin-B
overrides into one model dictionary. No data or training configuration.
"""

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        arch='base',
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_base_patch4_window7_224.pth',
        pretrained2d=True,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'))
