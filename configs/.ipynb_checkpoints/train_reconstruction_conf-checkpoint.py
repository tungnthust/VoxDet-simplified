# optimizer
optimizer = dict(type='Adam', lr=5e-5, betas=(0,0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[8, 12])
total_epochs = 16
# model settings
model = dict(
    type='ZidRCNN',
    pretrained='torchvision://resnet50',
    neg_rpn=False,
    mode='recon',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OlnRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            # Use a single anchor per location.
            scales=[8],
            ratios=[1.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='TBLRBBoxCoder',
            normalizer=1.0,),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='IoULoss', linear=True, loss_weight=10.0),
        objectness_type='Centerness',
        loss_objectness=dict(type='L1Loss', loss_weight=1.0),
        ),
    roi_head=dict(
        type='ZidRoIHead3DGConvMix',
        mode='recon',
        support_guidance=dict(
            in_channel3d=32,
            hidden_3d=32,
            in_channel2d=256,
            hidden_2d=512,
            learn_rot=True,
            num_r=1),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxSuperHead',
            neg_head=True,
            contrastive=dict(
                test_topk=10,
                avg_conrta=True,
                bg_conrta=False,
                num_conrta=10),
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=1.0,
                ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            bbox_score_type='BoxIoU',  # 'BoxIoU' or 'Centerness'
            loss_bbox_score=dict(type='L1Loss', loss_weight=1.0),
            loss_contra=dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=0.0,
                ),
            )),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            # Objectness assigner and sampler 
            objectness_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.1,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            objectness_sampler=dict(
                type='RandomSampler',
                num=256,
                # Ratio 0 for negative samples.
                pos_fraction=1.,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='InstanceSampler', # make sure gt is always input for ROI head
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.0,  # No nms
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.7),
            # max_per_img should be greater enough than k of AR@k evaluation
            # because the cross-dataset AR evaluation does not count those
            # proposals on the 'seen' classes into the budget (k), to avoid
            # evaluating recall on seen-class objects. It's recommended to use
            # max_per_img=1500 or 2000 when evaluating upto AR@1000.
            max_per_img=1500,
            )
    ))

# Dataset
dataset_type = 'ReconZidDataset'
data_root = '/home/minhnh/project_drive/CV/FewshotObjectDetection/data/OWID/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        p1_path=data_root + 'P1/',
        ins_scale=(0, 9600),
        video_num=1000,
        input_size=256),
    val=dict(
        type=dataset_type,
        ins_scale=(9600, -1),
        p1_path=data_root + 'P1/',
        input_size=256,
        video_num=1,
        test_mode=True))

checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_layers = None
resume_from = None
resume_checkpoint = '/home/minhnh/project_drive/CV/FewshotObjectDetection/VoxDet-simplified/results/ckpt/model_ep14_004200.pt'
workflow = [('train', 1)]

work_dir='outputs/VoxDet_p1'
