default_scope = "mmpretrain"
custom_imports = dict(imports=["mmdet.models"], allow_failed_imports=False)
num_classes = 2
dataset = "cmmd"
fold = 0
size = (1024, 512)  # h, w
epochs = 24
batch_size = 24
model = dict(
    type="BreastCancerAuxCls",
    backbone=dict(
        type="MobileNetV3",
        arch="small_050",
        with_cp=True,
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="StackedLinearClsHead",
        num_classes=num_classes,
        in_channels=288,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type="HSwish"),
        loss=dict(type="SoftmaxEQLLoss", num_classes=num_classes),
        init_cfg=dict(type="Normal", layer="Linear", mean=0.0, std=0.01, bias=0.0),
        cal_acc=False,
    ),
    model_config=dataset,
)
data_preprocessor = dict(
    num_classes=num_classes,
    mean=[77.52425988, 77.52425988, 77.52425988],
    std=[51.8555656, 51.8555656, 51.8555656],
    to_rgb=True,
)
bgr_mean = [77.52425988, 77.52425988, 77.52425988]
bgr_std = [51.8555656, 51.8555656, 51.8555656]
train_pipeline = [
    dict(
        type="LoadImageRSNABreastAux",
        img_prefix=f"datasets/mmbreast/{dataset}/cleaned_images",
    ),
    dict(
        type="Resize",
        scale=(size[0] * 1.2, size[1] * 1.2),
        keep_ratio=True,
        interpolation="bicubic",
    ),
    dict(
        type="RandomFlip",
        prob=0.75,
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(
        type="RandAugment",
        policies="timm_increasing",
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.1,
        hparams=dict(pad_val=[47, 50, 79], interpolation="bicubic"),
    ),
    dict(type="TrainAugment"),
    dict(
        type="RandomErasing",
        erase_prob=0.25,
        mode="rand",
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[77.52425988, 77.52425988, 77.52425988],
        fill_std=[51.8555656, 51.8555656, 51.8555656],
    ),
    dict(type="ValTransform", size=size),
    dict(type="PackMxInputs"),
]
test_pipeline = [
    dict(
        type="LoadImageRSNABreastAux",
        img_prefix=f"datasets/mmbreast/{dataset}/cleaned_images",
    ),
    dict(type="ValTransform", size=size),
    dict(type="PackInputs"),
]
train_dataloader = dict(
    pin_memory=False,
    persistent_workers=True,
    collate_fn=dict(type="default_collate"),
    batch_size=batch_size,
    num_workers=16,
    dataset=dict(
        type="CsvGeneralDataset",
        ann_file=f"datasets/mmbreast/{dataset}/cleaned_label_split.csv",
        metainfo=dict(
            classes=(0, 1),
        ),
        split=fold,
        train=True,
        label_key="cancer",
        pipeline=train_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
    # sampler=dict(
    #     type="BalanceSamplerV2",
    #     batch_size=batch_size,
    #     num_sched_epochs=4,
    #     num_epochs=epochs,
    #     start_ratio=0.506,
    #     end_ratio=0.506,
    # ),
)
val_dataloader = dict(
    pin_memory=False,
    persistent_workers=True,
    collate_fn=dict(type="default_collate"),
    batch_size=batch_size,
    num_workers=16,
    dataset=dict(
        type="CsvGeneralDataset",
        ann_file=f"datasets/mmbreast/{dataset}/cleaned_label_split.csv",
        metainfo=dict(
            classes=(0, 1),
        ),
        split=fold,
        train=False,
        label_key="cancer",
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = dict(type="RSNAPFBeta")
test_dataloader = dict(
    pin_memory=False,
    persistent_workers=True,
    collate_fn=dict(type="default_collate"),
    batch_size=batch_size,
    num_workers=16,
    dataset=dict(
        type="CsvGeneralDataset",
        ann_file=f"datasets/mmbreast/{dataset}/cleaned_label_split.csv",
        metainfo=dict(
            classes=(0, 1),
        ),
        split=fold,
        train=False,
        label_key="cancer",
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
test_evaluator = dict(type="RSNAPFBeta")
optim_wrapper = dict(
    type="AmpOptimWrapper",
    accumulative_counts=8,
    optimizer=dict(
        type="AdamW",
        lr=0.00015,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys=dict(
            {
                ".absolute_pos_embed": dict(decay_mult=0.0),
                ".relative_position_bias_table": dict(decay_mult=0.0),
            }
        ),
    ),
)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=True,
        end=1,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-05,
        by_epoch=True,
        begin=1,
        T_max=epochs,
        convert_to_iter_based=True,
    ),
]
train_cfg = dict(by_epoch=True, max_epochs=epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=64)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=4,
        save_best="pfbeta",
        max_keep_ckpts=3,
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="VisualizationHook", enable=False),
)
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExponentialMovingAverage",
        momentum=0.02,
        update_buffers=True,
        priority=49,
    ),
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Visualizer", vis_backends=[dict(type="LocalVisBackend")])
log_level = "INFO"
load_from = "https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small-050_3rdparty_in1k_20221114-e0b86be1.pth"
resume = False
work_dir = f"./work_folder/{dataset}/mobilenet_v3-small_050/fold_{fold}"
fp16 = dict(loss_scale=256.0, velocity_accum_type="half", accum_type="half")
launcher = "none"
