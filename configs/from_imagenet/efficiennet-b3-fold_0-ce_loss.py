_base_ = [
    "base.py",
]

num_classes = 2
fold = 0
size = (2048, 1024)  # h, w
batch_size = 16
model = dict(
    type="BreastCancerAuxCls",
    backbone=dict(
        type="EfficientNet",
        arch="b3",
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=num_classes,
        in_channels=1536,
        loss=dict(type="CrossEntropyLoss", num_classes=num_classes),
        init_cfg=None,
    ),
    init_cfg=dict(type="TruncNormal", layer=["Conv2d", "Linear"], std=0.02, bias=0.0),
)
train_pipeline = [
    dict(
        type="LoadImageRSNABreastAux",
        img_prefix="../datasets/mmbreast/",
    ),
    dict(
        type="CropBreastRegion",
        threshhold=0.1,
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
        img_prefix="../datasets/mmbreast/",
    ),
    dict(
        type="CropBreastRegion",
        threshhold=0.1,
    ),
    dict(type="ValTransform", size=size),
    dict(type="PackMxInputs"),
]
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        split=fold,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        split=fold,
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        split=fold,
        pipeline=test_pipeline,
    ),
)
resume = False
load_from = "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty-ra-noisystudent_in1k_20221103-a4ab5fd6.pth"
work_dir = f"./work_folder/from_imagenet/efficient-b3-fold_{fold}-ce_loss/"
