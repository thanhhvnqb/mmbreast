_base_ = [
    "base.py",
]
fold = 0
num_classes = 2
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
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        init_cfg=None,
    ),
    with_arcface=True,
    arcface_cfg=dict(loss_module=dict(type="CrossEntropyLoss", loss_weight=0.05)),
    init_cfg=dict(type="TruncNormal", layer=["Conv2d", "Linear"], std=0.02, bias=0.0),
)
resume = False
load_from = "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty-ra-noisystudent_in1k_20221103-a4ab5fd6.pth"
work_dir = f"./work_folder/from_imagenet/efficient-b3-fold_{fold}-ce_loss-arcface_0.5/"
