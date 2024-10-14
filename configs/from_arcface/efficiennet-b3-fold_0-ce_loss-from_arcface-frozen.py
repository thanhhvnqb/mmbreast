_base_ = [
    "../from_imagenet/base.py",
]
fold = 0
num_classes = 2
model = dict(
    type="BreastCancerAuxCls",
    backbone=dict(
        type="EfficientNet",
        arch="b3",
        frozen_stages=7,
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=num_classes,
        in_channels=1536,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        init_cfg=None,
    ),
    init_cfg=dict(type="TruncNormal", layer=["Conv2d", "Linear"], std=0.02, bias=0.0),
)
resume = False
# load_from = "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty-ra-noisystudent_in1k_20221103-a4ab5fd6.pth"
load_from = (
    "./work_folder/from_imagenet/efficient-b3-fold_0-arcface/best_pfbeta_epoch_50.pth"
)
work_dir = (
    f"./work_folder/from_arceface/efficient-b3-fold_{fold}-ce_loss-from_arcface-frozen/"
)
