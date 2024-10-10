_base_ = [
    "base.py",
]
fold = 0
num_classes = 2
model = dict(
    type="BreastCancerAuxCls",
    backbone=dict(type="MobileViT", arch="small"),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=num_classes,
        in_channels=640,
        loss=dict(type="SoftmaxEQLLoss", num_classes=num_classes),
    ),
)
resume = False
load_from = "https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth"
work_dir = f"./work_folder/from_imagenet/mobilevit-small-fold_{fold}/"
