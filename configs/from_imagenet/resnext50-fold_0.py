_base_ = [
    "base.py",
]
fold = 0
num_classes = 2
model = dict(
    type="BreastCancerAuxCls",
    backbone=dict(
        type="ResNeXt",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        groups=32,
        width_per_group=4,
        style="pytorch",
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=num_classes,
        in_channels=2048,
        loss=dict(type="SoftmaxEQLLoss", num_classes=num_classes),
    ),
)
resume = False
load_from = "https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth"
work_dir = f"./work_folder/from_imagenet/resnext50-fold_{fold}/"
