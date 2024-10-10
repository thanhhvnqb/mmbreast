_base_ = [
    "base.py",
]
fold = 0
num_classes = 2
model = dict(
    type="BreastCancerAuxCls",
    backbone=dict(
        type="ResNet", depth=50, num_stages=4, out_indices=(3,), style="pytorch"
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
load_from = "https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
work_dir = f"./work_folder/from_imagenet/resnet50-fold_{fold}/"
