_base_ = [
    "base.py",
]
fold = 0
num_classes = 2
model = dict(
    type="BreastCancerAuxCls",
    backbone=dict(
        type="MobileNetV3",
        arch="large",
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="StackedLinearClsHead",
        num_classes=num_classes,
        in_channels=960,
        mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type="HSwish"),
        loss=dict(type="SoftmaxEQLLoss", num_classes=num_classes),
        init_cfg=dict(type="Normal", layer="Linear", mean=0.0, std=0.01, bias=0.0),
        cal_acc=False,
    ),
)
resume = False
load_from = "https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_large-3ea3c186.pth"
work_dir = f"./work_folder/from_imagenet/mobilenet_v3-large-fold_{fold}/"
