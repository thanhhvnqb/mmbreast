_base_ = [
    "base.py",
]
fold = 0
num_classes = 2
model = dict(
    type="BreastCancerAuxCls",
    backbone=dict(
        type="ConvNeXt",
        arch="nano",
        drop_path_rate=0.1,
        layer_scale_init_value=0.0,
        use_grn=True,
    ),
    head=dict(
        type="LinearClsHead",
        num_classes=num_classes,
        in_channels=640,
        loss=dict(type="SoftmaxEQLLoss", num_classes=num_classes),
        init_cfg=None,
    ),
    init_cfg=dict(type="TruncNormal", layer=["Conv2d", "Linear"], std=0.02, bias=0.0),
)
resume = False
load_from = "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-nano_3rdparty-fcmae_in1k_20230104-3dd1f29e.pth"
work_dir = f"./work_folder/from_imagenet/convnextv2-nano-fold_{fold}/"
