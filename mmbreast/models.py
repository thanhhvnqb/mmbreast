from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from mmpretrain.registry import MODELS
from mmpretrain.models.classifiers.image import ImageClassifier
from mmpretrain.models.heads import ClsHead
from mmpretrain.evaluation.metrics import Accuracy
from mmengine.dist import all_reduce as allreduce
from .transforms import MxDataSample


# From https://github.com/Ezra-Yu/ACCV2022_FGIA_1st
@MODELS.register_module(force=True)
class SoftmaxEQLLoss(_Loss):
    def __init__(
        self, num_classes, indicator="pos", loss_weight=1.0, tau=1.0, eps=1e-4
    ):
        super(SoftmaxEQLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ["pos", "neg", "pos_and_neg"], "Wrong indicator type!"
        self.indicator = indicator

        # initial variables
        self.register_buffer("pos_grad", torch.zeros(num_classes))
        self.register_buffer("neg_grad", torch.zeros(num_classes))
        self.register_buffer("pos_neg", torch.ones(num_classes))

    def forward(
        self,
        input,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if self.indicator == "pos":
            indicator = self.pos_grad.detach()
        elif self.indicator == "neg":
            indicator = self.neg_grad.detach()
        elif self.indicator == "pos_and_neg":
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError

        one_hot = F.one_hot(label, self.num_classes)
        self.targets = one_hot.detach()

        matrix = indicator[None, :].clamp(min=self.eps) / indicator[:, None].clamp(
            min=self.eps
        )
        factor = matrix[label.long(), :].pow(self.tau)

        cls_score = input + (factor.log() * (1 - one_hot.detach()))
        loss = F.cross_entropy(cls_score, label)
        return loss * self.loss_weight

    def collect_grad(self, grad):
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)

        allreduce(pos_grad)
        allreduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)


@MODELS.register_module(force=True)
class KLDivLoss(nn.KLDivLoss):
    pass


# From https://github.com/GOKORURI007/pytorch_arcface/blob/main/arcface.py
@MODELS.register_module()
class ArcfaceClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        scale=64,
        margin=0.5,
        easy_margin=False,
        init_cfg: Optional[dict] = dict(type="Normal", layer="Linear", std=0.01),
        **kwargs,
    ):
        super(ArcfaceClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_channels))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(pre_logits), F.normalize(self.weight)).clamp(
            -1 + 1e-7, 1 - 1e-7
        )
        sin_theta = torch.sqrt(
            (1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7)
        )
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size(), device=pre_logits.device)
        target = torch.ones(cos_theta.size(0), device=pre_logits.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output_1 = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)

        one_hot = torch.zeros(cos_theta.size(), device=pre_logits.device)
        target = torch.zeros(cos_theta.size(0), device=pre_logits.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)

        output[:, 1] = output_1[:, 1]
        output *= self.scale
        return output

    def loss(
        self, feats: Tuple[torch.Tensor], data_samples: List[MxDataSample], **kwargs
    ) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        pre_logits = self.pre_logits(feats)

        # Unpack data samples and pack targets
        if "gt_score" in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(pre_logits), F.normalize(self.weight)).clamp(
            -1 + 1e-7, 1 - 1e-7
        )
        sin_theta = torch.sqrt(
            (1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7)
        )
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size(), device=pre_logits.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        cls_score = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        cls_score *= self.scale

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs
        )
        losses["loss"] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, (
                "If you enable batch augmentation "
                "like mixup during training, `cal_acc` is pointless."
            )
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update({f"accuracy_top-{k}": a for k, a in zip(self.topk, acc)})

        return losses


@MODELS.register_module(force=True)
class BreastCancerAuxCls(ImageClassifier):
    def __init__(
        self,
        backbone: dict,
        model_config: str,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        pretrained: Optional[str] = None,
        train_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        with_auxiliary: bool = True,
        with_arcface: bool = False,
        arcface_cfg: Optional[dict] = None,
    ):
        super().__init__(
            backbone, neck, head, pretrained, train_cfg, data_preprocessor, init_cfg
        )
        in_channels = head["in_channels"]
        self.model_config = model_config
        self.with_auxiliary = with_auxiliary
        if with_auxiliary:
            if model_config in ["rsna", "bmcd", "cddcesm", "miniddsm", "vindr"]:
                self.nn_view = nn.Linear(in_channels, 6)
            else:
                self.nn_view = None
            self.nn_BIRADS = nn.Linear(in_channels, 5)
            if model_config in ["rsna"]:
                self.nn_difficulty = nn.Linear(in_channels, 2)
            else:
                self.nn_difficulty = None
            if model_config in ["rsna", "bmcd", "cddcesm", "miniddsm", "vindr"]:
                self.nn_density = nn.Linear(in_channels, 4)
            else:
                self.nn_density = None

            self.nn_sigmoid = nn.Linear(in_channels, 3)

            self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.01)
            self.sigmoid_loss = torch.nn.BCEWithLogitsLoss()

            self.BIRADS_lossfn = SoftmaxEQLLoss(num_classes=5)
            self.diff_lossfn = SoftmaxEQLLoss(num_classes=2)
            self.density_lossfn = SoftmaxEQLLoss(num_classes=4)
        self.with_arcface = with_arcface
        if with_arcface:
            loss_module_arcface = arcface_cfg.get(
                "loss_module", dict(type="CrossEntropyLoss", loss_weight=1.0)
            )
            if not isinstance(loss_module_arcface, nn.Module):
                loss_module_arcface = MODELS.build(loss_module_arcface)
            self.loss_module_arcface = loss_module_arcface
            margin = arcface_cfg.get("margin", 0.5)
            self.arcface_weight = arcface_cfg.get("arcface_weight", 1.0)
            self.scale = arcface_cfg.get("scale", 64)
            self.margin = margin
            self.easy_margin = arcface_cfg.get("easy_margin", False)
            self.ce = nn.CrossEntropyLoss()
            self.weight = nn.Parameter(
                torch.FloatTensor(self.head.num_classes, in_channels)
            )
            self.cos_m = math.cos(margin)
            self.sin_m = math.sin(margin)
            self.th = math.cos(math.pi - margin)
            self.mm = math.sin(math.pi - margin) * margin

    def loss(self, inputs: torch.Tensor, data_samples: List[MxDataSample]) -> dict:
        feats = self.extract_feat(inputs)
        # From https://github.com/cornell-zhang/FracBNN/blob/main/imagenet.py#L167
        if isinstance(self.head.loss_module, KLDivLoss):
            # Unpack data samples and pack targets
            if "gt_score" in data_samples[0]:
                # Batch augmentation may convert labels to one-hot format scores.
                target = torch.stack([i.gt_score for i in data_samples])
            else:
                target = torch.cat([i.gt_label for i in data_samples])
            outputs = self.head(feats)
            loss_cancer = dict()
            one_hot = torch.zeros(outputs.size(), device="cuda")
            one_hot.scatter_(1, target.view(-1, 1).long(), 1)
            loss = self.head.loss_module(
                outputs.log_softmax(dim=1), one_hot.softmax(dim=1)
            )
            loss_cancer["loss"] = loss
        else:
            loss_cancer = self.head.loss(feats, data_samples)

        if self.with_auxiliary:
            cancer_target = torch.cat([i.gt_label for i in data_samples]).to(
                inputs.get_device()
            )
            aux_target = torch.stack([i.aux_label.label for i in data_samples]).to(
                inputs.get_device()
            )
            feats = feats[-1]
            aux_weight = 0.1

            if self.nn_view:
                loss_view = self.ce_loss(
                    self.nn_view(feats), aux_target[:, 0].to(torch.long)
                )
                loss_cancer.update(
                    {
                        "loss_view": loss_view * aux_weight,
                    }
                )

            BIRADS_mask = aux_target[:, 1] > -1
            if torch.sum(BIRADS_mask) > 0:
                loss_BIRADS = self.BIRADS_lossfn(
                    self.nn_BIRADS(feats)[BIRADS_mask, :],
                    aux_target[:, 1][BIRADS_mask].to(torch.long),
                )
                loss_cancer.update(
                    {
                        "loss_BIRADS": loss_BIRADS * aux_weight,
                    }
                )
            if self.nn_difficulty:
                difficulty_mask = torch.logical_and(
                    aux_target[:, 3] > -1, cancer_target < 1
                )
                if torch.sum(difficulty_mask) > 0:
                    loss_difficulty = self.diff_lossfn(
                        self.nn_difficulty(feats)[difficulty_mask, :],
                        aux_target[:, 3][difficulty_mask].to(torch.long),
                    )
                    loss_cancer.update(
                        {
                            "loss_difficulty": loss_difficulty * aux_weight,
                        }
                    )

            sig_out = self.nn_sigmoid(feats)
            if self.model_config in ["rsna"]:
                invasive_mask = torch.logical_and(
                    aux_target[:, 2] > -1, cancer_target > 0
                )
                if torch.sum(invasive_mask) > 0:
                    loss_invasive = self.sigmoid_loss(
                        sig_out[:, 0][invasive_mask], aux_target[:, 2][invasive_mask]
                    )
                    loss_cancer.update(
                        {
                            "loss_invasive": loss_invasive * aux_weight,
                        }
                    )

            # implant_mask = aux_target[:, 4] < 255
            # if torch.sum(implant_mask) > 0:
            #     loss_implant = self.sigmoid_loss(sig_out[:, 1][implant_mask],
            #                                      aux_target[:, 4][implant_mask])
            #     loss_cancer.update({'loss_implant': loss_implant * aux_weight, })

            age_mask = aux_target[:, 5] > -1
            if torch.sum(age_mask) > 0:
                loss_age = self.sigmoid_loss(
                    sig_out[:, 2][age_mask], aux_target[:, 5][age_mask] / 100
                )
                loss_cancer.update(
                    {
                        "loss_age": loss_age * aux_weight,
                    }
                )

            if self.nn_density:
                density_mask = aux_target[:, 6] > -1
                if torch.sum(density_mask) > 0:
                    loss_density = self.density_lossfn(
                        self.nn_density(feats)[density_mask],
                        aux_target[:, 6][density_mask].to(torch.long),
                    )
                    loss_cancer.update({"loss_density": loss_density * aux_weight})
        if self.with_arcface:
            loss_arcface = self.loss_arcface(feats, cancer_target)
            loss_cancer.update({"loss_arcface": loss_arcface * self.arcface_weight})

        return loss_cancer

    def psuedo_label(self, inputs, **kwargs):
        feats = self.extract_feat(inputs)
        feats = feats[-1]
        BIRADS = self.nn_BIRADS(feats).softmax(dim=1)
        Density = self.nn_density(feats).softmax(dim=1)
        return (BIRADS, Density)

    def loss_arcface(self, embedding: torch.Tensor, target, **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(
            -1 + 1e-7, 1 - 1e-7
        )
        sin_theta = torch.sqrt(
            (1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7)
        )
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size(), device=embedding.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        cls_score = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        cls_score *= self.scale

        # compute loss
        loss = self.loss_module_arcface(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs
        )
        return loss
