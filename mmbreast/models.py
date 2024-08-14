from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from mmpretrain.registry import MODELS
from mmpretrain.models.classifiers.image import ImageClassifier
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
class RSNAAuxCls(ImageClassifier):
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
    ):
        super().__init__(
            backbone, neck, head, pretrained, train_cfg, data_preprocessor, init_cfg
        )
        in_channels = head["in_channels"]
        self.model_config = model_config
        if model_config in ["rsna", "bmcd", "cddcesm", "miniddsm", "vindr"]:
            self.nn_view = nn.Linear(in_channels, 6)
        else:
            self.nn_view = None
        self.nn_BIRADS = nn.Linear(in_channels, 3)
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

        self.BIRADS_lossfn = SoftmaxEQLLoss(num_classes=3)
        self.diff_lossfn = SoftmaxEQLLoss(num_classes=2)
        self.density_lossfn = SoftmaxEQLLoss(num_classes=4)

    def loss(self, inputs: torch.Tensor, data_samples: List[MxDataSample]) -> dict:
        feats = self.extract_feat(inputs)
        loss_cancer = self.head.loss(feats, data_samples)
        cancer_target = torch.stack([i.gt_label for i in data_samples]).to("cuda:0")
        aux_target = torch.stack([i.aux_label.label for i in data_samples]).to("cuda:0")
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
                aux_target[:, 3] > -1, cancer_target[:, 0] < 1
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
                aux_target[:, 2] > -1, cancer_target[:, 0] > 0
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

        return loss_cancer

    def psuedo_label(self, inputs, **kwargs):
        feats = self.extract_feat(inputs)
        feats = feats[-1]
        BIRADS = self.nn_BIRADS(feats).softmax(dim=1)
        Density = self.nn_density(feats).softmax(dim=1)
        return (BIRADS, Density)
