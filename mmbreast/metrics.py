from typing import Sequence, List

import numpy as np
from sklearn import metrics
import torch

from mmpretrain.registry import METRICS
from mmengine.evaluator import BaseMetric


def pfbeta_np(gts, preds, beta=1):
    preds = preds.clip(0, 1.0)
    y_true_count = gts.sum()
    ctp = preds[gts == 1].sum()
    cfp = preds[gts == 0].sum()
    beta_squared = beta * beta
    if ctp + cfp == 0:
        c_precision = 0.0
    else:
        c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        ret = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return ret
    else:
        return 0.0


def pfbeta(labels, predictions, beta=1.0):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + 1e-8)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


def _compute_fbeta(precision, recall, beta=1.0):
    if ((beta**2) * precision + recall) == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / ((beta**2) * precision + recall)


def compute_usual_metrics(gts, preds, beta=1.0, sample_weights=None):
    """Binary prediction only."""
    cfm = metrics.confusion_matrix(
        gts, preds, labels=[0, 1], sample_weight=sample_weights
    )

    tn, fp, fn, tp = cfm.ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    recall = tp / (tp + fn)
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    fbeta = _compute_fbeta(precision, recall, beta=beta)
    f1_score = metrics.f1_score(gts, preds, average="weighted")
    return {
        "acc": acc,
        "f1score": f1_score,
        "fbeta": fbeta,
        "precision": precision,
        "recall": recall,
    }


@METRICS.register_module(force=True)
class RSNAPFBeta(BaseMetric):

    def process(self, data_batch, data_samples: Sequence[dict]):
        for data_sample in data_samples:
            result = dict()
            # pred_label = data_sample["pred_label"]
            result["pred_scores"] = data_sample["pred_score"]
            result["gt_score"] = data_sample["gt_label"]

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        sort_by = "f1score"
        # concat
        target = torch.stack([res["gt_score"] for res in results])
        pred = torch.stack([res["pred_scores"][1] for res in results])
        target = target.squeeze().cpu().numpy().flatten()
        pred = pred.squeeze().cpu().numpy().flatten()
        assert len(target) == len(pred), f"target: {len(target)}, pred: {len(pred)}"

        # Probabilistic-fbeta
        pfbeta = pfbeta_np(target, pred, beta=1.0)
        # AUC
        fpr, tpr, _ = metrics.roc_curve(target, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        # PR-AUC
        precisions, recalls, _ = metrics.precision_recall_curve(target, pred)
        pr_auc = metrics.auc(recalls, precisions)

        ####
        # METRICS FOR CATEGORICAL PREDICTION #####
        ####
        # PER THRESHOLD METRIC
        per_thres_metrics = []
        for thres in np.arange(0.0, 1.0, 0.01):
            bin_preds = (pred > thres).astype(np.uint8)
            metric_at_thres = compute_usual_metrics(target, bin_preds, beta=1.0)

            per_thres_metrics.append((thres, metric_at_thres))

        per_thres_metrics.sort(key=lambda x: x[1][sort_by], reverse=True)

        # handle multiple thresholds with same scores
        top_score = per_thres_metrics[0][1][sort_by]
        same_scores = []
        for j, (thres, metric_at_thres) in enumerate(per_thres_metrics):
            if metric_at_thres[sort_by] == top_score:
                same_scores.append(abs(thres - 0.5))
            else:
                assert metric_at_thres[sort_by] < top_score
                break
        if len(same_scores) == 1:
            best_thres, best_metric = per_thres_metrics[0]
        else:
            # the nearer 0.5 threshold is --> better
            best_idx = np.argmin(np.array(same_scores))
            best_thres, best_metric = per_thres_metrics[best_idx]

        result_metrics = dict()

        result_metrics["pfbeta"] = pfbeta
        result_metrics["auc"] = auc
        result_metrics["pr_auc"] = pr_auc
        result_metrics["best_thres"] = best_thres
        result_metrics.update({f"best_{k}": v for k, v in best_metric.items()})

        return result_metrics
