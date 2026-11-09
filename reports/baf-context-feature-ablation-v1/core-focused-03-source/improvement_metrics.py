"""Operating policies and paired, fixed-model uncertainty for improvements."""
from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

from .common import bootstrap_recall, budget_metrics, budget_name, ranking

POLICIES = ["f1", "f2", "precision30", "recall50"]


def confusion_metrics(tn, fp, fn, tp):
    tn, fp, fn, tp = map(float, (tn, fp, fn, tp))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                f2=5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0.,
                mcc=(tp * tn - fp * fn) / denom if denom else 0.,
                balanced_accuracy=(recall + (tn / (tn + fp) if tn + fp else 0.)) / 2,
                fpr=fp / (fp + tn) if fp + tn else 0.,
                accuracy=(tp + tn) / (tn + fp + fn + tp),
                flagged=int(tp + fp), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def threshold_stats(labels, scores, threshold):
    y, pred = np.asarray(labels, dtype=bool), np.asarray(scores) >= threshold
    result = confusion_metrics(np.sum(~y & ~pred), np.sum(~y & pred), np.sum(y & ~pred), np.sum(y & pred))
    return {"threshold": float(threshold), **result, "flagged_fraction": result["flagged"] / len(y)}


def choose_policies(labels, scores, precision_target=.3, recall_target=.5):
    p, r, t = precision_recall_curve(labels, scores)
    p, r = p[:-1], r[:-1]
    result = {}
    for beta in [1, 2]:
        score = (1 + beta ** 2) * p * r / np.maximum(beta ** 2 * p + r, 1e-15)
        result[f"f{beta}"] = float(t[np.flatnonzero(score == score.max())[-1]])
    feasible = (p >= precision_target) & (r > 0)
    if feasible.any():
        best_recall = r[feasible].max()
        result["precision30"] = float(t[np.flatnonzero(feasible & (r == best_recall))[-1]])
    else:
        result["precision30"] = float(np.nextafter(np.max(scores), np.inf))
    feasible = r >= recall_target
    best_precision = p[feasible].max()
    result["recall50"] = float(t[np.flatnonzero(feasible & (p == best_precision))[-1]])
    return result


def rank_metrics(labels, scores, row_ids, fractions):
    result = dict(n_eval=len(labels), fraud_eval=int(np.sum(labels)),
                  average_precision=float(average_precision_score(labels, scores)),
                  roc_auc=float(roc_auc_score(labels, scores)), brier=float(brier_score_loss(labels, scores)))
    for fraction in fractions:
        budget = budget_metrics(labels, scores, row_ids, fraction)
        metric = confusion_metrics(budget["tn"], budget["fp"], budget["fn"], budget["tp"])
        result.update({f"{k}_at_{budget_name(fraction)}": v for k, v in {**budget, **metric}.items()})
    result.update({f"default_{k}": v for k, v in threshold_stats(labels, scores, .5).items()})
    return result


def weighted_ap(labels, order, score_group_ends, multiplicity):
    y, w = np.asarray(labels)[order], np.asarray(multiplicity)[order]
    tp = np.cumsum(y * w)[score_group_ends]
    count = np.cumsum(w)[score_group_ends]
    delta_tp = np.diff(np.r_[0, tp])
    return float(np.sum(np.divide(tp, count, out=np.zeros_like(tp, dtype=float), where=count > 0) * delta_tp) / tp[-1]) if tp[-1] else 0.


def paired_intervals(predictions, thresholds, baseline, replicates, seed):
    """Row bootstrap preserves paired cases, score ties and frozen policies."""
    ref = predictions[baseline]
    y, ids = ref["labels"], ref["row_ids"]
    prepared = {}
    for name, data in predictions.items():
        if not np.array_equal(data["row_ids"], ids) or not np.array_equal(data["labels"], y):
            raise ValueError("Paired intervals require identical evaluation cases")
        order = ranking(data["scores"], ids)
        sorted_scores = data["scores"][order]
        ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y) - 1]
        pred = data["scores"] >= thresholds[name]["f1"]
        prepared[name] = (order, ends, pred)
    def metrics(name, weights):
        order, ends, pred = prepared[name]
        tp, fp, fn = float(np.dot(weights, y * pred)), float(np.dot(weights, (1 - y) * pred)), float(np.dot(weights, y * ~pred))
        return np.array([weighted_ap(y, order, ends, weights), bootstrap_recall(y, order, weights, .01),
                         2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.])
    estimates = {name: metrics(name, np.ones(len(y), dtype=int)) for name in predictions}
    values = {name: [] for name in predictions if name != baseline}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        weights = np.bincount(rng.integers(0, len(y), size=len(y)), minlength=len(y))
        reference = metrics(baseline, weights)
        for name in values:
            values[name].append(metrics(name, weights) - reference)
    rows = []
    for name, draws in values.items():
        interval = np.quantile(draws, [.025, .975], axis=0)
        for i, metric in enumerate(["average_precision", "recall_at_1pct", "f1_policy"]):
            rows.append(dict(candidate=name, baseline=baseline, metric=metric,
                             difference=float(estimates[name][i] - estimates[baseline][i]),
                             ci_low=float(interval[0, i]), ci_high=float(interval[1, i]), replicates=replicates,
                             uncertainty="conditional paired row bootstrap; fixed models and thresholds; no search/multiplicity correction"))
    return rows
