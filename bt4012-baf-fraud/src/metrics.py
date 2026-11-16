"""Evaluation metrics for rare-event fraud detection.

Accuracy is never used: with ~1% fraud, flagging nothing scores ~99%.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y, s, fpr):
    """Score threshold that flags `fpr` of the legitimate applications."""
    return float(np.quantile(s[y == 0], 1 - fpr))


def threshold_at_capacity(s, capacity):
    """Score threshold that flags the top `capacity` share of all applications."""
    return float(np.quantile(s, 1 - capacity))


def at_threshold(y, s, thr):
    """Confusion-matrix metrics when we flag every score above `thr`."""
    pred = s > thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "alert_rate": float(pred.mean())}


def recall_at_fpr(y, s, fpr=0.05):
    """BAF benchmark metric: recall when the threshold is set on the SAME data
    so that exactly `fpr` of legitimate applications are flagged."""
    return at_threshold(y, s, threshold_at_fpr(y, s, fpr))["recall"]


def ranking_metrics(y, s, fpr=0.05):
    return {"pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)),
            f"recall_at_{int(fpr * 100)}fpr": recall_at_fpr(y, s, fpr)}


def capacity_metrics(y, s, capacities):
    """If analysts can review only the top X% of applications, how much fraud do they catch?"""
    out = {}
    for c in capacities:
        m = at_threshold(y, s, threshold_at_capacity(s, c))
        tag = f"top{c * 100:g}pct"
        out[f"{tag}_precision"] = m["precision"]
        out[f"{tag}_recall"] = m["recall"]
    return out


def bootstrap_ci(y, s, fn, n=200, seed=0, alpha=0.05):
    """Percentile bootstrap confidence interval for any metric fn(y, s)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].sum() == 0:
            continue
        vals.append(fn(y[b], s[b]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def expected_cost(y, s, thr, cost_fn, cost_fp=1.0):
    m = at_threshold(y, s, thr)
    return m["fn"] * cost_fn + m["fp"] * cost_fp


def best_cost_threshold(y, s, cost_fn, cost_fp=1.0, n_grid=400):
    """Threshold that minimises total cost on (validation) data."""
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.9999, n_grid)))
    costs = [expected_cost(y, s, t, cost_fn, cost_fp) for t in grid]
    return float(grid[int(np.argmin(costs))])


def group_rates(y, s, thr, groups):
    """FPR, recall and alert rate per group, plus the predictive-equality ratio
    (lowest FPR / highest FPR; 1.0 = equal false-positive burden)."""
    rows = {}
    for g in np.unique(groups):
        mask = groups == g
        rows[g] = at_threshold(y[mask], s[mask], thr)
    fprs = [r["fpr"] for r in rows.values()]
    ratio = min(fprs) / max(fprs) if max(fprs) > 0 else 1.0
    return rows, ratio
