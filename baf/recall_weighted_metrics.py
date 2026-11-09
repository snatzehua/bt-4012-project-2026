"""Recall-weighted operating rules fitted on earlier prediction months only."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import ranking
from .improvement_metrics import confusion_metrics

POLICIES = ["pooled", "latest", "rank_fraction"]


def fbeta(tp, fp, fn, beta=2):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    numerator = (1 + beta * beta) * np.asarray(tp, dtype=float)
    denominator = numerator + beta * beta * np.asarray(fn) + np.asarray(fp)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def check_data(data):
    y, s, ids = (np.asarray(data[k]) for k in ["labels", "scores", "row_ids"])
    if any(a.ndim != 1 for a in [y, s, ids]) or not len(y) or len(y) != len(s) or len(y) != len(ids):
        raise ValueError("Invalid prediction population")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all() or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid labels, scores or row IDs")


def aligned(a, b):
    check_data(a)
    check_data(b)
    if not np.array_equal(a["row_ids"], b["row_ids"]) or not np.array_equal(a["labels"], b["labels"]):
        raise ValueError("Evaluation populations differ")


def pool(items):
    result = {k: np.concatenate([a[k] for a in items]) for k in ["labels", "scores", "row_ids"]}
    check_data(result)
    return result


def fit_policy(data, kind="pooled", beta=2):
    check_data(data)
    if kind not in POLICIES:
        raise ValueError("Unknown policy")
    y, s = data["labels"], data["scores"]
    order = ranking(s, data["row_ids"])
    tp = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    values = fbeta(tp, k - tp, y.sum() - tp, beta)
    if kind == "rank_fraction":
        j = int(np.argmax(values)) if y.sum() else -1
        return dict(kind=kind, numerator=j + 1, denominator=len(y), beta=beta)
    ends = np.r_[np.flatnonzero(s[order][:-1] != s[order][1:]), len(y) - 1]
    j = int(ends[np.argmax(values[ends])])
    threshold = float(s[order][j]) if y.sum() else float(np.nextafter(s.max(), np.inf))
    return dict(kind=kind, threshold=threshold, beta=beta)


def decisions(data, policy):
    check_data(data)
    if policy["kind"] == "rank_fraction":
        num, den = policy["numerator"], policy["denominator"]
        if den <= 0 or not 0 <= num <= den:
            raise ValueError("Invalid learned fraction")
        # Exact rational ceiling avoids an extra review due to floating-point error.
        k = (num * len(data["labels"]) + den - 1) // den
        pred = np.zeros(len(data["labels"]), dtype=bool)
        pred[ranking(data["scores"], data["row_ids"])[:k]] = True
        return pred
    if policy["kind"] not in ["pooled", "latest"]:
        raise ValueError("Unknown policy")
    return data["scores"] >= policy["threshold"]


def measure(data, policy):
    pred = decisions(data, policy)
    y = data["labels"].astype(bool)
    result = confusion_metrics(np.sum(~y & ~pred), np.sum(~y & pred), np.sum(y & ~pred), np.sum(y & pred))
    result.update(f3=float(fbeta(result["tp"], result["fp"], result["fn"], 3)),
                  flagged_fraction=float(pred.mean()), n=len(y), fraud=int(y.sum()),
                  average_precision=float(average_precision_score(y, data["scores"])) if y.any() else 0.,
                  roc_auc=float(roc_auc_score(y, data["scores"])) if 0 < y.sum() < len(y) else None)
    return result


def earlier_months(months, evaluation):
    if not months or any(m >= evaluation for m in months):
        raise ValueError("Policy sources must precede evaluation")
    return sorted(set(months))


def learn(history, kind, evaluation, beta=2):
    months = earlier_months(list(history), evaluation)
    used = months[-1:] if kind == "latest" else months
    policy = fit_policy(pool([history[m] for m in used]), kind, beta)
    return {**policy, "source_months": used}


def choose_member(history, specs, family, evaluation, beta=2):
    """Member selection can inspect only the supplied earlier predictions."""
    scores = []
    for spec in specs:
        if spec["family"] != family:
            continue
        items = history[spec["id"]]
        months = earlier_months(list(items), evaluation)
        policy = fit_policy(pool([items[m] for m in months]), beta=beta)
        score = np.mean([measure(items[m], policy)["f2"] for m in months])
        scores.append((-score, spec["id"]))
    if not scores:
        raise ValueError("No eligible member")
    return min(scores)[1]


def blend(a, b, weight):
    aligned(a, b)
    if not 0 <= weight <= 1:
        raise ValueError("Invalid blend weight")
    return dict(labels=a["labels"], row_ids=a["row_ids"], scores=weight * a["scores"] + (1 - weight) * b["scores"])


def paired_f2(data, policies, baseline, replicates=500, seed=1027):
    names = list(data)
    ref = data[baseline]
    y = np.asarray(ref["labels"], dtype=np.int64)
    masks = {}
    for name in names:
        aligned(ref, data[name])
        masks[name] = decisions(data[name], policies[name]).astype(np.int64)
    estimates = {n: measure(data[n], policies[n])["f2"] for n in names}
    draws = {n: [] for n in names if n != baseline}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
        positives = np.dot(counts, y)
        values = {}
        for n in names:
            tp = np.dot(counts, y * masks[n])
            fp = np.dot(counts, (1 - y) * masks[n])
            values[n] = float(fbeta(tp, fp, positives - tp))
        for n in draws:
            draws[n].append(values[n] - values[baseline])
    return [dict(candidate=n, baseline=baseline, difference=estimates[n] - estimates[baseline],
                 ci_low=float(np.quantile(v, .025)), ci_high=float(np.quantile(v, .975)),
                 replicates=replicates, uncertainty="conditional paired row bootstrap; fixed decisions; no selection correction")
            for n, v in draws.items()]
