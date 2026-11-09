"""Data, sampling, models and operating metrics shared by the study."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
SENTINEL_MINUS_ONE = [
    "prev_address_months_count", "current_address_months_count", "bank_months_count",
    "session_length_in_minutes", "device_distinct_emails_8w",
]
EXCLUDED = ["fraud_bool", "month", "device_fraud_count"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def read_config(path="configs/study.json"):
    return json.loads(Path(path).read_text())


def load_data(config):
    if digest(config["data_path"]) != config["data_sha256"]:
        raise ValueError("Data checksum differs from the protocol")
    frame = pd.read_csv(config["data_path"])
    if len(frame) != 1_000_000 or frame.shape[1] != 32:
        raise ValueError("Unexpected dataset shape")
    if not frame.fraud_bool.isin([0, 1]).all() or not frame.month.isin(range(8)).all():
        raise ValueError("Unexpected labels or months")
    return frame


def features(frame):
    result = frame.drop(columns=EXCLUDED).copy()
    for col in SENTINEL_MINUS_ONE:
        result.loc[result[col] == -1, col] = np.nan
    result.loc[result.intended_balcon_amount < 0, "intended_balcon_amount"] = np.nan
    return result


def recent_indices(months, eligible, k, tie_seed):
    eligible = np.asarray(eligible, dtype=np.int64)
    if k > len(eligible) or k < 1:
        raise ValueError("Infeasible recent sample size")
    # Target-independent priority, including random selection within a month boundary.
    priority = np.random.default_rng(tie_seed).random(len(months))
    order = np.lexsort((priority[eligible], -np.asarray(months)[eligible]))
    return np.sort(eligible[order[:k]])


def matched_indices(labels, eligible, recent, seed):
    labels = np.asarray(labels)
    eligible = np.asarray(eligible, dtype=np.int64)
    recent = np.asarray(recent, dtype=np.int64)
    if not np.isin(recent, eligible).all():
        raise ValueError("Recent sample contains ineligible rows")
    rng = np.random.default_rng(seed)
    parts = []
    for label in [0, 1]:
        pool = eligible[labels[eligible] == label]
        count = int(np.sum(labels[recent] == label))
        if count > len(pool):
            raise ValueError("Infeasible matched class count")
        parts.append(rng.choice(pool, count, replace=False))
    result = np.sort(np.concatenate(parts))
    if len(result) != len(recent) or labels[result].sum() != labels[recent].sum():
        raise AssertionError("Matched sampling violated its controls")
    return result


def make_model(name, weighted, config, columns):
    numeric = [c for c in columns if c not in CATEGORICAL]
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if name == "logistic":
        numeric_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL),
    ], sparse_threshold=1.0)
    params = config["model_parameters"][name].copy()
    params.update(random_state=config["model_seed"], class_weight="balanced" if weighted else None)
    if name == "logistic":
        model = LogisticRegression(**params)
    elif name == "lightgbm":
        model = LGBMClassifier(n_jobs=config["threads"], **params)
    elif name == "random_forest":
        model = RandomForestClassifier(n_jobs=config["threads"], **params)
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def budget_name(fraction):
    return f"{fraction * 100:g}pct"


def ranking(scores, row_ids):
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite prediction scores")
    return np.lexsort((np.asarray(row_ids), -scores))


def budget_metrics(labels, scores, row_ids, fraction):
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0 or not 0 < fraction <= 1:
        raise ValueError("Invalid review population or fraction")
    k = min(n, math.ceil(n * fraction))
    tp = int(labels[ranking(scores, row_ids)[:k]].sum())
    positives = int(labels.sum())
    return {"reviewed": k, "tp": tp, "fp": k - tp, "fn": positives - tp,
            "tn": n - k - (positives - tp), "precision": tp / k,
            "recall": tp / positives if positives else 0.0}


def threshold_metrics(labels, scores, threshold, prefix):
    pred = np.asarray(scores) >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {f"{prefix}_threshold": float(threshold), f"{prefix}_precision": float(precision),
            f"{prefix}_recall": float(recall), f"{prefix}_f1": float(f1),
            **{f"{prefix}_{key}": int(value) for key, value in zip(["tn", "fp", "fn", "tp"], [tn, fp, fn, tp])}}


def best_f1_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(labels, scores, row_ids, fractions, threshold=None):
    result = {"n_eval": int(len(labels)), "fraud_eval": int(np.sum(labels)),
              "average_precision": float(average_precision_score(labels, scores)),
              "roc_auc": float(roc_auc_score(labels, scores))}
    result.update(threshold_metrics(labels, scores, 0.5, "default"))
    for fraction in fractions:
        key = budget_name(fraction)
        result.update({f"{metric}_at_{key}": value for metric, value in budget_metrics(labels, scores, row_ids, fraction).items()})
    if threshold is not None:
        result.update(threshold_metrics(labels, scores, threshold, "selected"))
    return result


def bootstrap_recall(labels, order, multiplicity, fraction):
    """Recall of a ranked bootstrap sample, retaining duplicated sampled rows."""
    weights = multiplicity[order]
    quota = math.ceil(fraction * int(weights.sum()))
    before = np.cumsum(weights) - weights
    selected = np.minimum(weights, np.maximum(quota - before, 0))
    positives = int(np.dot(labels, multiplicity))
    return float(np.dot(labels[order], selected) / positives) if positives else 0.0
