"""Separate exploratory development, freeze and consumed-holdout comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import digest, features, load_data, read_config, write_json
from .engineering import correct_prior, engineer, make_estimator, recency_weights, undersample_indices
from .improvement_metrics import POLICIES, choose_policies, paired_intervals, rank_metrics, threshold_stats

SOURCES = ["baf/common.py", "baf/engineering.py", "baf/improvement_metrics.py", "baf/improvements.py"]


def fingerprint(path):
    return {"config": digest(path), "sources": {p: digest(p) for p in SOURCES},
            "software": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "joblib", "threadpoolctl"]},
            "python": platform.python_version()}


def guard_preservation(config):
    for path, expected in config["preserve_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Original-study artifact changed: {path}")
    out = Path(config["output_dir"])
    snapshot = out / "original_local_snapshot.json"
    if snapshot.exists():
        previous = read_config(snapshot)
        for path, expected in previous.items():
            if digest(path) != expected:
                raise ValueError(f"Original local artifact changed: {path}")
    else:
        local = Path("outputs/baf-study-v1/selection.json")
        write_json(snapshot, {str(local): digest(local)} if local.exists() else {})


def paths(config, phase, month, name):
    stem = f"{phase}-m{month}-{name}"
    out = Path(config["output_dir"])
    return out / "runs" / f"{stem}.json", out / "predictions" / f"{stem}.npz"


def prediction(config, phase, month, name):
    _, path = paths(config, phase, month, name)
    with np.load(path) as data:
        return {key: data[key] for key in ["labels", "scores", "row_ids"]}


def save_prediction(path, ids, y, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, row_ids=ids, labels=y, scores=scores)


def fit_run(frame, base, engineered, spec, month, phase, config, identity):
    result_path, pred_path = paths(config, phase, month, spec["id"])
    if result_path.exists() and pred_path.exists():
        result = read_config(result_path)
        if result["fingerprint"] != identity:
            raise ValueError("Stale improvement cache; choose a new output directory")
        print("Cached", result_path.stem, flush=True)
        return result
    eligible = frame.index[frame.month < month].to_numpy()
    evaluation = frame.index[frame.month == month].to_numpy()
    if not len(evaluation) or not len(eligible):
        raise ValueError("Empty chronological split")
    labels = frame.fraud_bool
    chosen = eligible
    if "negatives_per_positive" in spec:
        # Global row IDs are preserved even when development excludes month 7.
        chosen = undersample_indices(labels.reindex(range(frame.index.max() + 1)).to_numpy(), eligible, spec["negatives_per_positive"], config["seed"])
    assert not np.intersect1d(chosen, evaluation).size
    assert frame.loc[chosen, "month"].max() < month
    x = engineered if spec["engineered"] else base
    x_train, x_eval = x.loc[chosen], x.loc[evaluation]
    model, adapter = make_estimator(spec, config, x_train)
    if adapter is not None:
        x_train, x_eval = adapter.transform(x_train), adapter.transform(x_eval)
    weights = recency_weights(frame.loc[chosen, "month"], spec["half_life_months"]) if "half_life_months" in spec else None
    kwargs = {"sample_weight": weights} if weights is not None else {}
    start = time.perf_counter()
    print("Fitting", result_path.stem, f"rows={len(chosen):,} features={x.shape[1]}", flush=True)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=config["threads"]):
        warnings.simplefilter("always")
        model.fit(x_train, labels.loc[chosen].to_numpy(), **kwargs)
        fit_seconds = time.perf_counter() - start
        scores = model.predict_proba(x_eval)[:, 1]
    if "negatives_per_positive" in spec:
        scores = correct_prior(scores, labels.loc[eligible].mean(), labels.loc[chosen].mean())
    y = labels.loc[evaluation].to_numpy()
    result = dict(candidate=spec["id"], phase=phase, eval_month=month, family=spec["family"], spec=spec,
                  train_rows=len(chosen), train_fraud=int(labels.loc[chosen].sum()), train_month_max=int(frame.loc[chosen, "month"].max()),
                  train_month_counts={str(k):int(v) for k,v in frame.loc[chosen,"month"].value_counts().sort_index().items()},
                  train_ids_sha256=hashlib.sha256(chosen.astype("<i8").tobytes()).hexdigest(),
                  evaluation_ids_sha256=hashlib.sha256(evaluation.astype("<i8").tobytes()).hexdigest(),
                  fit_seconds=fit_seconds, feature_count=x.shape[1], fingerprint=identity,
                  warnings=sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
                  **rank_metrics(y, scores, evaluation, config["review_fractions"]))
    save_prediction(pred_path, evaluation, y, scores)
    if phase == "exploratory":
        model_path = Path(config["output_dir"]) / "models" / f"{spec['id']}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(model=model, adapter=adapter, spec=spec), model_path)
    write_json(result_path, result)
    print(f"Done {result_path.stem}: AP={result['average_precision']:.4f}, recall@1%={result['recall_at_1pct']:.4f}, fit={fit_seconds:.1f}s", flush=True)
    return result


def records(config, phase):
    return [read_config(p) for p in sorted((Path(config["output_dir"]) / "runs").glob(f"{phase}-*.json"))]


def table_records(rows):
    return pd.DataFrame([{k:v for k,v in r.items() if k not in ["spec", "fingerprint", "warnings", "train_month_counts"]} for r in rows])


def blend_run(config, phase, month, spec, identity):
    result_path, pred_path = paths(config, phase, month, spec["id"])
    if result_path.exists() and pred_path.exists():
        result = read_config(result_path)
        if result["fingerprint"] != identity or result["spec"] != spec:
            raise ValueError("Stale blend cache")
        return result
    a, b = [prediction(config, phase, month, name) for name in spec["members"]]
    if not np.array_equal(a["row_ids"], b["row_ids"]) or not np.array_equal(a["labels"], b["labels"]):
        raise ValueError("Blend populations differ")
    scores = spec["weight"] * a["scores"] + (1 - spec["weight"]) * b["scores"]
    result = dict(candidate=spec["id"], family="blend", phase=phase, eval_month=month, spec=spec, fingerprint=identity,
                  evaluation_ids_sha256=hashlib.sha256(a["row_ids"].astype("<i8").tobytes()).hexdigest(),
                  fit_seconds=0., train_month_max=month-1, warnings=[],
                  **rank_metrics(a["labels"], scores, a["row_ids"], config["review_fractions"]))
    save_prediction(pred_path, a["row_ids"], a["labels"], scores)
    write_json(result_path, result)
    return result


def pooled(config, name, months):
    if not months:
        raise ValueError("Threshold fitting requires earlier validation months")
    items = [prediction(config, "validation", m, name) for m in months]
    return {key: np.concatenate([p[key] for p in items]) for key in items[0]}


def policies(config, data):
    return choose_policies(data["labels"], data["scores"], config["precision_target"], config["recall_target"])


def crossfit_policies(config, names):
    result = []
    for name in names:
        for month in config["validation_months"][1:]:
            earlier = [m for m in config["validation_months"] if m < month]
            thresholds = policies(config, pooled(config, name, earlier))
            data = prediction(config, "validation", month, name)
            for policy, threshold in thresholds.items():
                result.append(dict(candidate=name, eval_month=month, threshold_source_months=",".join(map(str,earlier)), policy=policy,
                                   **threshold_stats(data["labels"], data["scores"], threshold)))
    return pd.DataFrame(result)


def validate(config_path):
    config, identity = read_config(config_path), fingerprint(config_path)
    out = Path(config["output_dir"])
    if (out / "selection.json").exists():
        raise ValueError("Selection frozen; use a separate exploratory study")
    guard_preservation(config)
    frame = load_data(config)
    frame = frame[frame.month < config["test_month"]]
    base = features(frame)
    engineered = engineer(base)
    for month in config["validation_months"]:
        for spec in config["candidates"]:
            fit_run(frame, base, engineered, spec, month, "validation", config, identity)
    results = records(config, "validation")
    expected = {(s["id"],m) for s in config["candidates"] for m in config["validation_months"]}
    actual = {(r["candidate"],r["eval_month"]) for r in results if r["family"] != "blend"}
    if actual != expected:
        raise ValueError("Validation base runs incomplete")
    base_table = table_records([r for r in results if r["family"] != "blend"])
    means = base_table.groupby(["candidate","family"]).average_precision.mean().reset_index()
    members = []
    for family in ["lightgbm","catboost"]:
        members.append(means[means.family == family].sort_values(["average_precision","candidate"],ascending=[False,True]).iloc[0].candidate)
    blends = [dict(id=f"blend{int(w*100)}", family="blend", members=members, weight=w) for w in config["blend_lightgbm_weights"]]
    for month in config["validation_months"]:
        for spec in blends:
            blend_run(config, "validation", month, spec, identity)
    write_json(out / "blends.json", dict(fingerprint=identity, specs=blends))
    table = table_records(records(config,"validation"))
    table.to_csv(out / "validation_metrics.csv",index=False)
    names = sorted(table.candidate.unique())
    crossfit_policies(config, names).to_csv(out / "crossfit_policies.csv",index=False)
    summary = table.groupby("candidate",as_index=False)[["average_precision","roc_auc","brier","recall_at_1pct","precision_at_1pct","fit_seconds"]].mean()
    cf = pd.read_csv(out / "crossfit_policies.csv")
    for policy in ["f1","f2"]:
        scores = cf[cf.policy == policy].groupby("candidate")[policy].mean()
        summary[f"crossfit_{policy}"] = summary.candidate.map(scores)
    summary.sort_values("average_precision",ascending=False).to_csv(out / "validation_summary.csv",index=False)
    print(f"Validation complete: {len(expected)} base fits, {len(blends)*len(config['validation_months'])} blends",flush=True)


def freeze(config_path):
    config, identity = read_config(config_path), fingerprint(config_path)
    out = Path(config["output_dir"])
    guard_preservation(config)
    if (out / "selection.json").exists():
        raise ValueError("Refusing to overwrite selection")
    rows = records(config,"validation")
    blends = read_config(out/"blends.json")
    specs = {s["id"]:s for s in config["candidates"] + blends["specs"]}
    expected = {(name,m) for name in specs for m in config["validation_months"]}
    if {(r["candidate"],r["eval_month"]) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("Incomplete validation")
    if blends["fingerprint"] != identity or any(r["fingerprint"] != identity for r in rows):
        raise ValueError("Stale validation")
    if any(any("ConvergenceWarning" in w for w in r["warnings"]) for r in rows):
        raise ValueError("Unresolved convergence warnings")
    summary = pd.read_csv(out / "validation_summary.csv")
    selected = {}
    for objective in ["average_precision","recall_at_1pct","crossfit_f1","crossfit_f2"]:
        selected[objective] = summary.sort_values([objective,"average_precision","candidate"],ascending=[False,False,True]).iloc[0].candidate
    names = sorted(set(selected.values()) | {"baseline_lgbm","baseline_balanced","baseline_logistic"})
    needed = set(names)
    for name in names:
        if specs[name]["family"] == "blend":
            needed.update(specs[name]["members"])
    thresholds = {name:policies(config,pooled(config,name,config["validation_months"])) for name in names}
    selection = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),fingerprint=identity,data_sha256=config["data_sha256"],
                     selected=selected,evaluate_names=names,fit_specs=[specs[n] for n in sorted(needed) if specs[n]["family"]!="blend"],
                     blend_specs=[specs[n] for n in names if specs[n]["family"]=="blend"],thresholds=thresholds,
                     exploratory=True,month7_previously_consumed=True,config=config)
    write_json(out / "selection.json",selection)
    print("Frozen selections",json.dumps(selected),flush=True)


def load_selection(config_path):
    config = read_config(config_path)
    path = Path(config["output_dir"]) / "selection.json"
    if not path.exists():
        raise ValueError("Freeze validation selections before exploratory month-7 scoring")
    selected = read_config(path)
    if selected["fingerprint"] != fingerprint(config_path) or selected["data_sha256"] != config["data_sha256"]:
        raise ValueError("Stale frozen selection")
    guard_preservation(config)
    return config, selected


def exploratory(config_path):
    config, selected = load_selection(config_path)
    out = Path(config["output_dir"])
    selected.setdefault("exploratory_access_at_utc",datetime.now(timezone.utc).isoformat())
    write_json(out/"selection.json",selected)
    frame = load_data(config)
    base, month = features(frame),config["test_month"]
    engineered = engineer(base)
    for spec in selected["fit_specs"]:
        fit_run(frame,base,engineered,spec,month,"exploratory",config,selected["fingerprint"])
    for spec in selected["blend_specs"]:
        blend_run(config,"exploratory",month,spec,selected["fingerprint"])
    results = [r for r in records(config,"exploratory") if r["candidate"] in selected["evaluate_names"]]
    table_records(results).to_csv(out/"exploratory_metrics.csv",index=False)
    policy_rows = []
    predictions = {}
    for name in selected["evaluate_names"]:
        data = predictions[name] = prediction(config,"exploratory",month,name)
        for policy,threshold in selected["thresholds"][name].items():
            stats = threshold_stats(data["labels"],data["scores"],threshold)
            policy_rows.append(dict(candidate=name,policy=policy,precision_target_met=stats["precision"]>=config["precision_target"],
                                    recall_target_met=stats["recall"]>=config["recall_target"],**stats))
    pd.DataFrame(policy_rows).to_csv(out/"exploratory_policies.csv",index=False)
    intervals = paired_intervals(predictions,selected["thresholds"],"baseline_lgbm",config["bootstrap_replicates"],config["bootstrap_seed"])
    pd.DataFrame(intervals).to_csv(out/"paired_intervals.csv",index=False)
    guard_preservation(config)
    print("Exploratory evaluation complete",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command",choices=["validate","freeze","exploratory"])
    parser.add_argument("--config",default="configs/improvements.json")
    args = parser.parse_args()
    globals()[args.command](args.config)
