"""Execute validation, freeze choices, then perform the final evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .common import (
    best_f1_threshold, bootstrap_recall, budget_name, digest, evaluate, features,
    load_data, make_model, matched_indices, ranking, read_config, recent_indices, write_json,
)


def identity(config_path):
    return {"config_sha256": digest(config_path),
            "source_sha256": hashlib.sha256(b"".join(Path(p).read_bytes() for p in ["baf/common.py", "baf/study.py"])).hexdigest()}


def run_name(phase, month, spec):
    return f"{phase}-m{month}-{spec['model']}-{spec['strategy']}-k{spec['k']}-s{spec['seed']}-w{int(spec['weighted'])}"


def specs_for_validation(config):
    for model in config["models"]:
        yield dict(model=model, strategy="expanding", k=0, seed=config["sampling_seeds"][0], weighted=False)
        for k in config["sample_sizes"]:
            yield dict(model=model, strategy="recent", k=k, seed=config["sampling_seeds"][0], weighted=False)
            for seed in config["sampling_seeds"]:
                yield dict(model=model, strategy="historical_matched", k=k, seed=seed, weighted=False)
        yield dict(model=model, strategy="expanding", k=0, seed=config["sampling_seeds"][0], weighted=True)
    for weighted in [False, True]:
        yield dict(model="random_forest", strategy="expanding", k=0, seed=config["sampling_seeds"][0], weighted=weighted)


def train_indices(frame, eligible, spec, config):
    if spec["strategy"] in ["expanding", "random_diagnostic"]:
        return np.sort(eligible)
    recent = recent_indices(frame.month.to_numpy(), eligible, spec["k"], config["tie_seed"])
    if spec["strategy"] == "recent":
        return recent
    return matched_indices(frame.fraud_bool.to_numpy(), eligible, recent, spec["seed"])


def fit_run(frame, x, eligible, evaluation, spec, month, phase, config, fingerprint, threshold=None):
    out = Path(config["output_dir"])
    name = run_name(phase, month, spec)
    result_path = out / "runs" / f"{name}.json"
    prediction_path = out / "predictions" / f"{name}.npz"
    if result_path.exists() and prediction_path.exists():
        result = json.loads(result_path.read_text())
        if result["fingerprint"] != fingerprint:
            raise ValueError(f"Stale run cache: {name}; use a new output directory")
        print("Cached", name, flush=True)
        return result
    chosen = train_indices(frame, eligible, spec, config)
    if np.intersect1d(chosen, evaluation).size:
        raise AssertionError("Training/evaluation overlap")
    if phase != "diagnostic" and frame.loc[chosen, "month"].max() >= month:
        raise AssertionError("Future training information")
    y_train = frame.loc[chosen, "fraud_bool"].to_numpy()
    y_eval = frame.loc[evaluation, "fraud_bool"].to_numpy()
    estimator = make_model(spec["model"], spec["weighted"], config, x.columns)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=config["threads"]):
        warnings.simplefilter("always")
        estimator.fit(x.loc[chosen], y_train)
        fit_seconds = time.perf_counter() - start
        scores = estimator.predict_proba(x.loc[evaluation])[:, 1]
    result = {"run": name, "phase": phase, "eval_month": month, **spec,
              "train_rows": len(chosen), "train_fraud": int(y_train.sum()),
              "train_months": {str(k): int(v) for k, v in frame.loc[chosen, "month"].value_counts().sort_index().items()},
              "train_row_ids_sha256": hashlib.sha256(chosen.astype("<i8").tobytes()).hexdigest(),
              "evaluation_ids_sha256": hashlib.sha256(np.asarray(evaluation, dtype="<i8").tobytes()).hexdigest(),
              "fit_seconds": fit_seconds, "fingerprint": fingerprint,
              "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
              **evaluate(y_eval, scores, evaluation, config["review_fractions"], threshold)}
    if phase == "validation":
        result["f1_threshold_from_validation"] = best_f1_threshold(y_eval, scores)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, row_ids=evaluation, labels=y_eval, scores=scores)
    if phase == "final" and spec["strategy"] == "expanding" and not spec["weighted"]:
        model = estimator.named_steps["model"]
        names = estimator.named_steps["preprocess"].get_feature_names_out()
        values = model.coef_[0] if spec["model"] == "logistic" else model.feature_importances_
        pd.DataFrame({"feature": names, "coefficient" if spec["model"] == "logistic" else "importance": values}).to_csv(out / f"{spec['model']}_features.csv", index=False)
    write_json(result_path, result)
    print(f"Done {name}: recall@1%={result['recall_at_1pct']:.4f}, AP={result['average_precision']:.4f}, fit={fit_seconds:.1f}s", flush=True)
    return result


def collect(config, phase):
    results = [json.loads(p.read_text()) for p in sorted((Path(config["output_dir"]) / "runs").glob(f"{phase}-*.json"))]
    if not results:
        raise ValueError(f"No completed {phase} runs")
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ["train_months", "fingerprint", "warnings"]} for r in results])
    table.to_csv(Path(config["output_dir"]) / f"{phase}_metrics.csv", index=False)
    return results, table


def validate(config_path):
    config = read_config(config_path)
    if (Path(config["output_dir"]) / "selection.json").exists():
        raise ValueError("Selection already frozen; use a new study directory for revised development")
    frame = load_data(config)
    x = features(frame)
    fingerprint = identity(config_path)
    for month in config["validation_months"]:
        eligible = np.flatnonzero(frame.month.to_numpy() < month)
        evaluation = np.flatnonzero(frame.month.to_numpy() == month)
        for spec in specs_for_validation(config):
            fit_run(frame, x, eligible, evaluation, spec, month, "validation", config, fingerprint)
    _, table = collect(config, "validation")
    expected = len(config["validation_months"]) * len(list(specs_for_validation(config)))
    if len(table) != expected:
        raise AssertionError(f"Expected {expected} validation runs, got {len(table)}")
    print(f"Validation complete: {len(table)} fits", flush=True)


def freeze(config_path):
    config = read_config(config_path)
    out = Path(config["output_dir"])
    if (out / "selection.json").exists():
        raise ValueError("Refusing to overwrite frozen choices")
    results, table = collect(config, "validation")
    expected = len(config["validation_months"]) * len(list(specs_for_validation(config)))
    if len(table) != expected or not set(table.eval_month) == set(config["validation_months"]):
        raise ValueError("Validation is incomplete")
    fingerprint = identity(config_path)
    if any(r["fingerprint"] != fingerprint for r in results):
        raise ValueError("Validation fingerprints do not match current source/config")
    if any(any("ConvergenceWarning" in w for w in r["warnings"]) for r in results):
        raise ValueError("Unresolved convergence warnings; inspect development runs")
    selected_k = {}
    final_specs = []
    for model in config["models"]:
        candidates = table[(table.model == model) & (table.strategy == "recent")]
        means = candidates.groupby("k", as_index=False).recall_at_1pct.mean().sort_values(["recall_at_1pct", "k"], ascending=[False, True])
        selected_k[model] = int(means.iloc[0].k)
        final_specs.append(dict(model=model, strategy="recent", k=selected_k[model], seed=config["sampling_seeds"][0], weighted=False))
        for seed in config["sampling_seeds"]:
            final_specs.append(dict(model=model, strategy="historical_matched", k=selected_k[model], seed=seed, weighted=False))
    for model in [*config["models"], "random_forest"]:
        for weighted in [False, True]:
            final_specs.append(dict(model=model, strategy="expanding", k=0, seed=config["sampling_seeds"][0], weighted=weighted))
    for spec in final_specs:
        subset = table.copy()
        for key, value in spec.items():
            subset = subset[subset[key] == value]
        if len(subset) != len(config["validation_months"]):
            raise ValueError(f"Incomplete candidate: {spec}")
        spec["threshold"] = float(subset.f1_threshold_from_validation.median())
    # Seeds are averaged within month before months; choose only predeclared final comparators.
    group_keys = ["model", "strategy", "k", "weighted"]
    monthly = table.groupby([*group_keys, "eval_month"], as_index=False)[["recall_at_1pct", "average_precision"]].mean()
    means = monthly.groupby(group_keys, as_index=False)[["recall_at_1pct", "average_precision"]].mean()
    allowed = means.apply(lambda r: r.strategy == "expanding" or int(r.k) == selected_k.get(r.model), axis=1)
    means = means[allowed].sort_values(["recall_at_1pct", "average_precision", *group_keys], ascending=[False, False, True, True, True, True])
    winner = means.iloc[0]
    selection = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint,
                 "data_sha256": config["data_sha256"], "config": config, "selected_k": selected_k,
                 "validation_winner": {key: winner[key].item() if hasattr(winner[key], "item") else winner[key] for key in group_keys},
                 "final_specs": final_specs, "test_consumed": False}
    means.to_csv(out / "validation_selection_summary.csv", index=False)
    write_json(out / "selection.json", selection)
    print("Choices frozen:", json.dumps(selection["selected_k"]), "winner:", selection["validation_winner"], flush=True)


def final(config_path):
    config = read_config(config_path)
    out = Path(config["output_dir"])
    selection_path = out / "selection.json"
    if not selection_path.exists():
        raise ValueError("Freeze validation choices before final evaluation")
    selection = json.loads(selection_path.read_text())
    fingerprint = identity(config_path)
    if selection["fingerprint"] != fingerprint or selection["data_sha256"] != config["data_sha256"]:
        raise ValueError("Frozen selection is stale")
    frame = load_data(config)
    x = features(frame)
    month = config["test_month"]
    eligible = np.flatnonzero(frame.month.to_numpy() < month)
    evaluation = np.flatnonzero(frame.month.to_numpy() == month)
    selection["test_consumed"] = True
    selection.setdefault("first_final_access_at_utc", datetime.now(timezone.utc).isoformat())
    write_json(selection_path, selection)
    for saved_spec in selection["final_specs"]:
        spec = saved_spec.copy()
        threshold = spec.pop("threshold")
        fit_run(frame, x, eligible, evaluation, spec, month, "final", config, fingerprint, threshold)
    collect(config, "final")
    paired_bootstrap(config)


def paired_bootstrap(config):
    results, _ = collect(config, "final")
    out = Path(config["output_dir"])
    differences = []
    for model in config["models"]:
        recent = next(r for r in results if r["model"] == model and r["strategy"] == "recent")
        history = [r for r in results if r["model"] == model and r["strategy"] == "historical_matched"]
        expanding = next(r for r in results if r["model"] == model and r["strategy"] == "expanding" and not r["weighted"])
        runs = [recent, *history, expanding]
        data = [np.load(out / "predictions" / f"{r['run']}.npz") for r in runs]
        labels = data[0]["labels"]
        ids = data[0]["row_ids"]
        if any(not np.array_equal(ids, d["row_ids"]) or not np.array_equal(labels, d["labels"]) for d in data):
            raise AssertionError("Paired comparisons must have identical cases")
        orders = [ranking(d["scores"], ids) for d in data]
        rng = np.random.default_rng(config["bootstrap_seed"])
        samples = []
        for _ in range(config["bootstrap_replicates"]):
            counts = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))
            recalls = [bootstrap_recall(labels, order, counts, config["primary_fraction"]) for order in orders]
            samples.append([recalls[0] - np.mean(recalls[1:-1]), recalls[0] - recalls[-1]])
        samples = np.asarray(samples)
        for j, comparison in enumerate(["recent_minus_mean_matched", "recent_minus_expanding"]):
            point = recent["recall_at_1pct"] - (np.mean([r["recall_at_1pct"] for r in history]) if j == 0 else expanding["recall_at_1pct"])
            low, high = np.quantile(samples[:, j], [0.025, 0.975])
            differences.append({"model": model, "k": recent["k"], "comparison": comparison,
                                "difference": point, "ci_low": low, "ci_high": high,
                                "replicates": config["bootstrap_replicates"], "uncertainty": "conditional paired test-row bootstrap; no multiplicity adjustment"})
        for d in data:
            d.close()
    pd.DataFrame(differences).to_csv(out / "paired_differences.csv", index=False)
    print(pd.DataFrame(differences).to_string(index=False), flush=True)


def diagnostic(config_path):
    config = read_config(config_path)
    frame = load_data(config)
    x = features(frame)
    dev = np.flatnonzero(frame.month.to_numpy() < config["test_month"])
    eligible, evaluation = train_test_split(dev, test_size=0.2, stratify=frame.loc[dev, "fraud_bool"], random_state=config["model_seed"])
    evaluation = np.sort(evaluation)
    for model in [*config["models"], "random_forest"]:
        spec = dict(model=model, strategy="random_diagnostic", k=0, seed=config["sampling_seeds"][0], weighted=False)
        fit_run(frame, x, eligible, evaluation, spec, -1, "diagnostic", config, identity(config_path))
    collect(config, "diagnostic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["validate", "freeze", "final", "diagnostic"])
    parser.add_argument("--config", default="configs/study.json")
    args = parser.parse_args()
    {"validate": validate, "freeze": freeze, "final": final, "diagnostic": diagnostic}[args.phase](args.config)
