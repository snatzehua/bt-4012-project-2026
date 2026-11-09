"""Separate F2 search, chronological policy evaluation and immutable selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, features, load_data, read_config, write_json
from .engineering import engineer
from .improvements import fingerprint as old_fingerprint, fit_run, paths, prediction, save_prediction
from .recall_weighted_metrics import POLICIES, aligned, blend, check_data, choose_member, learn, measure, paired_f2

SOURCES = ["baf/common.py", "baf/engineering.py", "baf/improvement_metrics.py", "baf/improvements.py",
           "baf/recall_weighted_metrics.py", "baf/recall_weighted.py"]


def fingerprint(path):
    prior = old_fingerprint(path)
    prior["sources"] = {p: digest(p) for p in SOURCES}
    return prior


def guard(config):
    if config["beta"] != 2 or config["policies"] != POLICIES:
        raise ValueError("This study's declared objective/policies changed")
    for path, expected in config["preserve_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Preserved artifact changed: {path}")
    if old_fingerprint(config["prior_config"]) != config["prior_fingerprint"]:
        raise ValueError("Prior source/config/software identity changed")


def ids_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def verify_record(record, identity, spec, data, expected):
    if record["fingerprint"] != identity or record["spec"] != spec:
        raise ValueError("Stale run identity or spec")
    aligned(data, expected)
    if record["evaluation_ids_sha256"] != ids_digest(expected["row_ids"]):
        raise ValueError("Evaluation identity mismatch")
    if record["train_month_max"] >= record["eval_month"]:
        raise ValueError("Training crosses evaluation boundary")


def fit_or_reuse(frame, base, engineered, spec, month, phase, config, identity):
    record_path, pred_path = paths(config, phase, month, spec["id"])
    ids = frame.index[frame.month == month].to_numpy()
    expected = dict(row_ids=ids, labels=frame.loc[ids, "fraud_bool"].to_numpy(), scores=np.zeros(len(ids)))
    train_ids = frame.index[frame.month < month].to_numpy()
    def verify(record, data, expected_identity):
        verify_record(record, expected_identity, spec, data, expected)
        if record["train_ids_sha256"] != ids_digest(train_ids) or record["train_rows"] != len(train_ids):
            raise ValueError("Training population mismatch")
    if record_path.exists() and pred_path.exists():
        record = read_config(record_path)
        verify(record, prediction(config, phase, month, spec["id"]), identity)
        if record["prediction_sha256"] != digest(pred_path):
            raise ValueError("Cached prediction changed")
        return record
    prior = read_config(config["prior_config"])
    old_record, old_pred = paths(prior, phase, month, spec["id"])
    old_specs = {s["id"]: s for s in prior["candidates"]}
    if old_record.exists() and old_pred.exists() and old_specs.get(spec["id"]) == spec:
        record = read_config(old_record)
        data = prediction(prior, phase, month, spec["id"])
        verify(record, data, config["prior_fingerprint"])
        save_prediction(pred_path, data["row_ids"], data["labels"], data["scores"])
        record = {**record, "fingerprint": identity, "reused": True,
                  "reuse_record": str(old_record), "reuse_record_sha256": digest(old_record),
                  "reuse_prediction": str(old_pred), "reuse_prediction_sha256": digest(old_pred)}
        print("Reused", record_path.stem, flush=True)
    else:
        record = fit_run(frame, base, engineered, spec, month, phase, config, identity)
        record["reused"] = False
    record["prediction_sha256"] = digest(pred_path)
    write_json(record_path, record)
    verify(record, prediction(config, phase, month, spec["id"]), identity)
    return record


def histories(config):
    return {s["id"]: {m: prediction(config, "validation", m, s["id"])
                       for m in config["validation_months"]} for s in config["candidates"]}


def chosen_members(history, config, evaluation):
    earlier = {name: {m: data for m, data in items.items() if m < evaluation}
               for name, items in history.items()}
    return [choose_member(earlier, config["candidates"], family, evaluation)
            for family in ["lightgbm", "catboost"]]


def cf_predictions(history, config, month):
    members = chosen_members(history, config, month)
    earlier = [m for m in config["validation_months"] if m < month]
    items = {name: (data[month], {m: data[m] for m in earlier}, []) for name, data in history.items()}
    for weight in config["blend_lightgbm_weights"]:
        name = f"blend{int(weight * 100)}"
        data = {m: blend(history[members[0]][m], history[members[1]][m], weight) for m in earlier + [month]}
        items[name] = (data[month], {m: data[m] for m in earlier}, members)
    return items


def crossfit(config):
    history = histories(config)
    rows = []
    for month in config["validation_months"][1:]:
        reference = history["baseline_lgbm"][month]
        for name, (data, past, members) in cf_predictions(history, config, month).items():
            aligned(reference, data)
            for kind in config["policies"]:
                policy = learn(past, kind, month)
                rows.append(dict(candidate=name, policy=kind, eval_month=month,
                                 source_months=",".join(map(str, policy["source_months"])),
                                 members=",".join(members), parameters=json.dumps(policy, sort_keys=True),
                                 **measure(data, policy)))
    return pd.DataFrame(rows)


def validate(config_path):
    config, identity = read_config(config_path), fingerprint(config_path)
    guard(config)
    out = Path(config["output_dir"])
    if (out / "selection.json").exists():
        raise ValueError("Study is frozen; use a separate study for tuning")
    frame = load_data(config)
    frame = frame[frame.month < config["test_month"]]
    base = features(frame)
    engineered = engineer(base)
    for month in config["validation_months"]:
        for spec in config["candidates"]:
            fit_or_reuse(frame, base, engineered, spec, month, "validation", config, identity)
    table = crossfit(config)
    table.to_csv(out / "crossfit.csv", index=False)
    metrics = ["f2", "f1", "f3", "precision", "recall", "flagged_fraction", "average_precision", "tp", "fp", "fn"]
    summary = table.groupby(["candidate", "policy"], as_index=False)[metrics].mean()
    summary = summary.sort_values(["f2", "recall", "precision", "candidate", "policy"], ascending=[False, False, False, True, True])
    summary.to_csv(out / "summary.csv", index=False)
    previous = pd.read_csv(Path(read_config(config["prior_config"])["output_dir"]) / "validation_summary.csv").set_index("candidate")
    for name in ["baseline_lgbm", "baseline_balanced"]:
        value = summary[(summary.candidate == name) & (summary.policy == "pooled")].iloc[0].f2
        if not np.isclose(value, previous.loc[name, "crossfit_f2"], atol=1e-12):
            raise ValueError("Prior F2 baseline did not reproduce")
    write_json(out / "validation_complete.json", dict(fingerprint=identity, crossfit_sha256=digest(out / "crossfit.csv"),
               summary_sha256=digest(out / "summary.csv"), expected_runs=len(config["candidates"]) * len(config["validation_months"])))
    guard(config)
    print(summary.head(8).to_string(index=False), flush=True)


def audit_runs(config, identity, phase="validation"):
    specs = {s["id"]: s for s in config["candidates"]}
    records = [read_config(p) for p in sorted((Path(config["output_dir"]) / "runs").glob(f"{phase}-*.json"))]
    expected = {(s, m) for s in specs for m in config["validation_months"]} if phase == "validation" else None
    if expected is not None and {(r["candidate"], r["eval_month"]) for r in records} != expected:
        raise ValueError("Missing or extra validation results")
    for record in records:
        name, month = record["candidate"], record["eval_month"]
        data = prediction(config, phase, month, name)
        verify_record(record, identity, specs[name], data, data)
        rp, pp = paths(config, phase, month, name)
        if record["prediction_sha256"] != digest(pp):
            raise ValueError("Saved prediction changed")
        if record.get("reused"):
            for field in ["reuse_record", "reuse_prediction"]:
                if digest(record[field]) != record[field + "_sha256"]:
                    raise ValueError("Reuse source changed")
        if record["warnings"]:
            raise ValueError("Inspect model warnings before selection")
    return records


def freeze(config_path):
    config, identity = read_config(config_path), fingerprint(config_path)
    guard(config)
    out = Path(config["output_dir"])
    path = out / "selection.json"
    if path.exists():
        raise ValueError("Refusing to overwrite selection")
    audit_runs(config, identity)
    complete = read_config(out / "validation_complete.json")
    if complete["fingerprint"] != identity or complete["crossfit_sha256"] != digest(out / "crossfit.csv") or complete["summary_sha256"] != digest(out / "summary.csv"):
        raise ValueError("Stale development summary")
    cf = crossfit(config)
    stored = pd.read_csv(out / "crossfit.csv")
    if not np.allclose(cf.f2, stored.f2):
        raise ValueError("Crossfit metrics mismatch")
    summary = pd.read_csv(out / "summary.csv")
    selected = summary.sort_values(["f2", "recall", "precision", "candidate", "policy"], ascending=[False,False,False,True,True]).iloc[0]
    history = histories(config)
    members = chosen_members(history, config, config["test_month"])
    eval_names = sorted({selected.candidate, "baseline_lgbm", "baseline_balanced"})
    final_history = {}
    blends = {}
    for name in eval_names:
        if name.startswith("blend"):
            weight = int(name[5:]) / 100
            blends[name] = dict(members=members, weight=weight)
            final_history[name] = {m: blend(history[members[0]][m], history[members[1]][m], weight) for m in config["validation_months"]}
        else:
            final_history[name] = history[name]
    rules = {name: {kind: learn(final_history[name], kind, config["test_month"]) for kind in POLICIES} for name in eval_names}
    needed = set(eval_names) - set(blends)
    if blends:
        needed.update(members)
    selection = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), fingerprint=identity, config=config,
                     selected_candidate=str(selected.candidate), selected_policy=str(selected.policy), selected_development_f2=float(selected.f2),
                     evaluate_names=eval_names, blends=blends, rules=rules,
                     fit_specs=[s for s in config["candidates"] if s["id"] in needed],
                     exploratory=True, month7_previously_consumed=True)
    write_json(path, selection)
    print("Frozen", selection["selected_candidate"], selection["selected_policy"], selection["selected_development_f2"], flush=True)


def load_selection(config_path):
    config = read_config(config_path)
    guard(config)
    selected = read_config(Path(config["output_dir"]) / "selection.json")
    if selected["fingerprint"] != fingerprint(config_path) or selected["config"] != config:
        raise ValueError("Frozen study identity changed")
    return config, selected


def exploratory(config_path):
    config, selected = load_selection(config_path)
    out = Path(config["output_dir"])
    access = out / "exploratory_access.json"
    if not access.exists():
        write_json(access, dict(access_at_utc=datetime.now(timezone.utc).isoformat(), selection_sha256=digest(out / "selection.json"),
                               month7_previously_consumed=True))
    elif read_config(access)["selection_sha256"] != digest(out / "selection.json"):
        raise ValueError("Selection changed after access")
    frame = load_data(config)
    base = features(frame)
    engineered = engineer(base)
    month = config["test_month"]
    for spec in selected["fit_specs"]:
        fit_or_reuse(frame, base, engineered, spec, month, "exploratory", config, selected["fingerprint"])
    data = {s["id"]: prediction(config, "exploratory", month, s["id"]) for s in selected["fit_specs"]}
    for name, spec in selected["blends"].items():
        a, b = [data[n] for n in spec["members"]]
        data[name] = blend(a, b, spec["weight"])
        save_prediction(paths(config, "exploratory", month, name)[1], a["row_ids"], a["labels"], data[name]["scores"])
    rows = []
    for name in selected["evaluate_names"]:
        aligned(data["baseline_lgbm"], data[name])
        for kind, policy in selected["rules"][name].items():
            rows.append(dict(candidate=name, policy=kind, selected=(name==selected["selected_candidate"] and kind==selected["selected_policy"]),
                             parameters=json.dumps(policy, sort_keys=True), **measure(data[name], policy)))
    pd.DataFrame(rows).to_csv(out / "exploratory.csv", index=False)
    reference = "baseline_balanced__pooled"
    paired_data, paired_rules = {}, {}
    for row in rows:
        name, kind = row["candidate"], row["policy"]
        key = name + "__" + kind
        paired_data[key], paired_rules[key] = data[name], selected["rules"][name][kind]
    intervals = paired_f2(paired_data, paired_rules, reference, config["bootstrap_replicates"], config["bootstrap_seed"])
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    guard(config)
    print(pd.DataFrame(rows)[["candidate","policy","f2","precision","recall","flagged_fraction","tp","fp","fn"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "freeze", "exploratory"])
    parser.add_argument("--config", default="configs/recall-weighted.json")
    args = parser.parse_args()
    globals()[args.command](args.config)
