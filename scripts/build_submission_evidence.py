#!/usr/bin/env python3
"""Export checked historical evidence; never train, select, or access raw data.

Only explicitly named completed-study tables enter this submission view. Hashes
pin the evidence inspected when the submission workflow was assembled. A mismatch
is an error, not an invitation to update a frozen study or silently accept new data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {'reports/baf-study-v1/monthly.csv': 'd147364db42f87a5dfa3c4c813b84fcaa2c198de12d29d9ef6839f2db45da47a',
 'reports/baf-study-v1/validation_recency.csv': '97ac11b3590775eb84b059f700c6e2a66eb2a1af60ef2faf02f0d30e79050758',
 'reports/baf-study-v1/paired_differences.csv': '7996756295d11810dcc304e4af89c60bfd570f56f4aaeac02b63948bc72aa42f',
 'reports/baf-study-v1/final_summary.csv': '9d520524c861c8e4344fba6918f4d34012bc17e6031b0b53f27986505b03ec94',
 'reports/baf-study-v1/final_metrics.csv': '5bc1b5bf6ac23441503644d2ab44111746bb5359d0475b10479142fe80b7b49e',
 'reports/baf-improvements-v1/validation_summary.csv': 'b262d4d771b808d3de5f72402dd2eb4752b152f051b0ce52b8950dc7b642e115',
 'reports/baf-improvements-v1/logistic_fixed_c_summary.csv': '2fd354bc10a22776bea6dd1489450e2ecba5058b4d6aa7e93ede6ca55f7f70f6',
 'reports/baf-improvements-v1/exploratory_metrics.csv': '2e7e457b89f29bfb2d535e6e4c4f14c72f89bd0576283fc0324a0025bcd6b64c',
 'reports/baf-bagged-anchor-fusion-v1/summary.csv': '10f6e8afada75830f94e8dc14142861436e15778cc5ee3e3e90eb9dcb09e7084',
 'reports/baf-bagged-anchor-fusion-v1/exploratory.csv': 'fc210d14210a25e7130b175ec5aa9531022cc76ae8a68fc7bc1f0562d993ed2b',
 'reports/baf-bagged-anchor-fusion-v1/development_intervals.csv': '4aa92293c1bc302b1db985c2b12faa509e80a016db19b95b44e2a24454ac8cff',
 'reports/baf-bagged-anchor-fusion-v1/consumed_intervals.csv': '716632c0716798395bee5be01000af4bf07806a522950428c79405a6b51cf0a6',
 'reports/baf-study-v1/selection.json': 'cf88f7795898feb6be74d43e1591288939d93f99eb13a3eba841008a65e3f986',
 'reports/baf-improvements-v1/selection.json': 'bd9a851c06917fcfeb7547bea86b98997a794ec0601ad11d315673d0475b02c3',
 'reports/baf-bagged-anchor-fusion-v1/selection.json': 'ec308166981ad973ab0fa6bee693e21d7bc8fe0e351bed32f4cb8b02eb8b8d0c',
 'configs/study.json': 'a4a1bf6c4d1d724a478c7a16809f69e93a3f32905768c3d2897359cbec6dbdac',
 'configs/improvements.json': '69ce021f440b840327127b6faada39fe89df7b0f4a870c8eaf7c0739c9a6e13d',
 'configs/bagged-anchor-fusion.json': '50e749a9f46bf52e0ec4b2bc5703edfe3a44e4ba4d78449e84ad23bb80e57f85'}


def read_csv(root: Path, name: str) -> list[dict[str, str]]:
    with (root / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_close(actual: float, expected: float, name: str) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{name}: {actual} != {expected}")


def validate_confusion(row: dict, *, suffix: str = "", prefix: str = "",
                       n_key: str = "n_eval", fraud_key: str = "fraud_eval") -> None:
    get = lambda key: float(row[f"{prefix}{key}{suffix}"])
    tp, fp, fn, tn = (get(k) for k in ("tp", "fp", "fn", "tn"))
    if any(not math.isfinite(v) or v < 0 or not v.is_integer() for v in (tp, fp, fn, tn)):
        raise ValueError("Confusion counts must be finite nonnegative integers")
    check_close(tp + fp + fn + tn, float(row[n_key]), "population")
    check_close(tp + fn, float(row[fraud_key]), "fraud population")
    check_close(get("precision"), tp / (tp + fp) if tp + fp else 0, "precision")
    check_close(get("recall"), tp / (tp + fn) if tp + fn else 0, "recall")
    f2_key = f"{prefix}f2{suffix}"
    if f2_key in row:
        check_close(float(row[f2_key]), 5 * tp / (5 * tp + 4 * fn + fp) if tp else 0, "F2")


def verify_sources(root: Path) -> dict[str, str]:
    found = {}
    for name, expected in SOURCE_SHA256.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"Missing required evidence: {name}")
        found[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if found[name] != expected:
            raise ValueError(f"Frozen evidence hash mismatch: {name}")
    return found


def make_csv(rows: list[dict]) -> str:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def collect(root: Path = ROOT) -> tuple[dict, dict[str, str]]:
    sources = verify_sources(root)
    study = "reports/baf-study-v1/"
    imp = "reports/baf-improvements-v1/"
    anchor = "reports/baf-bagged-anchor-fusion-v1/"
    final = read_csv(root, study + "final_metrics.csv")
    for row in final:
        if int(row["eval_month"]) != 7:
            raise ValueError("Final study table contains a non-month-7 row")
        for capacity in ("0.5pct", "1pct", "2pct", "5pct"):
            validate_confusion(row, suffix="_at_" + capacity)
            check_close(float(row["reviewed_at_" + capacity]),
                        float(row["tp_at_" + capacity]) + float(row["fp_at_" + capacity]), "review count")
            quota = math.ceil(float(capacity.removesuffix("pct")) / 100 * int(row["n_eval"]))
            check_close(float(row["reviewed_at_" + capacity]), quota, "exact capacity quota")
        validate_confusion(row, prefix="default_")
        validate_confusion(row, prefix="selected_")
    evaluation_ids = {r["evaluation_ids_sha256"] for r in final}
    if len(evaluation_ids) != 1:
        raise ValueError("Original comparators do not evaluate identical applications")
    imetrics = read_csv(root, imp + "exploratory_metrics.csv")
    if {r["evaluation_ids_sha256"] for r in imetrics} != evaluation_ids:
        raise ValueError("Improvement and original evaluation applications differ")
    for row in imetrics:
        if int(row["eval_month"]) != 7 or int(row["train_month_max"]) >= 7:
            raise ValueError("Improvement chronological boundary is invalid")
        for capacity in ("0.5pct", "1pct", "2pct", "5pct"):
            validate_confusion(row, suffix="_at_" + capacity)
    consumed = read_csv(root, anchor + "exploratory.csv")
    roles = ("unweighted", "balanced", "fusion", "latest_fusion", "selected")
    consumed = [r for r in consumed if r["role"] in roles]
    if len(consumed) != len(roles) or set(r["role"] for r in consumed) != set(roles):
        raise ValueError("Missing or duplicate frozen transfer role")
    dev = read_csv(root, anchor + "summary.csv")
    f2 = []
    for role in roles:
        row = next(r for r in consumed if r["role"] == role)
        validate_confusion(row, n_key="n", fraud_key="fraud")
        matches = [r for r in dev if r["candidate"] == row["candidate"] and r["policy"] == row["policy"]]
        if len(matches) != 1:
            raise ValueError("Ambiguous development candidate/policy")
        f2.append({"role": role, "candidate": row["candidate"], "policy": row["policy"],
                   "development_macro_f2": matches[0]["f2"], "consumed_month7_f2": row["f2"],
                   "precision": row["precision"], "recall": row["recall"],
                   "flagged_fraction": row["flagged_fraction"],
                   **{k: row[k] for k in ("tp", "fp", "fn", "tn")}})
    selection = json.loads((root / anchor / "selection.json").read_text())
    selected = next(r for r in f2 if r["role"] == "selected")
    if selection["selected_candidate"] != selected["candidate"] or selection["selected_policy"] != selected["policy"]:
        raise ValueError("Frozen selection does not match consumed selected role")
    check_close(float(selection["selected_development_f2"]), float(selected["development_macro_f2"]), "selected development F2")
    tables = {
        "monthly.csv": read_csv(root, study + "monthly.csv"),
        "recency_development.csv": read_csv(root, study + "validation_recency.csv"),
        "recency_intervals.csv": read_csv(root, study + "paired_differences.csv"),
        "original_final_summary.csv": read_csv(root, study + "final_summary.csv"),
        "logistic_fixed_c.csv": read_csv(root, imp + "logistic_fixed_c_summary.csv"),
        "improvement_development.csv": read_csv(root, imp + "validation_summary.csv"),
        "improvement_month7.csv": [{k: r[k] for k in ("candidate", "average_precision", "precision_at_1pct", "recall_at_1pct", "tp_at_1pct", "fp_at_1pct", "fn_at_1pct", "tn_at_1pct", "reviewed_at_1pct")} for r in imetrics],
        "f2_comparison.csv": f2,
        "f2_development_intervals.csv": read_csv(root, anchor + "development_intervals.csv"),
        "f2_consumed_intervals.csv": read_csv(root, anchor + "consumed_intervals.csv"),
    }
    files = {name: make_csv(rows) for name, rows in tables.items()}
    manifest = {
        "schema_version": 1,
        "scope": "Read-only historical evidence export; no model fits or new statistical inference",
        "evaluation_warning": "Month 7 is already consumed. Later studies and this consolidation are exploratory; no fresh confirmation is claimed.",
        "aggregation_warning": "Development macro F2 averages chronological month-specific F2. Do not recompute it from averaged confusion counts.",
        "selection_warning": "Unweighted month-7 superiority is descriptive, not a replacement for the development-selected strategy.",
        "excluded_incomplete_studies": ["test-context-feature-ablation-f2", "test-spline-logistic-f2 (stopped at convergence gate)"],
        "source_sha256": sources,
        "export_sha256": {name: hashlib.sha256(data.encode()).hexdigest() for name, data in sorted(files.items())},
        "validated": {"original_final_rows": len(final), "improvement_final_rows": len(imetrics), "f2_consumed_roles": len(f2), "model_fits": 0},
    }
    return manifest, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission" / "generated")
    parser.add_argument("--check", action="store_true", help="Verify sources/metrics without writing output")
    args = parser.parse_args()
    manifest, files = collect()
    if not args.check:
        output = args.output.resolve()
        # An export may never overwrite the frozen source tree.
        if output == ROOT or any(output == p or p in output.parents for p in (ROOT / "reports", ROOT / "baf", ROOT / "configs", ROOT / "openspec")):
            raise ValueError("Output must not overwrite scientific sources or frozen evidence")
        output.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (output / name).write_text(data, encoding="utf-8", newline="")
        (output / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["validated"], sort_keys=True))


if __name__ == "__main__":
    main()
