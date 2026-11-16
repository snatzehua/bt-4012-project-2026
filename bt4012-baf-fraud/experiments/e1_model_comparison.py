"""E1 - Model comparison under the out-of-time protocol.

Train on months 0-4, tune and set thresholds on month 5, test on months 6-7.
All models use class weighting here (E4 compares imbalance strategies).

Outputs
  results/e1_tuning.csv, results/e1_best_params.json
  results/e1_model_comparison.csv   one row per model x seed
  results/e1_summary.csv            mean +- std over seeds, with bootstrap CIs
  results/e1_per_month.csv          test months 6 and 7 separately
  results/e1_scores.csv.gz          validation and test scores (seed 0) used by E5/E6
  results/models/*.joblib           fitted models (seed 0) used by E6
  figures/e1_*.png
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import AGE_COL, MONTH, TARGET
from src.data import months
from src.metrics import (at_threshold, bootstrap_ci, capacity_metrics, ranking_metrics,
                         recall_at_fpr, threshold_at_fpr)
from src.models import MODEL_NAMES, PARAM_GRID, fit_predict
from src.utils import log, plt, save_json, savefig

STRATEGY = "class_weight"


def run(df, nums, cats, S):
    tr, va, te = months(df, S.train_months), months(df, S.val_months), months(df, S.test_months)
    y_va, y_te = va[TARGET].to_numpy(), te[TARGET].to_numpy()
    log(f"E1: train {len(tr):,} rows (months {S.train_months}), val {len(va):,}, test {len(te):,}")

    # ---- 1. tune a small grid per model on the validation month ----------
    tuning, best = [], {}
    for name in MODEL_NAMES:
        for params in PARAM_GRID[name]:
            _, (s_va,), secs = fit_predict(name, tr, [va], nums, cats, S, STRATEGY, seed=0, params=params)
            tuning.append({"model": name, "params": params, "val_pr_auc": average_precision_score(y_va, s_va),
                           "val_recall_at_5fpr": recall_at_fpr(y_va, s_va, S.fpr_target), "fit_seconds": secs})
            log(f"  tune {name} {params}: val PR-AUC {tuning[-1]['val_pr_auc']:.4f}")
        cand = [t for t in tuning if t["model"] == name]
        best[name] = max(cand, key=lambda t: t["val_pr_auc"])["params"]
    pd.DataFrame(tuning).to_csv(S.results_dir / "e1_tuning.csv", index=False)
    best_model = max((t for t in tuning if t["params"] == best[t["model"]]), key=lambda t: t["val_pr_auc"])["model"]
    save_json({"params": best, "best_model": best_model, "strategy": STRATEGY}, S.results_dir / "e1_best_params.json")
    log(f"E1: best params {best}; best model on validation: {best_model}")

    # ---- 2. evaluate each tuned model on the untouched test months --------
    rows, per_month, score_frames = [], [], []
    for name in MODEL_NAMES:
        seeds = [0] if name == "logreg" else S.seeds  # logistic regression is deterministic
        for seed in seeds:
            model, (s_va, s_te), secs = fit_predict(name, tr, [va, te], nums, cats, S, STRATEGY,
                                                    seed=seed, params=best[name])
            thr = threshold_at_fpr(y_va, s_va, S.fpr_target)  # chosen on validation only
            at = at_threshold(y_te, s_te, thr)
            row = {"model": name, "seed": seed, "fit_seconds": secs,
                   **ranking_metrics(y_te, s_te, S.fpr_target),
                   **{f"valthr_{k}": at[k] for k in ["precision", "recall", "f1", "fpr", "alert_rate"]},
                   **capacity_metrics(y_te, s_te, S.review_capacities)}
            if seed == 0:
                row["pr_auc_ci"] = bootstrap_ci(y_te, s_te, average_precision_score, S.n_bootstrap)
                row["recall_at_5fpr_ci"] = bootstrap_ci(
                    y_te, s_te, lambda a, b: recall_at_fpr(a, b, S.fpr_target), S.n_bootstrap)
                if name != "random_forest":  # forests on ~600k rows are very large on disk; E6 doesn't need them
                    joblib.dump(model, S.results_dir / "models" / f"{name}.joblib", compress=3)
                for split, d, s in [("val", va, s_va), ("test", te, s_te)]:
                    score_frames.append(pd.DataFrame({"model": name, "split": split, MONTH: d[MONTH].to_numpy(),
                                                      TARGET: d[TARGET].to_numpy(), AGE_COL: d[AGE_COL].to_numpy(),
                                                      "score": s}))
            rows.append(row)
            for m in S.test_months:
                mask = te[MONTH].to_numpy() == m
                pm = at_threshold(y_te[mask], s_te[mask], thr)
                per_month.append({"model": name, "seed": seed, "month": m,
                                  **ranking_metrics(y_te[mask], s_te[mask], S.fpr_target),
                                  "valthr_recall": pm["recall"], "valthr_fpr": pm["fpr"]})
            log(f"  {name} seed {seed}: test PR-AUC {row['pr_auc']:.4f}, "
                f"recall@{S.fpr_target:.0%}FPR {row['recall_at_5fpr']:.3f}, realised FPR {at['fpr']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(S.results_dir / "e1_model_comparison.csv", index=False)
    pd.DataFrame(per_month).to_csv(S.results_dir / "e1_per_month.csv", index=False)
    pd.concat(score_frames).to_csv(S.results_dir / "e1_scores.csv.gz", index=False)

    metrics = ["pr_auc", "roc_auc", "recall_at_5fpr", "valthr_recall", "valthr_precision", "valthr_fpr",
               "top1pct_precision", "top5pct_recall", "fit_seconds"]
    summ = res.groupby("model")[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    ci = res[res.seed == 0].set_index("model")
    for c in ["pr_auc_ci", "recall_at_5fpr_ci"]:
        summ[c.replace("_ci", "_ci_low")] = ci[c].map(lambda t: t[0])
        summ[c.replace("_ci", "_ci_high")] = ci[c].map(lambda t: t[1])
    summ.to_csv(S.results_dir / "e1_summary.csv")

    # ---- figures ----------------------------------------------------------
    scores = pd.concat(score_frames)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in MODEL_NAMES:
        d = scores[(scores.model == name) & (scores.split == "test")]
        p, r, _ = precision_recall_curve(d[TARGET], d["score"])
        ax.plot(r, p, label=f"{name} (AP {average_precision_score(d[TARGET], d['score']):.3f})")
    ax.axhline(y_te.mean(), ls="--", lw=1, color="#999", label=f"base rate {y_te.mean():.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall on test months")
    ax.legend(fontsize=8)
    savefig(fig, S.fig_dir / "e1_pr_curves.png")

    s0 = res[res.seed == 0].set_index("model").loc[MODEL_NAMES]
    lo = s0["recall_at_5fpr"] - s0["recall_at_5fpr_ci"].map(lambda c: c[0])
    hi = s0["recall_at_5fpr_ci"].map(lambda c: c[1]) - s0["recall_at_5fpr"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(MODEL_NAMES, s0["recall_at_5fpr"], yerr=[lo, hi], capsize=4, color="#3b6ea5")
    ax.set_ylabel(f"Recall at {S.fpr_target:.0%} FPR"); ax.set_title("Test recall with 95% bootstrap CI")
    savefig(fig, S.fig_dir / "e1_recall_at_fpr.png")
    log("E1 done")
