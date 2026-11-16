"""E1 - Model comparison under the out-of-time protocol.

Train on months 0-4, tune and set thresholds on month 5, test on months 6-7.
The hyperparameters AND the class-imbalance strategy (none / class_weight / undersample)
are chosen per model on the validation month only (Lecture 3: tune on validation, never
on test). E4 then reports all strategies on the test months for comparison.
A constant-score "dummy" baseline (Lecture 3: start with a simple benchmark) gives the
no-skill reference: PR-AUC = fraud prevalence, ROC-AUC = 0.5.

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

STRATEGIES = ["none", "class_weight", "undersample"]


def run(df, nums, cats, S):
    tr, va, te = months(df, S.train_months), months(df, S.val_months), months(df, S.test_months)
    y_tr, y_va, y_te = tr[TARGET].to_numpy(), va[TARGET].to_numpy(), te[TARGET].to_numpy()
    log(f"E1: train {len(tr):,} rows (months {S.train_months}), val {len(va):,}, test {len(te):,}")

    # ---- 1. tune a small grid per model on the validation month ----------
    tuning, best, best_strategy, best_val = [], {}, {}, {}
    for name in MODEL_NAMES:
        for strategy in STRATEGIES:
            for params in PARAM_GRID[name]:
                _, (s_va,), secs = fit_predict(name, tr, [va], nums, cats, S, strategy, seed=0, params=params)
                tuning.append({"model": name, "strategy": strategy, "params": params,
                               "val_pr_auc": average_precision_score(y_va, s_va),
                               "val_recall_at_5fpr": recall_at_fpr(y_va, s_va, S.fpr_target), "fit_seconds": secs})
                log(f"  tune {name} {strategy} {params}: val PR-AUC {tuning[-1]['val_pr_auc']:.4f}")
        top = max((t for t in tuning if t["model"] == name), key=lambda t: t["val_pr_auc"])
        best[name], best_strategy[name], best_val[name] = top["params"], top["strategy"], top["val_pr_auc"]
    pd.DataFrame(tuning).to_csv(S.results_dir / "e1_tuning.csv", index=False)
    best_model = max(best_val, key=best_val.get)
    save_json({"params": best, "best_model": best_model, "strategy": best_strategy,
               "val_pr_auc": best_val}, S.results_dir / "e1_best_params.json")
    log(f"E1: best params {best}; strategies {best_strategy}; best model on validation: {best_model}")

    # ---- 2. evaluate each tuned model on the untouched test months --------
    rows, per_month, score_frames = [], [], []
    for name in MODEL_NAMES:
        seeds = [0] if name == "logreg" else S.seeds  # logistic regression is deterministic
        for seed in seeds:
            model, (s_tr, s_va, s_te), secs = fit_predict(name, tr, [tr, va, te], nums, cats, S,
                                                          best_strategy[name], seed=seed, params=best[name])
            thr = threshold_at_fpr(y_va, s_va, S.fpr_target)  # chosen on validation only
            at = at_threshold(y_te, s_te, thr)
            row = {"model": name, "strategy": best_strategy[name], "seed": seed, "fit_seconds": secs,
                   # Lecture 3 overfitting check: training score much higher than test = overfit
                   "train_pr_auc": float(average_precision_score(y_tr, s_tr)),
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

    # constant-score reference (no skill): every application gets the training fraud rate
    s_dummy = np.full(len(y_te), y_tr.mean())
    rows.append({"model": "dummy", "strategy": "none", "seed": 0, "fit_seconds": 0.0,
                 "train_pr_auc": float(y_tr.mean()), "pr_auc": float(y_te.mean()), "roc_auc": 0.5,
                 "recall_at_5fpr": 0.05, "valthr_precision": float(y_te.mean()), "valthr_recall": 0.05,
                 "valthr_fpr": 0.05, "top1pct_precision": float(y_te.mean()), "top5pct_recall": 0.05})
    res = pd.DataFrame(rows)
    res.to_csv(S.results_dir / "e1_model_comparison.csv", index=False)
    pd.DataFrame(per_month).to_csv(S.results_dir / "e1_per_month.csv", index=False)
    pd.concat(score_frames).to_csv(S.results_dir / "e1_scores.csv.gz", index=False)

    metrics = ["train_pr_auc", "pr_auc", "roc_auc", "recall_at_5fpr", "valthr_recall", "valthr_precision", "valthr_fpr",
               "top1pct_precision", "top5pct_recall", "fit_seconds"]
    summ = res.groupby("model")[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    ci = res[res.seed == 0].set_index("model")
    for c in ["pr_auc_ci", "recall_at_5fpr_ci"]:
        summ[c.replace("_ci", "_ci_low")] = ci[c].map(lambda t: t[0] if isinstance(t, (tuple, list)) else np.nan)
        summ[c.replace("_ci", "_ci_high")] = ci[c].map(lambda t: t[1] if isinstance(t, (tuple, list)) else np.nan)
    summ.insert(0, "strategy", res.groupby("model")["strategy"].first())
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
