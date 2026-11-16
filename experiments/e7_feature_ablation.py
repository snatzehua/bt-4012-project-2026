"""E7 - Does feature engineering help, and for which model?

Same protocol as E1 (train months 0-4, threshold on month 5, test months 6-7, tuned
hyperparameters and imbalance strategy from E1). Each model is fitted twice:

  raw         cleaned raw columns only (standard preprocessing: impute, scale, one-hot)
  engineered  raw + row-wise engineered features for every model, and log / level /
              quantile-bin transforms for logistic regression (src/features.py)

The difference engineered - raw is tested with a PAIRED bootstrap on the test months
(both models scored on the same resampled applications).

Outputs
  results/e7_feature_ablation.csv     one row per model x feature set x seed
  results/e7_summary.csv              mean over seeds + paired bootstrap CI of the gain
  results/e7_engineered_importance.csv  rank / gain share of each engineered feature (LightGBM)
  results/e7_ratio_stability.csv      PSI over time: raw velocity columns vs velocity ratios
  figures/e7_feature_ablation.png
"""
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.e0_eda_drift import _psi
from src.config import MONTH, TARGET
from src.data import months
from src.features import ENGINEERED
from src.metrics import (at_threshold, paired_bootstrap_diff, ranking_metrics, recall_at_fpr,
                         threshold_at_fpr)
from src.models import MODEL_NAMES, fit_predict
from src.utils import load_json, log, plt, savefig, strategy_for

FEATURE_SETS = ["raw", "engineered"]


def run(df, nums, cats, S):
    best = load_json(S.results_dir / "e1_best_params.json")
    raw_nums = [c for c in nums if c not in ENGINEERED]
    eng_nums = raw_nums + [c for c in ENGINEERED if c in df]
    cols = {"raw": raw_nums, "engineered": eng_nums}

    tr, va, te = months(df, S.train_months), months(df, S.val_months), months(df, S.test_months)
    y_va, y_te = va[TARGET].to_numpy(), te[TARGET].to_numpy()

    rows, summary, scores, gbm_eng = [], [], {}, None
    for name in MODEL_NAMES:
        seeds = [0] if name == "logreg" else S.seeds
        for fs in FEATURE_SETS:
            S_fs = replace(S, feature_set=fs)
            for seed in seeds:
                model, (s_va, s_te), secs = fit_predict(name, tr, [va, te], cols[fs], cats, S_fs,
                                                        strategy_for(best, name), seed=seed,
                                                        params=best["params"][name])
                thr = threshold_at_fpr(y_va, s_va, S.fpr_target)
                at = at_threshold(y_te, s_te, thr)
                rows.append({"model": name, "feature_set": fs, "seed": seed,
                             "n_numeric_inputs": len(cols[fs]), "fit_seconds": secs,
                             "val_pr_auc": average_precision_score(y_va, s_va),
                             **ranking_metrics(y_te, s_te, S.fpr_target),
                             "valthr_recall": at["recall"], "valthr_precision": at["precision"],
                             "valthr_fpr": at["fpr"]})
                if seed == 0:
                    scores[(name, fs)] = s_te
                    if name == "lightgbm" and fs == "engineered":
                        gbm_eng = model
                log(f"E7 {name:<13} {fs:<10} seed {seed}: val PR-AUC {rows[-1]['val_pr_auc']:.4f}, "
                    f"test PR-AUC {rows[-1]['pr_auc']:.4f}, recall@5%FPR {rows[-1]['recall_at_5fpr']:.3f}")

        # paired bootstrap of the gain (seed 0 scores, same resamples for both)
        res = pd.DataFrame([r for r in rows if r["model"] == name])
        means = res.groupby("feature_set")[["val_pr_auc", "pr_auc", "roc_auc", "recall_at_5fpr",
                                            "valthr_recall", "valthr_fpr"]].mean()
        out = {"model": name}
        for fs in FEATURE_SETS:
            for m in means.columns:
                out[f"{m}_{fs}"] = means.loc[fs, m]
        for metric, fn in [("pr_auc", average_precision_score),
                           ("recall_at_5fpr", lambda a, b: recall_at_fpr(a, b, S.fpr_target))]:
            d, lo, hi = paired_bootstrap_diff(y_te, scores[(name, "engineered")], scores[(name, "raw")],
                                              fn, S.n_bootstrap)
            out.update({f"gain_{metric}": d, f"gain_{metric}_ci_low": lo, f"gain_{metric}_ci_high": hi})
        summary.append(out)
        log(f"E7 {name}: engineered - raw recall@5%FPR {out['gain_recall_at_5fpr']:+.3f} "
            f"[{out['gain_recall_at_5fpr_ci_low']:+.3f}, {out['gain_recall_at_5fpr_ci_high']:+.3f}]")

    res = pd.DataFrame(rows)
    res.to_csv(S.results_dir / "e7_feature_ablation.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(S.results_dir / "e7_summary.csv", index=False)

    # ---- which engineered features does LightGBM actually use? -----------------
    if gbm_eng is not None:
        imp = pd.Series(gbm_eng.booster_.feature_importance("gain"), index=gbm_eng.booster_.feature_name())
        imp = (imp / imp.sum()).sort_values(ascending=False)
        rank = pd.DataFrame({"gain_share": imp, "rank": np.arange(1, len(imp) + 1)}, index=imp.index)
        rank["engineered"] = rank.index.isin(ENGINEERED)
        rank.to_csv(S.results_dir / "e7_engineered_importance.csv", index_label="feature")

    # ---- are the velocity ratios more stable over time than the raw velocities? --
    ref = df[df[MONTH] == df[MONTH].min()]
    stab = []
    for col in ["velocity_6h", "velocity_24h", "velocity_4w",
                "velocity_6h_to_24h", "velocity_24h_to_4w", "velocity_6h_to_4w"]:
        if col not in df:
            continue
        psi = {m: _psi(ref[col], g[col]) for m, g in df.groupby(MONTH) if m != ref[MONTH].iloc[0]}
        stab.append({"variable": col, "engineered": col in ENGINEERED,
                     "max_psi_vs_month0": max(psi.values()), "mean_psi_vs_month0": float(np.mean(list(psi.values())))})
    pd.DataFrame(stab).to_csv(S.results_dir / "e7_ratio_stability.csv", index=False)

    # ---- figure -----------------------------------------------------------------
    agg = res.groupby(["model", "feature_set"])["recall_at_5fpr"].agg(["mean", "std"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(MODEL_NAMES)); w = 0.38
    for i, fs in enumerate(FEATURE_SETS):
        vals = [agg.loc[(m, fs), "mean"] for m in MODEL_NAMES]
        errs = [agg.loc[(m, fs), "std"] for m in MODEL_NAMES]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3, label=fs, color=["#9bb8d6", "#3b6ea5"][i])
    for j, m in enumerate(MODEL_NAMES):
        r = summ.set_index("model").loc[m]
        ax.annotate(f"{100 * r['gain_recall_at_5fpr']:+.1f} pp\n[{100 * r['gain_recall_at_5fpr_ci_low']:+.1f}, "
                    f"{100 * r['gain_recall_at_5fpr_ci_high']:+.1f}]",
                    (x[j], agg.loc[(m, "engineered"), "mean"]), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=7)
    lo = agg["mean"].min()
    ax.set_ylim(max(0, lo - 0.08), agg["mean"].max() + 0.06)
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel("Test recall at 5% FPR")
    ax.set_title("Raw vs engineered features (gain with 95% paired-bootstrap CI)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, S.fig_dir / "e7_feature_ablation.png")
    log("E7 done")
