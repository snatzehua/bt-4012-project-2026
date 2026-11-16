"""E3 - Does a recent rolling window beat expanding history?

For each test month t:
  expanding            train on every month < t
  rolling_k            train on months t-k .. t-1 only
  expanding_matched_k  a random sample of the expanding window with the SAME number of
                       rows as rolling_k.  Comparing rolling_k with expanding_matched_k
                       isolates RECENCY from simply having less data.

Model: tuned LightGBM from E1 (fast enough for many refits). Repeated over seeds.

Outputs: results/e3_windows.csv (every run), results/e3_summary.csv, figures/e3_*.png
"""
import numpy as np
import pandas as pd

from src.config import MONTH, TARGET
from src.data import months
from src.metrics import ranking_metrics
from src.models import fit_predict
from src.utils import load_json, log, plt, savefig, strategy_for

MODEL = "lightgbm"


def run(df, nums, cats, S):
    best = load_json(S.results_dir / "e1_best_params.json")
    params, strategy = best["params"][MODEL], strategy_for(best, MODEL)
    rows = []
    for t in S.window_test_months:
        te = months(df, [t])
        y = te[TARGET].to_numpy()
        expanding = months(df, list(range(0, t)))
        for seed in S.seeds:
            schemes = [("expanding", None, expanding)]
            for k in S.window_ks:
                if k >= t:
                    continue  # identical to expanding
                rolling = months(df, list(range(t - k, t)))
                matched = expanding.sample(n=len(rolling), random_state=seed)
                schemes += [("rolling", k, rolling), ("expanding_matched", k, matched)]
            for scheme, k, tr in schemes:
                _, (s,), secs = fit_predict(MODEL, tr, [te], nums, cats, S, strategy, seed=seed, params=params)
                rows.append({"test_month": t, "scheme": scheme, "k": k, "seed": seed,
                             "n_train": len(tr), "train_months": f"{tr[MONTH].min()}-{tr[MONTH].max()}",
                             "fit_seconds": secs, **ranking_metrics(y, s, S.fpr_target)})
                log(f"E3 t={t} {scheme:<17} k={k} seed={seed} n={len(tr):,}: "
                    f"recall@5%FPR {rows[-1]['recall_at_5fpr']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(S.results_dir / "e3_windows.csv", index=False)

    # Paired comparisons (same test month, same seed)
    key = ["test_month", "seed"]
    exp = res[res.scheme == "expanding"].set_index(key)
    out = []
    for k in S.window_ks:
        rol = res[(res.scheme == "rolling") & (res.k == k)].set_index(key)
        mat = res[(res.scheme == "expanding_matched") & (res.k == k)].set_index(key)
        if rol.empty:
            continue
        for metric in ["pr_auc", "recall_at_5fpr"]:
            d_exp = (rol[metric] - exp.loc[rol.index, metric])
            d_rec = (rol[metric] - mat.loc[rol.index, metric])
            out.append({"k": k, "metric": metric,
                        "rolling_mean": rol[metric].mean(), "expanding_mean": exp.loc[rol.index, metric].mean(),
                        "rolling_minus_expanding": d_exp.mean(), "share_rolling_wins": float((d_exp > 0).mean()),
                        "recency_effect_vs_matched": d_rec.mean(), "n_pairs": len(d_exp)})
    pd.DataFrame(out).to_csv(S.results_dir / "e3_summary.csv", index=False)

    # ---- figures ----------------------------------------------------------
    mean = res.groupby(["test_month", "scheme", "k"], dropna=False)["recall_at_5fpr"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    e = mean[mean.scheme == "expanding"]
    ax.plot(e.test_month, e.recall_at_5fpr, "o-", color="black", label="expanding")
    for k, c in zip(S.window_ks, ["#9bb8d6", "#3b6ea5", "#1d3f66"]):
        r = mean[(mean.scheme == "rolling") & (mean.k == k)]
        ax.plot(r.test_month, r.recall_at_5fpr, "o--", color=c, label=f"rolling k={k}")
    ax.set_xticks(S.window_test_months)
    ax.set_xlabel("Test month"); ax.set_ylabel("Recall at 5% FPR")
    ax.set_title("Training window vs future performance"); ax.legend(fontsize=8)
    savefig(fig, S.fig_dir / "e3_windows.png")

    summ = pd.DataFrame(out)
    summ = summ[summ.metric == "recall_at_5fpr"]
    if not summ.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        x = np.arange(len(summ)); w = 0.38
        ax.bar(x - w / 2, summ["rolling_minus_expanding"], w, label="rolling - expanding (total effect)", color="#9bb8d6")
        ax.bar(x + w / 2, summ["recency_effect_vs_matched"], w, label="rolling - size-matched (recency only)", color="#3b6ea5")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in summ.k])
        ax.set_ylabel("Change in recall at 5% FPR"); ax.legend(fontsize=7)
        ax.set_title("Separating recency from training-set size")
        savefig(fig, S.fig_dir / "e3_recency_vs_size.png")
    log("E3 done")
