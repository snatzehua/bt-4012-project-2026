"""E3b - Recency at a FIXED training size (separates recency from data size).

E3 compares rolling and expanding windows, but those differ in BOTH recency and size.
Here every arm has exactly K training rows, so only the AGE of the data differs.
For each test month t and size K:

  recent   the K most recent applications before t: whole months t-1, t-2, ...
           and a random part of the next older month to make exactly K rows
           (BAF has no timestamp inside a month, so this is as recent as the data allows)
  oldest   the K oldest applications: months 0, 1, ... filled the same way
  random   K applications sampled uniformly from all months before t

recent vs oldest = the recency effect at size K (the two never overlap: K is skipped
if 2K is larger than the available history). Reading the curves along K shows the
value of extra data at a fixed recency, which E3 cannot separate.

Outputs: results/e3b_fixed_k.csv (every fit), results/e3b_summary.csv (paired effects),
         figures/e3b_learning_curve.png, figures/e3b_recency_effect.png
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.config import MONTH, TARGET
from src.data import months
from src.metrics import ranking_metrics
from src.models import fit_predict
from src.utils import load_json, log, plt, savefig, strategy_for

ARMS = ["recent", "random", "oldest"]
COLORS = {"recent": "#1d3f66", "random": "#7f9fbf", "oldest": "#c0392b"}


def _take(pool, month_order, k, seed):
    """Whole months in `month_order` until K rows; the last month is randomly sampled."""
    parts, n = [], 0
    for m in month_order:
        need = k - n
        if need <= 0:
            break
        g = pool[pool[MONTH] == m]
        parts.append(g if len(g) <= need else g.sample(n=need, random_state=seed))
        n += len(parts[-1])
    return pd.concat(parts)


def run(df, nums, cats, S):
    best = load_json(S.results_dir / "e1_best_params.json")
    rows = []
    for t in S.fixed_k_test_months:
        te = months(df, [t])
        y = te[TARGET].to_numpy()
        pool = months(df, list(range(0, t)))
        for k_nominal in S.fixed_k_sizes:
            k = int(round(k_nominal * S.sample_frac))
            if 2 * k > len(pool):
                log(f"E3b t={t} K={k:,}: skipped (recent and oldest would overlap; history {len(pool):,})")
                continue
            for seed in S.seeds:
                arms = {"recent": _take(pool, range(t - 1, -1, -1), k, seed),
                        "oldest": _take(pool, range(0, t), k, seed),
                        "random": pool.sample(n=k, random_state=seed)}
                for name in S.fixed_k_models:
                    for arm in ARMS:
                        tr = arms[arm]
                        _, (s,), secs = fit_predict(name, tr, [te], nums, cats, S, strategy_for(best, name),
                                                    seed=seed, params=best["params"][name])
                        rows.append({"model": name, "test_month": t, "K": k_nominal, "n_train": len(tr),
                                     "arm": arm, "seed": seed,
                                     "train_months": f"{tr[MONTH].min()}-{tr[MONTH].max()}",
                                     "mean_data_age_months": float(t - tr[MONTH].mean()),
                                     "train_fraud_rate": float(tr[TARGET].mean()), "fit_seconds": secs,
                                     **ranking_metrics(y, s, S.fpr_target)})
                log(f"E3b t={t} K={k:,} seed={seed}: " + ", ".join(
                    f"{r['model'][:4]}/{r['arm']} {r['recall_at_5fpr']:.3f}" for r in rows[-len(ARMS) * len(S.fixed_k_models):]))
    if not rows:
        log("E3b: every K was too large for the available history; nothing to report")
        return
    res = pd.DataFrame(rows)
    res.to_csv(S.results_dir / "e3b_fixed_k.csv", index=False)

    # ---- paired effects: same model, test month, K and seed ---------------------
    key = ["model", "test_month", "K", "seed"]
    out = []
    for metric in ["recall_at_5fpr", "pr_auc", "roc_auc"]:
        wide = res.pivot_table(index=key, columns="arm", values=metric)
        for (model, K), g in wide.groupby(level=["model", "K"]):
            row = {"model": model, "K": K, "metric": metric, "n_pairs": len(g)}
            for arm in ARMS:
                row[f"{arm}_mean"] = g[arm].mean()
            for a, b in [("recent", "oldest"), ("recent", "random")]:
                d = g[a] - g[b]
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                row[f"{a}_minus_{b}"] = d.mean()
                row[f"{a}_minus_{b}_ci95"] = 1.96 * se if len(d) > 1 else np.nan
                row[f"{a}_minus_{b}_p"] = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 else np.nan
                row[f"share_{a}_beats_{b}"] = float((d > 0).mean())
            out.append(row)
    summ = pd.DataFrame(out)
    summ.to_csv(S.results_dir / "e3b_summary.csv", index=False)

    # ---- figures ------------------------------------------------------------
    models = list(S.fixed_k_models)
    mean = res.groupby(["model", "K", "arm"])["recall_at_5fpr"].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.6), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        for arm in ARMS:
            d = mean[(mean.model == model) & (mean.arm == arm)]
            ax.errorbar(d["K"], d["mean"], yerr=d["std"], marker="o", capsize=3, color=COLORS[arm],
                        label=f"{arm} K rows")
        ax.set_xscale("log"); ax.set_xlabel("Training rows K (log scale)")
        ax.set_title(model)
    axes[0][0].set_ylabel("Test recall at 5% FPR"); axes[0][0].legend(fontsize=8)
    fig.suptitle("Same amount of data, different age: recent vs oldest vs random history", fontsize=10)
    savefig(fig, S.fig_dir / "e3b_learning_curve.png")

    s = summ[summ.metric == "recall_at_5fpr"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ks = sorted(s.K.unique()); x = np.arange(len(ks)); w = 0.8 / len(models)
    for i, model in enumerate(models):
        d = s[s.model == model].set_index("K").reindex(ks)
        ax.bar(x + (i - (len(models) - 1) / 2) * w, 100 * d["recent_minus_oldest"], w,
               yerr=100 * d["recent_minus_oldest_ci95"], capsize=3, label=model,
               color=["#3b6ea5", "#9bb8d6"][i % 2])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{k // 1000}k" for k in ks])
    ax.set_xlabel("Training rows K"); ax.set_ylabel("Recent - oldest (pp recall at 5% FPR)")
    ax.set_title("Recency effect at fixed training size (95% CI)", fontsize=10); ax.legend(fontsize=8)
    savefig(fig, S.fig_dir / "e3b_recency_effect.png")
    log("E3b done")
