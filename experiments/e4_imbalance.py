"""E4 - Class-imbalance treatments.

Strategies: none | class_weight | undersample (keep all frauds, S.undersample_ratio
legitimate applications per fraud). Thresholds are chosen on the validation month
for 5% FPR, then applied to the test months, so we also see whether a strategy
makes the threshold less stable over time.

Outputs: results/e4_imbalance.csv, results/e4_summary.csv, figures/e4_imbalance.png
"""
import numpy as np
import pandas as pd

from src.config import TARGET
from src.data import months
from src.metrics import at_threshold, capacity_metrics, ranking_metrics, threshold_at_fpr
from src.models import fit_predict
from src.utils import load_json, log, plt, savefig

MODELS = ["logreg", "lightgbm"]
STRATEGIES = ["none", "class_weight", "undersample"]


def run(df, nums, cats, S):
    best = load_json(S.results_dir / "e1_best_params.json")
    tr, va, te = months(df, S.train_months), months(df, S.val_months), months(df, S.test_months)
    y_va, y_te = va[TARGET].to_numpy(), te[TARGET].to_numpy()
    rows = []
    for name in MODELS:
        for strategy in STRATEGIES:
            for seed in ([0] if (name == "logreg" and strategy != "undersample") else S.seeds):
                _, (s_va, s_te), secs = fit_predict(name, tr, [va, te], nums, cats, S, strategy,
                                                    seed=seed, params=best["params"][name])
                thr = threshold_at_fpr(y_va, s_va, S.fpr_target)
                at = at_threshold(y_te, s_te, thr)
                rows.append({"model": name, "strategy": strategy, "seed": seed, "fit_seconds": secs,
                             **ranking_metrics(y_te, s_te, S.fpr_target),
                             "valthr_recall": at["recall"], "valthr_precision": at["precision"],
                             "valthr_fpr": at["fpr"], "valthr_f1": at["f1"],
                             **capacity_metrics(y_te, s_te, S.review_capacities)})
                log(f"E4 {name} {strategy} seed {seed}: recall@5%FPR {rows[-1]['recall_at_5fpr']:.3f}, "
                    f"realised FPR {at['fpr']:.3f}, fit {secs:.0f}s")
    res = pd.DataFrame(rows)
    res.to_csv(S.results_dir / "e4_imbalance.csv", index=False)
    summ = res.groupby(["model", "strategy"]).mean(numeric_only=True).drop(columns="seed")
    summ.to_csv(S.results_dir / "e4_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    x = np.arange(len(STRATEGIES)); w = 0.38
    for ax, metric, title in [(axes[0], "recall_at_5fpr", "Recall at 5% FPR"),
                              (axes[1], "valthr_fpr", "Realised test FPR\n(threshold set on validation)")]:
        for i, name in enumerate(MODELS):
            vals = [summ.loc[(name, s), metric] for s in STRATEGIES]
            ax.bar(x + (i - 0.5) * w, vals, w, label=name, color=["#9bb8d6", "#3b6ea5"][i])
        ax.set_xticks(x); ax.set_xticklabels(STRATEGIES); ax.set_title(title, fontsize=10)
    axes[1].axhline(S.fpr_target, ls="--", color="#999", lw=1)
    axes[0].legend(fontsize=8)
    savefig(fig, S.fig_dir / "e4_imbalance.png")
    log("E4 done")
