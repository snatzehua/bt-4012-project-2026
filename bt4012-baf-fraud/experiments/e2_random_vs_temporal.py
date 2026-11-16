"""E2 - How optimistic is a random train/test split?

Temporal: train months 0-5, test months 6-7 (the deployable setting).
Random:   a stratified random split of all months with the SAME test size.
Same models, same hyperparameters; only the split differs.

Outputs: results/e2_random_vs_temporal.csv, results/e2_summary.csv, figures/e2_split_gap.png
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TARGET
from src.data import months
from src.metrics import ranking_metrics
from src.models import fit_predict
from src.utils import load_json, log, plt, savefig

MODELS = ["logreg", "lightgbm"]


def run(df, nums, cats, S):
    best = load_json(S.results_dir / "e1_best_params.json")
    tr_t = months(df, S.train_months + S.val_months)
    te_t = months(df, S.test_months)
    test_share = len(te_t) / (len(tr_t) + len(te_t))
    pool = pd.concat([tr_t, te_t])
    rows = []
    for name in MODELS:
        for seed in ([0] if name == "logreg" else S.seeds):
            tr_r, te_r = train_test_split(pool, test_size=test_share, stratify=pool[TARGET], random_state=seed)
            for split, tr, te in [("temporal", tr_t, te_t), ("random", tr_r, te_r)]:
                _, (s,), _ = fit_predict(name, tr, [te], nums, cats, S, best["strategy"],
                                         seed=seed, params=best["params"][name])
                rows.append({"model": name, "seed": seed, "split": split,
                             **ranking_metrics(te[TARGET].to_numpy(), s, S.fpr_target)})
                log(f"E2 {name} seed {seed} {split}: PR-AUC {rows[-1]['pr_auc']:.4f}, "
                    f"recall@5%FPR {rows[-1]['recall_at_5fpr']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(S.results_dir / "e2_random_vs_temporal.csv", index=False)

    summ = res.groupby(["model", "split"])[["pr_auc", "roc_auc", "recall_at_5fpr"]].mean().unstack("split")
    for m in ["pr_auc", "roc_auc", "recall_at_5fpr"]:
        summ[(m, "optimism_gap")] = summ[(m, "random")] - summ[(m, "temporal")]
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(S.results_dir / "e2_summary.csv")

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(MODELS)); w = 0.35
    for i, split in enumerate(["random", "temporal"]):
        vals = [summ.loc[m, f"recall_at_5fpr_{split}"] for m in MODELS]
        ax.bar(x + (i - 0.5) * w, vals, w, label=split, color=["#9bb8d6", "#3b6ea5"][i])
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylabel("Recall at 5% FPR"); ax.set_title("Random split vs out-of-time split"); ax.legend()
    savefig(fig, S.fig_dir / "e2_split_gap.png")
    log("E2 done")
