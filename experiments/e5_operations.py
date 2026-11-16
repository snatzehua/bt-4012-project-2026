"""E5 - Turning scores into decisions: review capacity, costs and fairness.

Uses the validation and test scores saved by E1. Every threshold is chosen on the
validation month and then applied, unchanged, to the test months.

Outputs
  results/e5_capacity.csv     precision/recall if analysts review the top X% of applications
  results/e5_fpr_stability.csv  does a 5%-FPR threshold set in month 5 still give 5% later?
  results/e5_costs.csv        cost-optimal thresholds for several cost ratios (sensitivity analysis)
  results/e5_fairness.csv     false-positive burden by age group (predictive equality)
  results/e5_fairness_no_age.csv  same, after refitting the main model WITHOUT customer_age
  results/e5_confusion_matrices.csv  confusion matrix at two deployable thresholds (Lecture 3)
  results/e5_lecture_cost_table.csv  Lecture 3 slides 46-48 style: which threshold wins under
                              "false alarms expensive" vs "misses expensive" costs
  figures/e5_*.png
"""
import numpy as np
import pandas as pd

from src.config import AGE_COL, AGE_THRESHOLD, MONTH, TARGET
from src.data import months
from src.metrics import (at_threshold, best_cost_threshold, expected_cost, group_rates,
                         threshold_at_capacity, threshold_at_fpr)
from src.models import fit_predict
from src.utils import load_json, log, plt, savefig, strategy_for


def run(df, nums, cats, S):
    best = load_json(S.results_dir / "e1_best_params.json")
    scores = pd.read_csv(S.results_dir / "e1_scores.csv.gz")
    main = best["best_model"]
    log(f"E5: operating-point analysis for {main}")

    def split(model, name):
        d = scores[(scores.model == model) & (scores.split == name)]
        return d[TARGET].to_numpy(), d["score"].to_numpy(), d

    y_va, s_va, _ = split(main, "val")
    y_te, s_te, d_te = split(main, "test")

    # ---- review capacity -----------------------------------------------------
    cap = []
    for c in S.review_capacities:
        thr_val = threshold_at_capacity(s_va, c)  # deployable: set on validation
        m = at_threshold(y_te, s_te, thr_val)
        cap.append({"capacity": c, "reviews_per_1000_apps": 1000 * m["alert_rate"],
                    "precision": m["precision"], "recall": m["recall"],
                    "frauds_caught_per_100_reviews": 100 * m["precision"],
                    "frauds_caught": m["tp"], "frauds_missed": m["fn"]})
    pd.DataFrame(cap).to_csv(S.results_dir / "e5_capacity.csv", index=False)

    # ---- threshold stability over time ----------------------------------------
    thr = threshold_at_fpr(y_va, s_va, S.fpr_target)
    stab = []
    for m in sorted(d_te[MONTH].unique()):
        mask = d_te[MONTH].to_numpy() == m
        r = at_threshold(y_te[mask], s_te[mask], thr)
        stab.append({"month": m, "target_fpr": S.fpr_target, "realised_fpr": r["fpr"],
                     "recall": r["recall"], "alert_rate": r["alert_rate"]})
    pd.DataFrame(stab).to_csv(S.results_dir / "e5_fpr_stability.csv", index=False)

    # ---- cost-sensitive thresholds (sensitivity analysis) ---------------------
    costs = []
    n_fraud = int(y_te.sum())
    for ratio in S.cost_ratios:
        t_cost = best_cost_threshold(y_va, s_va, cost_fn=ratio)
        m = at_threshold(y_te, s_te, t_cost)
        costs.append({"cost_ratio_fn_to_fp": ratio,
                      "cost_optimal_threshold": t_cost,
                      "test_cost": expected_cost(y_te, s_te, t_cost, ratio),
                      "test_cost_at_5fpr_threshold": expected_cost(y_te, s_te, thr, ratio),
                      "test_cost_flag_nothing": n_fraud * ratio,
                      "recall": m["recall"], "precision": m["precision"], "alert_rate": m["alert_rate"]})
    pd.DataFrame(costs).to_csv(S.results_dir / "e5_costs.csv", index=False)

    ratio = S.cost_ratios[len(S.cost_ratios) // 2]
    grid = np.quantile(s_te, np.linspace(0.5, 0.999, 200))
    curve = [(at_threshold(y_te, s_te, t)["alert_rate"], expected_cost(y_te, s_te, t, ratio)) for t in grid]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot([100 * a for a, _ in curve], [c for _, c in curve], color="#3b6ea5")
    chosen = best_cost_threshold(y_va, s_va, cost_fn=ratio)
    m = at_threshold(y_te, s_te, chosen)
    ax.plot(100 * m["alert_rate"], expected_cost(y_te, s_te, chosen, ratio), "o", color="#c0392b",
            label="threshold chosen on validation")
    ax.set_xlabel("Applications flagged (%)"); ax.set_ylabel("Total cost (false alarm = 1)")
    ax.set_title(f"Test cost when a missed fraud costs {ratio}x a false alarm"); ax.legend(fontsize=8)
    savefig(fig, S.fig_dir / "e5_cost_curve.png")

    # ---- fairness: false-positive burden by age group -------------------------
    fair = []
    for model in scores.model.unique():
        yv, sv, _ = split(model, "val")
        yt, st, dt = split(model, "test")
        t = threshold_at_fpr(yv, sv, S.fpr_target)
        groups = np.where(dt[AGE_COL].to_numpy() >= AGE_THRESHOLD, f">={AGE_THRESHOLD}", f"<{AGE_THRESHOLD}")
        per, ratio_pe = group_rates(yt, st, t, groups)
        for g, r in per.items():
            fair.append({"model": model, "group": g, "fpr": r["fpr"], "recall": r["recall"],
                         "alert_rate": r["alert_rate"], "predictive_equality_ratio": ratio_pe})
    fair = pd.DataFrame(fair)
    fair.to_csv(S.results_dir / "e5_fairness.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    models = list(fair.model.unique()); x = np.arange(len(models)); w = 0.38
    for i, g in enumerate(sorted(fair.group.unique())):
        vals = [fair[(fair.model == m) & (fair.group == g)]["fpr"].iloc[0] for m in models]
        ax.bar(x + (i - 0.5) * w, vals, w, label=f"age {g}", color=["#9bb8d6", "#3b6ea5"][i])
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("False positive rate")
    ax.set_title("False-positive burden by age group"); ax.legend(fontsize=8)
    savefig(fig, S.fig_dir / "e5_fairness.png")

    # ---- fairness check: does removing customer_age remove the gap? -----------------
    if main != "random_forest":
        no_age = [c for c in nums if c != AGE_COL]
        tr, va, te = months(df, S.train_months), months(df, S.val_months), months(df, S.test_months)
        _, (sv, st), _ = fit_predict(main, tr, [va, te], no_age, cats, S, strategy_for(best, main),
                                     seed=0, params=best["params"][main])
        t = threshold_at_fpr(va[TARGET].to_numpy(), sv, S.fpr_target)
        groups = np.where(te[AGE_COL].to_numpy() >= AGE_THRESHOLD, f">={AGE_THRESHOLD}", f"<{AGE_THRESHOLD}")
        per, ratio_pe = group_rates(te[TARGET].to_numpy(), st, t, groups)
        pd.DataFrame([{"model": f"{main} without {AGE_COL}", "group": g, "fpr": r["fpr"], "recall": r["recall"],
                       "alert_rate": r["alert_rate"], "predictive_equality_ratio": ratio_pe}
                      for g, r in per.items()]).to_csv(S.results_dir / "e5_fairness_no_age.csv", index=False)
        log(f"E5: without {AGE_COL}, predictive-equality ratio {ratio_pe:.2f}")

    # ---- Lecture 3: confusion matrices and the two-threshold cost table --------------
    thresholds = {f"T1: review top {S.review_capacities[0]:.0%} (set on validation)":
                  threshold_at_capacity(s_va, S.review_capacities[0]),
                  f"T2: {S.fpr_target:.0%} FPR (set on validation)": thr}
    cms = []
    for label, t in thresholds.items():
        m = at_threshold(y_te, s_te, t)
        cms.append({"threshold": label, "TP (fraud caught)": m["tp"], "FP (false alarm)": m["fp"],
                    "FN (fraud missed)": m["fn"], "TN (legit passed)": m["tn"],
                    "accuracy": (m["tp"] + m["tn"]) / len(y_te), "precision": m["precision"],
                    "recall": m["recall"]})
    cms = pd.DataFrame(cms)
    cms.to_csv(S.results_dir / "e5_confusion_matrices.csv", index=False)
    cost_rows = []
    for cost_name, c_fp, c_fn in [("A: false alarms expensive (FP 100, FN 1)", 100, 1),
                                  ("B: misses expensive (FP 1, FN 100)", 1, 100)]:
        costs = {r["threshold"]: r["FP (false alarm)"] * c_fp + r["FN (fraud missed)"] * c_fn
                 for _, r in cms.iterrows()}
        cost_rows.append({"cost_table": cost_name, **costs, "cheaper_threshold": min(costs, key=costs.get)})
    pd.DataFrame(cost_rows).to_csv(S.results_dir / "e5_lecture_cost_table.csv", index=False)

    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.4))
    for ax, (_, r) in zip(np.atleast_1d(axes), cms.iterrows()):
        mat = np.array([[r["TP (fraud caught)"], r["FN (fraud missed)"]], [r["FP (false alarm)"], r["TN (legit passed)"]]])
        ax.imshow(np.log10(mat + 1), cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", fontsize=10,
                        color="white" if (i, j) == (1, 1) else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["flagged", "not flagged"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["fraud", "legit"])
        ax.set_title(f"{r['threshold']}\naccuracy {r['accuracy']:.1%}, precision {r['precision']:.1%}, "
                     f"recall {r['recall']:.1%}", fontsize=8)
    savefig(fig, S.fig_dir / "e5_confusion_matrices.png")
    log("E5 done")
