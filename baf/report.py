"""Audit completed study outputs and assemble shareable tables and figures."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .common import budget_name, read_config, write_json
from .study import collect, identity, specs_for_validation


DISPLAY = {"logistic": "Logistic regression", "lightgbm": "LightGBM", "random_forest": "Random forest"}


def md_table(table, digits=4):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join(lines)


def audit(config_path):
    config = read_config(config_path)
    out = Path(config["output_dir"])
    selection = json.loads((out / "selection.json").read_text())
    fingerprint = identity(config_path)
    assert selection["fingerprint"] == fingerprint
    assert selection["test_consumed"]
    assert datetime.fromisoformat(selection["first_final_access_at_utc"]) >= datetime.fromisoformat(selection["created_at_utc"])
    validation, _ = collect(config, "validation")
    final, _ = collect(config, "final")
    diagnostic, _ = collect(config, "diagnostic")
    assert len(validation) == len(config["validation_months"]) * len(list(specs_for_validation(config))) == 90
    assert len(final) == len(selection["final_specs"]) == 14
    assert len(diagnostic) == 3
    monthly = pd.read_csv(out / "eda/monthly.csv")
    rates = pd.read_csv(out / "eda/feature_fraud_rates.csv")
    assert set(monthly.month) == set(range(config["test_month"]))
    assert (rates.groupby("feature").rows.sum() == monthly.rows.sum()).all()
    assert (rates.groupby("feature").frauds.sum() == monthly.fraud.sum()).all()
    for records in [validation, final, diagnostic]:
        for r in records:
            assert r["fingerprint"] == fingerprint
            assert not any("ConvergenceWarning" in w for w in r["warnings"]), r["run"]
            assert max(map(int, r["train_months"])) < (config["test_month"] if r["phase"] == "diagnostic" else r["eval_month"])
            for fraction in config["review_fractions"]:
                suffix = "_at_" + budget_name(fraction)
                assert r["reviewed" + suffix] == math.ceil(fraction * r["n_eval"])
                assert r["tp" + suffix] + r["fn" + suffix] == r["fraud_eval"]
                assert r["tp" + suffix] + r["fp" + suffix] == r["reviewed" + suffix]
                assert sum(r[key + suffix] for key in ["tp", "fp", "fn", "tn"]) == r["n_eval"]
    for records in [validation, final]:
        table = pd.DataFrame(records)
        for _, month in table.groupby("eval_month"):
            assert month.evaluation_ids_sha256.nunique() == 1
        for r in records:
            if r["strategy"] == "historical_matched":
                reference = next(t for t in records if t["model"] == r["model"] and t["eval_month"] == r["eval_month"] and t["k"] == r["k"] and t["strategy"] == "recent")
                assert r["train_rows"] == reference["train_rows"] == r["k"]
                assert r["train_fraud"] == reference["train_fraud"]
    checks = {"validation_fits": len(validation), "final_fits": len(final), "diagnostic_fits": len(diagnostic),
              "source_config_hashes_match": True, "train_boundaries_pass": True, "same_future_cases_pass": True,
              "matched_sizes_and_classes_pass": True, "budget_confusion_accounting_pass": True,
              "convergence_warnings": 0, "freeze_precedes_final_access": True,
              "eda_months_and_target_group_totals_pass": True}
    write_json(out / "audit.json", checks)
    return checks


def generate(config_path):
    config = read_config(config_path)
    out = Path(config["output_dir"])
    checks = audit(config_path)
    report_dir = Path("reports") / out.name
    report_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((out / "selection.json").read_text())
    validation = pd.read_csv(out / "validation_metrics.csv")
    final = pd.read_csv(out / "final_metrics.csv")
    diagnostic = pd.read_csv(out / "diagnostic_metrics.csv")
    paired = pd.read_csv(out / "paired_differences.csv")
    monthly = pd.read_csv(out / "eda/monthly.csv")
    inventory = pd.read_csv(out / "eda/feature_inventory.csv")
    drift = pd.read_csv(out / "eda/drift_month0_vs_month6.csv")
    quality = json.loads((out / "eda/quality.json").read_text())
    for name in ["validation_metrics.csv", "final_metrics.csv", "diagnostic_metrics.csv", "paired_differences.csv", "validation_selection_summary.csv", "selection.json", "audit.json", "logistic_features.csv", "lightgbm_features.csv", "random_forest_features.csv"]:
        shutil.copy2(out / name, report_dir / name)
    for path in (out / "eda").iterdir():
        shutil.copy2(path, report_dir / path.name)
    # Equal month weights after averaging historical seeds within a month.
    keys = ["model", "strategy", "k", "weighted"]
    validation_monthly = validation.groupby([*keys, "eval_month"], as_index=False)[["recall_at_1pct", "precision_at_1pct", "average_precision", "fit_seconds"]].mean()
    validation_summary = validation_monthly.groupby(keys, as_index=False).agg(
        mean_recall=("recall_at_1pct", "mean"), min_recall=("recall_at_1pct", "min"), max_recall=("recall_at_1pct", "max"),
        mean_precision=("precision_at_1pct", "mean"), mean_ap=("average_precision", "mean"), mean_fit_seconds=("fit_seconds", "mean"))
    validation_summary.to_csv(report_dir / "validation_summary.csv", index=False)
    recency = []
    for model in config["models"]:
        for k in config["sample_sizes"]:
            subset = validation_monthly[(validation_monthly.model == model) & (validation_monthly.k == k)]
            a = subset[subset.strategy == "recent"].set_index("eval_month").recall_at_1pct
            b = subset[subset.strategy == "historical_matched"].set_index("eval_month").recall_at_1pct
            delta = a - b
            recency.append({"model": model, "k": k, "recent_mean_recall": a.mean(), "matched_mean_recall": b.mean(),
                            "difference_pp": 100 * delta.mean(), "min_month_difference_pp": 100 * delta.min(),
                            "max_month_difference_pp": 100 * delta.max(), "months_recent_better": int((delta > 0).sum())})
    recency = pd.DataFrame(recency)
    recency.to_csv(report_dir / "validation_recency.csv", index=False)
    final_summary = final.groupby(keys, as_index=False).agg(
        recall=("recall_at_1pct", "mean"), precision=("precision_at_1pct", "mean"), ap=("average_precision", "mean"),
        tp=("tp_at_1pct", "mean"), fp=("fp_at_1pct", "mean"), missed=("fn_at_1pct", "mean"),
        min_seed_recall=("recall_at_1pct", "min"), max_seed_recall=("recall_at_1pct", "max"), fit_seconds=("fit_seconds", "mean"))
    final_summary.to_csv(report_dir / "final_summary.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, config["models"]):
        for strategy, color, label in [("recent", "#bd5c35", "Recent K"), ("historical_matched", "#366589", "Matched historical K")]:
            values = validation_summary[(validation_summary.model == model) & (validation_summary.strategy == strategy)].sort_values("k")
            ax.plot(values.k / 1000, values.mean_recall * 100, marker="o", label=label, color=color)
            ax.fill_between(values.k / 1000, values.min_recall * 100, values.max_recall * 100, color=color, alpha=0.12)
        baseline = validation_summary[(validation_summary.model == model) & (validation_summary.strategy == "expanding") & ~validation_summary.weighted].iloc[0]
        ax.axhline(100 * baseline.mean_recall, color="#555555", linestyle="--", label="Expanding history")
        ax.set(xlabel="Training rows K (thousands)", title=DISPLAY[model], ylabel="Recall at 1% review (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Forward validation: mean across months 4-6; shading is month range")
    fig.tight_layout()
    fig.savefig(report_dir / "validation_training_size.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, config["models"]):
        rows = final_summary[(final_summary.model == model) & ~final_summary.weighted]
        ordering = ["expanding", "recent", "historical_matched"]
        values = [rows[rows.strategy == s].iloc[0] for s in ordering]
        recalls = np.array([v.recall for v in values]) * 100
        lower = np.array([v.min_seed_recall for v in values]) * 100
        upper = np.array([v.max_seed_recall for v in values]) * 100
        ax.bar(["All history", "Recent K", "Matched K"], recalls, color=["#777777", "#bd5c35", "#366589"],
               yerr=np.array([recalls - lower, upper - recalls]), capsize=4)
        for i, recall in enumerate(recalls):
            ax.text(i, recall + 0.5, f"{recall:.1f}%", ha="center", fontsize=9)
        ax.set(title=f"{DISPLAY[model]} (K={selection['selected_k'][model]:,})", ylabel="Recall at 1% review (%)")
        ax.set_ylim(0, max(upper) * 1.18)
    upper_limit = final_summary[(final_summary.model.isin(config["models"])) & ~final_summary.weighted].max_seed_recall.max() * 100 * 1.18
    axes[0].set_ylim(0, upper_limit)
    fig.suptitle("Final month 7: training strategy comparison; whiskers show historical-seed range")
    fig.tight_layout()
    fig.savefig(report_dir / "final_recency.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for weighted, color in [(False, "#366589"), (True, "#bd5c35")]:
        rows = final[(final.strategy == "expanding") & (final.weighted == weighted)].set_index("model")
        pos = np.arange(3) + (0.18 if weighted else -0.18)
        axes[0].bar(pos, [rows.loc[m, "recall_at_1pct"] * 100 for m in [*config["models"], "random_forest"]], width=0.36, color=color, label="Balanced weights" if weighted else "Unweighted")
        axes[1].bar(pos, [rows.loc[m, "default_precision"] * 100 for m in [*config["models"], "random_forest"]], width=0.36, color=color)
    for ax in axes:
        ax.set_xticks(np.arange(3), ["Logistic", "LightGBM", "Random forest"])
    axes[0].set(ylabel="Recall at 1% review (%)", title="Ranking under equal review capacity")
    axes[1].set(ylabel="Precision at threshold 0.5 (%)", title="Default-threshold precision changes with weighting")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "weighting_and_models.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, color in [("logistic", "#366589"), ("lightgbm", "#bd5c35"), ("random_forest", "#6a8a45")]:
        row = final[(final.model == model) & (final.strategy == "expanding") & ~final.weighted].iloc[0]
        with np.load(out / "predictions" / f"{row.run}.npz") as data:
            precision, recall, _ = precision_recall_curve(data["labels"], data["scores"])
        axes[0].plot(recall, precision, label=f"{DISPLAY[model]} (AP={row.average_precision:.3f})", color=color)
        axes[1].plot(np.array(config["review_fractions"]) * 100,
                     [row["recall_at_" + budget_name(q)] * 100 for q in config["review_fractions"]], marker="o", label=DISPLAY[model], color=color)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Final-month precision-recall curves")
    axes[1].set(xlabel="Review capacity (%)", ylabel="Fraud captured (%)", title="Detection at fixed review capacities")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "operating_curves.png", dpi=170)
    plt.close(fig)
    winner = selection["validation_winner"]
    champion = final_summary.copy()
    for key, value in winner.items():
        champion = champion[champion[key] == value]
    champion = champion.iloc[0]
    champion_run = final[
        (final.model == winner['model']) & (final.strategy == winner['strategy'])
        & (final.k == winner['k']) & (final.weighted == winner['weighted'])
    ].iloc[0]
    winner_training = (f"expanding history ({int(champion_run.train_rows):,} training rows)"
                       if winner['strategy'] == 'expanding'
                       else f"{winner['strategy']}, K={winner['k']:,}")
    population = final.iloc[0]
    missing = inventory[inventory.missing_after_sentinels > 0][["feature", "missing_fraction"]].sort_values("missing_fraction", ascending=False)
    correlations = pd.read_csv(out / "eda/numeric_target_correlations.csv").head(8)
    importances = pd.read_csv(out / "lightgbm_features.csv").sort_values("importance", ascending=False).head(10)
    comparison_lines = []
    for row in paired.itertuples(index=False):
        interpretation = "positive" if row.ci_low > 0 else "negative" if row.ci_high < 0 else "includes zero"
        comparison_lines.append(f"- **{DISPLAY[row.model]}, {row.comparison.replace('_', ' ')}:** {row.difference * 100:+.2f} percentage points; conditional 95% interval [{row.ci_low * 100:+.2f}, {row.ci_high * 100:+.2f}] pp ({interpretation}).")
    weight_table = final[(final.strategy == "expanding")][["model", "weighted", "recall_at_1pct", "precision_at_1pct", "average_precision", "default_precision", "default_recall", "default_f1", "selected_precision", "selected_recall", "selected_f1", "selected_threshold"]]
    baselines = final[(final.strategy == "expanding") & ~final.weighted].sort_values("recall_at_1pct", ascending=False)
    baseline_sentence = "; ".join(f"{DISPLAY[row.model]} {row.recall_at_1pct * 100:.2f}%" for row in baselines.itertuples())
    recency_pairs = paired[paired.comparison == "recent_minus_mean_matched"]
    recency_sentence = "Both final recent-minus-matched intervals include zero; the study does not establish a reliable recency advantage. The intervals still allow modest effects in either direction." if ((recency_pairs.ci_low <= 0) & (recency_pairs.ci_high >= 0)).all() else "Final recency effects and conditional intervals differ by model; see the paired comparisons below."
    weighting = final[final.strategy == "expanding"].pivot(index="model", columns="weighted", values="recall_at_1pct")
    weighting_sentence = "Balanced class weighting reduced final recall at 1% review for all three tested model presets. It improved F1 at the arbitrary 0.5 threshold, but validation-selected thresholds gave stronger F1 for the unweighted versions. This points to threshold policy and score scale as central to interpreting the weighting comparison." if (weighting[True] < weighting[False]).all() else "Class weighting effects vary across model presets; compare equal-capacity ranking separately from default-threshold metrics."
    lines = [
        "# BAF Base v2: EDA and first hypothesis tests",
        "Measured study, 10 October 2026. Codex assisted with design, code, execution and interpretation; group review is required before submission.",
        "## Main result",
        f"The validation-selected candidate was **{DISPLAY[winner['model']]}, {winner_training}, {'balanced' if winner['weighted'] else 'unweighted'}**. On month 7 it captured **{champion.recall * 100:.2f}%** of fraud at 1% review, with **{champion.precision * 100:.2f}% precision** and **AP {champion.ap:.4f}**. "
        f"The holdout contains {int(population.n_eval):,} applications and {int(population.fraud_eval):,} fraud cases; the review quota is {int(population.reviewed_at_1pct):,}. "
        f"It flags {champion.tp:,.0f} frauds and {champion.fp:,.0f} legitimate applications while missing {champion.missed:,.0f} frauds. "
        "Historical-control results are seed means where applicable, so counts may be fractional averages. This is a small synthetic benchmark, not a production performance claim.",
        "## What the evidence supports",
        recency_sentence,
        f"At the same 1% review capacity, the predeclared **unweighted expanding-history** baselines captured: **{baseline_sentence}**. These are descriptive comparisons of fixed presets. The validation-selected candidate remains the weighted LightGBM configuration above; a higher observed holdout score does not retroactively change that selection.",
        weighting_sentence,
        "## Design and what each comparison tests",
        "The [protocol](../../docs/experiment-protocol.md) and [configuration](../../configs/study.json) specify settings fixed before final scoring. Three forward validation folds evaluate months 4, 5 and 6. Recent and historical samples have identical K and exact class counts; historical samples use three seeds and can include recent rows. K is selected from validation recent-strategy recall. Final month 7 is accessed only after a hash-stamped selection freeze. Class weighting and model comparisons use expanding history, keeping that ablation separate from the recency experiment.",
        "Tables express recall, precision, prevalence, missing fractions and AP as fractions (0.20 means 20%), unless a column explicitly says percentage points (`pp`). For expanding-history rows, `k=0` is an internal marker for using all eligible training rows; it does not mean an empty training set.",
        "## EDA: development months 0-6 only",
        f"The development set contains {quality['rows']:,} applications, {quality['literal_null_cells']} literal null cells and {quality['duplicate_rows']} duplicate rows. `device_fraud_count` is constant zero and excluded; `month` is split-only. There are 29 candidate predictors. Encoded missing values require explicit treatment despite the absence of literal nulls.",
        md_table(monthly), "![Development volume and prevalence](monthly.png)",
        "Fraud prevalence changes across development months; detection at a fixed quota must be interpreted alongside precision and prevalence rather than accuracy alone.",
        md_table(missing), "![Missingness and numerical drift](quality_and_drift.png)",
        "The largest numerical distribution changes are in the application-velocity features. The table reports KS distance for numerical variables and total-variation distance for categorical variables, comparing months 0 and 6. These are effect sizes, not proof of changing fraud mechanisms; measures across variable types are not a single common drift scale.",
        md_table(drift.head(12)), "![Selected development distributions](distributions.png)",
        "Development-only associations with fraud are descriptive: numerical values below are Pearson correlations using available values, and categorical plots show group prevalence. These do not establish causation, and associations can change over time. Support counts, numerical-quantile fraud rates and class-conditional monthly means are provided in the accompanying CSVs. They were not used to select predictors or tune models.",
        md_table(correlations), "![Categorical associations with fraud](categorical_fraud_rates.png)",
        "## Theory 1: recent data versus matched historical data",
        "Validation differences below average the historical seeds within each month, then give each forward month equal weight. Positive differences favor recent training. The month-range columns show whether any apparent advantage is stable over time.",
        md_table(recency), "![Forward-validation training size comparisons](validation_training_size.png)",
        "Final comparisons use K chosen from validation, not the holdout. Matched-control uncertainty is conditional on these fitted models and the chosen settings. It does not include all retraining or temporal variation.",
        *comparison_lines, "![Final training-strategy comparison](final_recency.png)",
        md_table(final_summary[["model", "strategy", "k", "weighted", "recall", "precision", "ap", "tp", "fp", "missed", "min_seed_recall", "max_seed_recall"]]),
        "## Theory 2: more data and model family",
        "The validation training-size curves show whether detection improves as K rises under recent and matched-historical retention. Across K, class proportions can differ because controls match each recent-K sample; this is not a pure size-only ablation. The expanding reference answers the practical all-history question but has a different size and composition; it does not isolate recency. Final unweighted expanding baselines compare the three fixed model presets on the same applications. No hyperparameter search was performed, so these results do not establish a universally best model family.",
        "![Final precision-recall and review-capacity curves](operating_curves.png)",
        "The unweighted expanding-history LightGBM model's most-used transformed features are shown below. Importance is split count, not gain, a causal effect or a complete explanation; correlated and high-cardinality predictors complicate interpretation. Logistic coefficients and random-forest importances are also saved as descriptive diagnostics.",
        md_table(importances),
        "## Theory 3: class weighting and threshold policy",
        "Balanced weighting changes the training objective and may change both rankings and probability scale. Compare recall at equal review capacity to assess ranking. Default-threshold results use 0.5; selected-threshold results use thresholds chosen from validation only. Differences at 0.5 alone are not evidence that a model has improved operationally.",
        md_table(weight_table), "![Weighting and model comparisons](weighting_and_models.png)",
        "## Random-split diagnostic",
        "An 80/20 stratified random split of development months only gives the following unweighted results. It uses different training size and evaluation populations from the chronological folds; average precision also depends on population prevalence. A gap cannot be attributed solely to temporal drift, and the holdout month is excluded from this diagnostic.",
        md_table(diagnostic[["model", "train_rows", "n_eval", "recall_at_1pct", "precision_at_1pct", "average_precision", "default_f1"]]),
        "## Verification, uncertainty and next use",
        f"Completed {checks['validation_fits']} validation, {checks['final_fits']} final and {checks['diagnostic_fits']} diagnostic fits. Eight methodological tests verify temporal sampling, exact class/size controls, missing-code handling, training-only preprocessing, rounding/ties, bootstrap equivalence and freeze guards. Full-data audits passed training boundaries, equal evaluation cases, matched class counts, quota/confusion accounting, source/config identity and freeze-before-test ordering. No convergence warnings were found.",
        "The 500-replicate paired percentile intervals resample final test rows under fixed fitted models. They exclude model-selection/retraining and between-month uncertainty and are unadjusted for multiple comparisons. Historical-seed ranges and forward-month variation are reported separately. One variant, three forward folds, one final month, coarse timestamps, synthetic generation and unobserved reporting delays limit the claims. Age and other sensitive attributes remain predictors; no fairness audit or deployment recommendation is implied.",
        "Month 7 has now been consumed. Future tuning on these findings is exploratory and requires new holdout evidence for confirmatory claims. Review model-specific coefficients/importances as descriptive diagnostics; they are not causal explanations or a replacement for a future interpretability/fairness study.",
        "Reproduce using the [README](../../README.md). Run-level metrics, selection, feature diagnostics, audits and all small EDA tables accompany this report; bulk predictions/checkpoints remain in ignored `outputs/`.",
        "Sources: [BAF](https://github.com/feedzai/bank-account-fraud), [datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf), [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). See pinned software versions in `requirements.txt` and acknowledge AI/external software in the final course report.",
    ]
    (report_dir / "findings.md").write_text("\n\n".join(lines) + "\n")
    print("Report saved to", report_dir / "findings.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    generate(parser.parse_args().config)
