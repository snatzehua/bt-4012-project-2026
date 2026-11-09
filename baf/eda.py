"""Development-only EDA; month 7 is held out until final model scoring."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .common import CATEGORICAL, EXCLUDED, features, load_data, read_config, write_json


def run(config_path):
    config = read_config(config_path)
    raw = load_data(config)
    dev = raw.loc[raw.month < config["test_month"]].copy()
    clean = features(dev)
    out = Path(config["output_dir"]) / "eda"
    out.mkdir(parents=True, exist_ok=True)
    monthly = dev.groupby("month").fraud_bool.agg(rows="size", fraud="sum", prevalence="mean")
    monthly.to_csv(out / "monthly.csv")
    inventory = []
    for col in dev:
        values = clean[col] if col in clean else dev[col]
        inventory.append({"feature": col, "dtype": str(dev[col].dtype), "distinct": int(dev[col].nunique()),
                          "literal_nulls": int(dev[col].isna().sum()), "missing_after_sentinels": int(values.isna().sum()),
                          "missing_fraction": float(values.isna().mean()),
                          "role": "target" if col == "fraud_bool" else "split_only" if col == "month" else "excluded_constant" if col in EXCLUDED else "predictor"})
    pd.DataFrame(inventory).to_csv(out / "feature_inventory.csv", index=False)
    numeric = [col for col in clean if col not in CATEGORICAL]
    clean[numeric].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.to_csv(out / "numeric_summary.csv")
    categories = []
    for month, group in dev.groupby("month"):
        for col in CATEGORICAL:
            for value, n in group[col].value_counts(dropna=False).items():
                categories.append({"month": int(month), "feature": col, "value": str(value), "count": int(n), "fraction": float(n / len(group))})
    pd.DataFrame(categories).to_csv(out / "categorical_composition.csv", index=False)
    # Descriptive development associations; these are not used for feature selection.
    associations = []
    for col in clean:
        if col in CATEGORICAL:
            groups = clean[col].astype(str)
        elif clean[col].nunique(dropna=True) <= 12:
            groups = clean[col].astype(str).where(clean[col].notna(), "Missing")
        else:
            groups = pd.qcut(clean[col], q=5, duplicates="drop").astype(str)
            groups = groups.where(clean[col].notna(), "Missing")
        grouped = pd.DataFrame({"group": groups, "fraud": dev.fraud_bool}).groupby("group").fraud.agg(rows="size", frauds="sum", prevalence="mean")
        for label, row in grouped.iterrows():
            associations.append({"feature": col, "group": str(label), "rows": int(row.rows), "frauds": int(row.frauds), "prevalence": float(row.prevalence)})
    pd.DataFrame(associations).to_csv(out / "feature_fraud_rates.csv", index=False)
    clean[numeric].corrwith(dev.fraud_bool).rename_axis("feature").rename("pearson_correlation_with_fraud").sort_values(key=lambda s: s.abs(), ascending=False).to_csv(out / "numeric_target_correlations.csv")
    conditional = clean.assign(month=dev.month, fraud_bool=dev.fraud_bool).groupby(["month", "fraud_bool"])[numeric].mean()
    conditional.to_csv(out / "class_conditional_monthly_means.csv")
    drift = []
    for col in numeric:
        a = clean.loc[dev.month == 0, col].dropna()
        b = clean.loc[dev.month == 6, col].dropna()
        drift.append({"feature": col, "measure": "ks_statistic", "effect": float(ks_2samp(a, b).statistic),
                      "missing_change": float(clean.loc[dev.month == 6, col].isna().mean() - clean.loc[dev.month == 0, col].isna().mean())})
    for col in CATEGORICAL:
        a = clean.loc[dev.month == 0, col].value_counts(normalize=True)
        b = clean.loc[dev.month == 6, col].value_counts(normalize=True)
        categories_union = a.index.union(b.index)
        effect = (a.reindex(categories_union, fill_value=0) - b.reindex(categories_union, fill_value=0)).abs().sum() / 2
        drift.append({"feature": col, "measure": "total_variation", "effect": float(effect), "missing_change": 0.0})
    drift = pd.DataFrame(drift).sort_values("effect", ascending=False)
    drift.to_csv(out / "drift_month0_vs_month6.csv", index=False)
    write_json(out / "quality.json", {"rows": len(dev), "columns": len(dev.columns),
               "literal_null_cells": int(dev.isna().sum().sum()), "duplicate_rows": int(dev.duplicated().sum()),
               "constant_columns": dev.nunique().loc[lambda x: x == 1].index.tolist(),
               "test_month_excluded": config["test_month"], "predictor_count": len(clean.columns)})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(monthly.index, monthly.rows / 1000, color="#366589")
    axes[0].set(xlabel="Development month", ylabel="Applications (thousands)", title="Monthly application volume")
    axes[1].plot(monthly.index, 100 * monthly.prevalence, marker="o", color="#bd5c35")
    axes[1].set(xlabel="Development month", ylabel="Fraud prevalence (%)", title="Class balance changes over time")
    fig.tight_layout()
    fig.savefig(out / "monthly.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    missing = pd.DataFrame(inventory).set_index("feature").missing_fraction.loc[lambda s: s > 0].sort_values()
    axes[0].barh(missing.index, missing.values * 100, color="#366589")
    axes[0].set(xlabel="Encoded missing values (%)", title="Missingness after documented sentinel handling")
    numeric_drift = drift.loc[drift.measure == "ks_statistic"].head(8).iloc[::-1]
    axes[1].barh(numeric_drift.feature, numeric_drift.effect, color="#bd5c35")
    axes[1].set(xlabel="KS distance (0-1)", title="Numerical distribution change: month 0 vs 6")
    fig.tight_layout()
    fig.savefig(out / "quality_and_drift.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["credit_risk_score", "customer_age", "income"]):
        for month, color in [(0, "#366589"), (6, "#bd5c35")]:
            values = clean.loc[dev.month == month, col].dropna()
            ax.hist(values, bins=25, density=True, histtype="step", label=f"Month {month}", color=color)
        ax.set(title=col, ylabel="Density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["payment_type", "employment_status", "housing_status"]):
        grouped = dev.groupby(col).fraud_bool.agg(["size", "mean"]).sort_index()
        ax.bar(grouped.index.astype(str), grouped["mean"] * 100, color="#366589")
        ax.axhline(dev.fraud_bool.mean() * 100, color="#bd5c35", linestyle="--", label="Overall prevalence")
        ax.set(title=col, ylabel="Fraud prevalence (%)")
        ax.tick_params(axis="x", rotation=40)
        ax.legend(fontsize=7)
    fig.suptitle("Development-only target associations; category support counts in CSV")
    fig.tight_layout()
    fig.savefig(out / "categorical_fraud_rates.png", dpi=170)
    plt.close(fig)
    print(monthly.to_string(), flush=True)
    print("EDA saved to", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    run(parser.parse_args().config)
