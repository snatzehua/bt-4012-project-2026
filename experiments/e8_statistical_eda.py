"""E8 - Statistical EDA as taught in Lecture 2 (Data Visualization and Analysis).

Every analysis that uses the fraud label is computed on the TRAINING months only, so
nothing here peeks at the validation or test months that are used later.

1. Univariate profile of every raw variable (Lecture 2, slide 12): % missing, quantiles,
   IQR, mean, median, standard deviation, median absolute deviation (MAD, robust),
   skewness and excess kurtosis; cardinality for categorical variables.
2. Outliers (slides 60-63): boxplot rule (outside Q1 - 1.5 IQR / Q3 + 1.5 IQR) and
   z-score rule (|z| > 3), and the fraud rate inside vs outside the outliers.
3. Bivariate tests against the fraud label, test chosen by data type (slide 41):
   numeric X vs binary Y -> Welch t-test (+ Cohen's d effect size);
   categorical/binary X vs Y -> chi-square test of independence (+ Cramer's V);
   every variable -> information gain IG = H(Y) - H(Y|X) in bits (slides 49-50),
   numeric variables binned into deciles so all variables share one scale.
   With ~675k rows almost every p-value is ~0 (slide 76 trap: huge samples make tiny
   differences significant), so the ranking uses effect sizes and IG, not p-values.
4. Redundancy (slide 27): Pearson correlation between numeric variables.
5. Benford's law (slides 64-78): first-digit test (chi-square goodness of fit, df = 8)
   for count-like variables that span at least one order of magnitude, for legitimate
   and fraudulent applications separately. Capped or bucketed variables are excluded
   (slide 85). Benford only says where to look, not what is fraud.

Outputs (results/): e8_univariate.csv, e8_categorical.csv, e8_outliers.csv,
  e8_bivariate_tests.csv, e8_correlation.csv, e8_high_correlation_pairs.csv, e8_benford.csv
Figures (figures/): e8_information_gain.png, e8_boxplots.png, e8_category_fraud_rate.png,
  e8_correlation_heatmap.png, e8_benford.png
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.config import TARGET
from src.data import months
from src.features import ENGINEERED
from src.utils import log, plt, savefig

BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def _entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def information_gain(x, y):
    """IG = H(Y) - H(Y|X) in bits; x must be discrete (missing values form their own level)."""
    x = pd.Series(x).astype(str).to_numpy()
    h_y = _entropy(np.bincount(y) / len(y))
    df = pd.DataFrame({"x": x, "y": y})
    g = df.groupby("x")["y"].agg(["size", "mean"])
    h_y_given_x = sum(r["size"] / len(y) * _entropy([1 - r["mean"], r["mean"]]) for _, r in g.iterrows())
    return h_y - h_y_given_x


def _discretise(s):
    """Deciles for continuous variables; the variable itself if it has few levels."""
    if s.nunique() <= 20:
        return s.astype(str)
    return pd.qcut(s, 10, duplicates="drop").astype(str)


def first_digit(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return (x / 10 ** np.floor(np.log10(x))).astype(int).clip(1, 9)


def benford_test(x):
    d = first_digit(x)
    obs = np.bincount(d, minlength=10)[1:]
    n = obs.sum()
    exp = BENFORD * n
    chi2, p = stats.chisquare(obs, exp)
    # Nigrini's mean absolute deviation of digit shares: an effect size that does not
    # explode with n (< 0.006 close conformity, > 0.015 nonconformity for first digits)
    mad = float(np.mean(np.abs(obs / n - BENFORD)))
    return n, chi2, p, mad, obs / n


def run(df, nums, cats, S):
    tr = months(df, S.train_months)
    y = tr[TARGET].to_numpy().astype(int)
    raw_nums = [c for c in nums if c not in ENGINEERED]
    binary = [c for c in raw_nums if tr[c].dropna().nunique() <= 2]
    numeric = [c for c in raw_nums if c not in binary]
    log(f"E8: {len(numeric)} numeric, {len(binary)} binary, {len(cats)} categorical variables "
        f"(training months {S.train_months}, {len(tr):,} rows)")

    # ---- 1. univariate profile -------------------------------------------------
    uni = []
    for c in numeric + binary:
        s = tr[c].dropna()
        q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
        uni.append({"variable": c, "type": "binary" if c in binary else "numeric",
                    "missing_pct": 100 * tr[c].isna().mean(), "n_unique": int(s.nunique()),
                    "min": s.min(), "q1": q1, "median": q2, "q3": q3, "max": s.max(), "iqr": q3 - q1,
                    "mean": s.mean(), "std": s.std(), "mad": float(np.median(np.abs(s - q2))),
                    "skewness": float(stats.skew(s)), "excess_kurtosis": float(stats.kurtosis(s)),
                    "mean_minus_median": s.mean() - q2})
    uni = pd.DataFrame(uni)
    uni.to_csv(S.results_dir / "e8_univariate.csv", index=False)

    cat_rows = []
    for c in cats:
        vc = tr[c].astype(str).value_counts()
        rate = tr.groupby(tr[c].astype(str), observed=True)[TARGET].mean()
        for level, n in vc.items():
            cat_rows.append({"variable": c, "cardinality": len(vc), "level": level, "count": int(n),
                             "share": n / len(tr), "fraud_rate": rate[level]})
    pd.DataFrame(cat_rows).to_csv(S.results_dir / "e8_categorical.csv", index=False)

    # ---- 2. outliers -----------------------------------------------------------
    out = []
    base = y.mean()
    for c in numeric:
        s = tr[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        box = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).to_numpy()
        z = ((s - s.mean()) / s.std()).abs().to_numpy() > 3
        row = {"variable": c, "boxplot_outlier_pct": 100 * box.mean(), "z3_outlier_pct": 100 * z.mean(),
               "fraud_rate_all": base}
        for name, m in [("boxplot", box), ("z3", z)]:
            row[f"fraud_rate_{name}_outliers"] = y[m].mean() if m.any() else np.nan
            row[f"fraud_lift_{name}_outliers"] = (y[m].mean() / base) if m.any() else np.nan
        out.append(row)
    pd.DataFrame(out).sort_values("fraud_lift_boxplot_outliers", ascending=False).to_csv(
        S.results_dir / "e8_outliers.csv", index=False)

    # ---- 3. bivariate tests vs fraud ---------------------------------------------
    tests = []
    for c in numeric:
        a, b = tr.loc[y == 1, c].dropna(), tr.loc[y == 0, c].dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt((a.var() + b.var()) / 2)
        tests.append({"variable": c, "type": "numeric", "test": "Welch t-test", "statistic": t, "p_value": p,
                      "effect_size": (a.mean() - b.mean()) / pooled if pooled else 0.0, "effect": "Cohen's d",
                      "mean_fraud": a.mean(), "mean_legit": b.mean(),
                      "median_fraud": a.median(), "median_legit": b.median(),
                      "information_gain_bits": information_gain(_discretise(tr[c]), y)})
    for c in binary + cats:
        table = pd.crosstab(tr[c].astype(str), y)
        chi2, p, dof, _ = stats.chi2_contingency(table)
        k = min(table.shape) - 1
        tests.append({"variable": c, "type": "binary" if c in binary else "categorical",
                      "test": "chi-square independence", "statistic": chi2, "p_value": p, "dof": dof,
                      "effect_size": float(np.sqrt(chi2 / (len(tr) * k))) if k else 0.0, "effect": "Cramer's V",
                      "information_gain_bits": information_gain(tr[c], y)})
    tests = pd.DataFrame(tests).sort_values("information_gain_bits", ascending=False)
    tests["reject_h0_at_5pct"] = tests["p_value"] < 0.05
    tests.to_csv(S.results_dir / "e8_bivariate_tests.csv", index=False)
    log(f"E8: H0 'no association with fraud' rejected for {int(tests.reject_h0_at_5pct.sum())}/{len(tests)} "
        f"variables; top IG: {', '.join(tests.variable.head(5))}")

    top = tests.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = {"numeric": "#3b6ea5", "binary": "#9bb8d6", "categorical": "#c0392b"}
    ax.barh(top["variable"], 1000 * top["information_gain_bits"], color=[colors[t] for t in top["type"]])
    for t, col in colors.items():
        ax.barh([], [], color=col, label=t)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("Information gain about fraud (millibits)")
    ax.set_title("Which variables tell us most about fraud? (training months)")
    savefig(fig, S.fig_dir / "e8_information_gain.png")

    show = [c for c in tests.variable if c in numeric][:6]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.4))
    for ax, c in zip(axes.flat, show):
        data = [tr.loc[y == 0, c].dropna(), tr.loc[y == 1, c].dropna()]
        ax.boxplot(data, tick_labels=["legit", "fraud"], showfliers=False, widths=0.6)
        ax.set_title(c, fontsize=9)
    fig.suptitle("Top numeric variables by information gain: legit vs fraud (outliers hidden)", fontsize=10)
    savefig(fig, S.fig_dir / "e8_boxplots.png")

    cat_show = [c for c in tests.variable if c in cats][:4]
    fig, axes = plt.subplots(1, len(cat_show), figsize=(3.2 * len(cat_show), 3.2), squeeze=False)
    for ax, c in zip(axes[0], cat_show):
        rate = tr.groupby(tr[c].astype(str), observed=True)[TARGET].mean().sort_values()
        ax.bar(rate.index, 100 * rate.values, color="#3b6ea5")
        ax.axhline(100 * base, ls="--", lw=1, color="#999")
        ax.set_title(c, fontsize=9); ax.tick_params(axis="x", rotation=60, labelsize=7)
    axes[0][0].set_ylabel("Fraud rate (%)")
    fig.suptitle("Fraud rate by category (dashed = overall rate)", fontsize=10)
    savefig(fig, S.fig_dir / "e8_category_fraud_rate.png")

    # ---- 4. redundancy: Pearson correlation ----------------------------------------
    corr = tr[numeric + binary].corr(method="pearson")
    corr.to_csv(S.results_dir / "e8_correlation.csv")
    pairs = [{"var_a": a, "var_b": b, "pearson_r": corr.loc[a, b]}
             for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) >= 0.5]
    pd.DataFrame(pairs, columns=["var_a", "var_b", "pearson_r"]).sort_values(
        "pearson_r", key=np.abs, ascending=False).to_csv(S.results_dir / "e8_high_correlation_pairs.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation between numeric variables (training months)")
    savefig(fig, S.fig_dir / "e8_correlation_heatmap.png")

    # ---- 5. Benford's law -----------------------------------------------------------
    ben, shares = [], {}
    for c in numeric:
        s = tr[c].dropna()
        s = s[s > 0]
        if len(s) < 1000:
            continue
        lo, hi = s.quantile([0.01, 0.99])
        spans = lo > 0 and hi / lo >= 10
        capped = s.nunique() < 50  # few distinct values = bucketed / capped (slide 85)
        row = {"variable": c, "p01": lo, "p99": hi, "orders_of_magnitude_p01_p99": np.log10(hi / lo) if lo > 0 else np.nan,
               "n_distinct": int(s.nunique()), "suitable": bool(spans and not capped)}
        if row["suitable"]:
            for grp, mask in [("legit", y == 0), ("fraud", y == 1)]:
                n, chi2, p, mad, share = benford_test(tr.loc[mask, c])
                row.update({f"{grp}_n": n, f"{grp}_chi2": chi2, f"{grp}_p_value": p, f"{grp}_mad": mad})
                shares[(c, grp)] = share
        ben.append(row)
    ben = pd.DataFrame(ben)
    ben.to_csv(S.results_dir / "e8_benford.csv", index=False)
    suit = ben[ben.suitable].sort_values("legit_mad")
    log(f"E8: Benford-suitable variables: {', '.join(suit.variable) or 'none'}")
    if len(suit):
        show = list(suit.variable[:3])
        fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 3.2), squeeze=False)
        d = np.arange(1, 10)
        for ax, c in zip(axes[0], show):
            ax.bar(d - 0.2, 100 * shares[(c, "legit")], 0.4, color="#9bb8d6", label="legit")
            ax.bar(d + 0.2, 100 * shares[(c, "fraud")], 0.4, color="#c0392b", label="fraud")
            ax.plot(d, 100 * BENFORD, "k.-", lw=1, label="Benford")
            r = suit.set_index("variable").loc[c]
            ax.set_title(f"{c}\nMAD legit {r['legit_mad']:.3f}, fraud {r['fraud_mad']:.3f}", fontsize=8)
            ax.set_xticks(d); ax.set_xlabel("First digit")
        axes[0][0].set_ylabel("Share (%)"); axes[0][0].legend(fontsize=7)
        savefig(fig, S.fig_dir / "e8_benford.png")
    log("E8 done")
