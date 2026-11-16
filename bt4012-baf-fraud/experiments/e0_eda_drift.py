"""E0 - Data understanding and drift.

Outputs
  results/e0_variable_summary.csv   tabular summary of every variable (for the report)
  results/e0_monthly.csv            rows and fraud rate per month
  results/e0_covariate_drift.csv    how much each feature's distribution moves over months
  results/e0_concept_drift.csv      how predictive each numeric feature is, month by month
  figures/e0_*.png
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.config import MONTH, TARGET
from src.utils import log, plt, savefig


def _psi(ref, cur, bins=10):
    """Population Stability Index between two samples of one feature.
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift."""
    eps = 1e-6
    if ref.dtype.name == "category" or ref.dtype == object:
        cats = ref.astype(str).unique()
        r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
        c = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    else:
        edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:  # (near) constant or binary feature: compare value shares
            vals = np.unique(ref.dropna())
            r = ref.value_counts(normalize=True, dropna=False).reindex(vals, fill_value=0)
            c = cur.value_counts(normalize=True, dropna=False).reindex(vals, fill_value=0)
        else:
            edges[0], edges[-1] = -np.inf, np.inf
            r = pd.cut(ref, edges).value_counts(normalize=True, sort=False)
            c = pd.cut(cur, edges).value_counts(normalize=True, sort=False)
            # missing values form their own bin
            r = pd.concat([r * ref.notna().mean(), pd.Series([ref.isna().mean()])])
            c = pd.concat([c * cur.notna().mean(), pd.Series([cur.isna().mean()])])
    r, c = np.asarray(r) + eps, np.asarray(c) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def _cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def run(df, nums, cats, S):
    log("E0: variable summary")
    rows = []
    for col in nums + cats:
        s = df[col]
        kind = "categorical" if col in cats else ("binary" if s.dropna().nunique() <= 2 else "numeric")
        row = {"variable": col, "type": kind, "missing_pct": round(100 * s.isna().mean(), 2),
               "n_unique": int(s.nunique())}
        if kind == "categorical":
            row["summary"] = ", ".join(f"{k} {v:.0%}" for k, v in s.value_counts(normalize=True).head(3).items())
        else:
            row["summary"] = f"mean {s.mean():.3g}, median {s.median():.3g}, min {s.min():.3g}, max {s.max():.3g}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(S.results_dir / "e0_variable_summary.csv", index=False)

    log("E0: monthly volume and fraud rate")
    monthly = df.groupby(MONTH)[TARGET].agg(rows="size", frauds="sum", fraud_rate="mean").reset_index()
    monthly.to_csv(S.results_dir / "e0_monthly.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(monthly[MONTH], 100 * monthly["fraud_rate"], color="#3b6ea5")
    ax.set_xlabel("Month"); ax.set_ylabel("Fraud rate (%)")
    ax.set_title(f"Fraud rate by month (overall {100 * df[TARGET].mean():.2f}%)")
    savefig(fig, S.fig_dir / "e0_fraud_rate_by_month.png")

    log("E0: covariate drift (does each feature's distribution change over months?)")
    # With ~1M rows almost every test is 'significant' (Week 2: large samples make tiny
    # differences significant), so we report effect sizes (PSI, Cramer's V) next to p-values.
    ref = df[df[MONTH] == df[MONTH].min()]
    drift = []
    for col in nums + cats:
        psi = {m: _psi(ref[col], g[col]) for m, g in df.groupby(MONTH) if m != ref[MONTH].iloc[0]}
        row = {"variable": col, "max_psi_vs_month0": max(psi.values()),
               "psi_last_month": psi[max(psi)]}
        if col in cats:
            table = pd.crosstab(df[col], df[MONTH])
            row.update(test="chi-square", p_value=stats.chi2_contingency(table)[1],
                       effect_size=_cramers_v(table), effect="Cramer's V")
        else:
            groups = [g[col].dropna().to_numpy() for _, g in df.groupby(MONTH)]
            f, p = stats.f_oneway(*groups) if all(len(g) > 1 for g in groups) else (np.nan, np.nan)
            grand = df[col].mean()
            ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups if len(g))
            ss_t = np.nansum((df[col] - grand) ** 2)
            row.update(test="one-way ANOVA", p_value=p, effect_size=ss_b / ss_t if ss_t else 0.0,
                       effect="eta squared")
        drift.append(row)
    drift = pd.DataFrame(drift).sort_values("max_psi_vs_month0", ascending=False)
    drift.to_csv(S.results_dir / "e0_covariate_drift.csv", index=False)

    top = drift.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top["variable"], top["max_psi_vs_month0"], color="#3b6ea5")
    ax.axvline(0.1, ls=":", lw=1, color="#999", label="0.1 moderate shift")
    ax.axvline(0.25, ls="--", lw=1, color="#666", label="0.25 large shift")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("Largest PSI vs month 0"); ax.set_title("Features whose distribution drifts most")
    savefig(fig, S.fig_dir / "e0_covariate_drift.png")

    log("E0: concept drift (does each feature's link to fraud change over months?)")
    conc = {}
    for col in nums:
        aucs = {}
        for m, g in df.groupby(MONTH):
            ok = g[col].notna()
            if g.loc[ok, TARGET].nunique() == 2:
                a = roc_auc_score(g.loc[ok, TARGET], g.loc[ok, col])
                aucs[m] = max(a, 1 - a)  # direction-free strength
        conc[col] = aucs
    conc = pd.DataFrame(conc).T
    conc["range"] = conc.max(axis=1) - conc.min(axis=1)
    conc = conc.sort_values("range", ascending=False)
    conc.to_csv(S.results_dir / "e0_concept_drift.csv")

    show = conc.drop(columns="range").head(12)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(show.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(show))); ax.set_yticklabels(show.index)
    ax.set_xticks(range(show.shape[1])); ax.set_xticklabels(show.columns)
    ax.set_xlabel("Month"); ax.set_title("Univariate AUC by month (top 12 by change)")
    fig.colorbar(im, ax=ax, label="AUC")
    savefig(fig, S.fig_dir / "e0_concept_drift.png")
    log("E0 done")
