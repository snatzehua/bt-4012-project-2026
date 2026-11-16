"""E6 - What drives the predictions, and where do the models fail?

Outputs
  results/e6_logreg_coefficients.csv   standardised coefficients (direction + size)
  results/e6_lightgbm_importance.csv   gain importance
  results/e6_shap_importance.csv       mean |SHAP| on a test sample (if `shap` is installed)
  results/e6_error_profiles.csv        median feature values for TP / FP / FN / TN
  results/e6_errors_by_month.csv
  figures/e6_*.png
"""
import joblib
import numpy as np
import pandas as pd

from src.config import MONTH, TARGET
from src.data import months, xy
from src.metrics import threshold_at_fpr
from src.utils import load_json, log, plt, savefig


def _barh(series, title, xlabel, path, color="#3b6ea5"):
    s = series[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(s) + 1))
    ax.barh(s.index, s.values, color=color)
    ax.set_title(title); ax.set_xlabel(xlabel)
    savefig(fig, path)


def run(df, nums, cats, S):
    mdir = S.results_dir / "models"
    te = months(df, S.test_months)
    X_te, y_te = xy(te, nums, cats)

    # ---- logistic regression coefficients ------------------------------------
    lr = joblib.load(mdir / "logreg.joblib")
    names = lr.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=[n.split("__", 1)[1] for n in names])
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(15)
    top.to_csv(S.results_dir / "e6_logreg_coefficients.csv", header=["coefficient"])
    _barh(top, "Logistic regression: largest standardised coefficients",
          "Coefficient (+ raises fraud odds)", S.fig_dir / "e6_logreg_coefficients.png")

    # ---- LightGBM importance --------------------------------------------------
    gbm = joblib.load(mdir / "lightgbm.joblib")
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=gbm.booster_.feature_name())
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.to_csv(S.results_dir / "e6_lightgbm_importance.csv", header=["gain_share"])
    _barh(imp.head(15), "LightGBM: share of total gain", "Gain share", S.fig_dir / "e6_lightgbm_importance.png")

    try:
        import shap
        sample = X_te.sample(n=min(3000, len(X_te)), random_state=0)
        sv = shap.TreeExplainer(gbm).shap_values(sample)
        sv = sv[1] if isinstance(sv, list) else sv
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns).sort_values(ascending=False)
        mean_abs.to_csv(S.results_dir / "e6_shap_importance.csv", header=["mean_abs_shap"])
        plt.figure()
        shap.summary_plot(sv, sample, max_display=15, show=False)
        savefig(plt.gcf(), S.fig_dir / "e6_shap_summary.png")
        log("E6: SHAP summary saved")
    except Exception as exc:  # shap is optional
        log(f"E6: SHAP skipped ({exc})")

    # ---- error analysis at the deployable threshold ---------------------------
    va = months(df, S.val_months)
    s_va = gbm.predict_proba(xy(va, nums, cats)[0])[:, 1]
    thr = threshold_at_fpr(va[TARGET].to_numpy(), s_va, S.fpr_target)
    s_te = gbm.predict_proba(X_te)[:, 1]
    pred = s_te > thr
    outcome = np.select([pred & (y_te == 1), pred & (y_te == 0), ~pred & (y_te == 1)], ["TP", "FP", "FN"], "TN")
    te = te.assign(outcome=outcome)
    key = list(imp.index[:10])
    key_num = [c for c in key if c in nums]
    prof = te.groupby("outcome")[key_num].median().T
    prof.loc["count"] = te["outcome"].value_counts().reindex(prof.columns).fillna(0)
    prof.to_csv(S.results_dir / "e6_error_profiles.csv")
    by_month = te.groupby([MONTH, "outcome"]).size().unstack(fill_value=0)
    by_month["miss_rate"] = by_month.get("FN", 0) / (by_month.get("FN", 0) + by_month.get("TP", 0))
    by_month.to_csv(S.results_dir / "e6_errors_by_month.csv")
    log("E6 done")
