"""E9 - Regularisation and overfitting for logistic regression (Lecture 3, slides 50-70).

Lecture 3 adds a complexity penalty to the loss:
    L2 (ridge):  loss + lambda * sum(w^2)   shrinks all weights, rarely to exactly 0
    L1 (lasso):  loss + lambda * sum(|w|)   sets many weights to exactly 0 (feature selection)
scikit-learn writes the strength as C = 1 / lambda (small C = strong penalty).

For each penalty and a grid of C we record training and validation PR-AUC (a large gap
= overfitting; both low = underfitting, Lecture 3 slide 57) and the number of non-zero
weights. Training PR-AUC is measured on ALL training-month applications (not only the
undersampled fitting rows), so it has the same kind of fraud rate as validation. The C with the best VALIDATION PR-AUC is then scored once on the test months.

To keep the many fits fast, every fit uses the undersampled training months (all frauds
plus `undersample_ratio` legitimate applications per fraud); E4 shows undersampling does
not hurt logistic regression's ranking. Preprocessing is the model's own pipeline, fitted
on those training rows only.

Outputs: results/e9_regularisation_path.csv, results/e9_l1_coefficients.csv,
         results/e9_selected.csv, figures/e9_regularisation.png, figures/e9_lasso_path.png
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.config import TARGET
from src.data import months, undersample, xy
from src.metrics import ranking_metrics
from src.models import make_model
from src.utils import log, plt, savefig

# scikit-learn 1.8 deprecates `penalty` in favour of `l1_ratio`; keep `penalty` so the code
# also runs on 1.7, and hide the deprecation notice
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*", category=UserWarning)

C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0]


def run(df, nums, cats, S):
    tr_full, va, te = months(df, S.train_months), months(df, S.val_months), months(df, S.test_months)
    tr = undersample(tr_full, S.undersample_ratio, seed=0)
    y_tr, y_va, y_te = tr[TARGET].to_numpy(), va[TARGET].to_numpy(), te[TARGET].to_numpy()

    # fit the preprocessing once (on the training rows); only the classifier changes below
    prep = make_model("logreg", nums, cats, S).named_steps["prep"]
    X_tr = prep.fit_transform(xy(tr, nums, cats)[0])
    X_trf, y_trf = prep.transform(xy(tr_full, nums, cats)[0]), tr_full[TARGET].to_numpy()
    X_va = prep.transform(xy(va, nums, cats)[0])
    X_te = prep.transform(xy(te, nums, cats)[0])
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    log(f"E9: {X_tr.shape[0]:,} undersampled training rows, {X_tr.shape[1]} model inputs")

    rows, coefs = [], []
    for penalty in ["l2", "l1"]:
        for C in C_GRID:
            clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", max_iter=2000, random_state=0)
            clf.fit(X_tr, y_tr)
            s_tr, s_va = clf.decision_function(X_trf), clf.decision_function(X_va)
            w = clf.coef_[0]
            rows.append({"penalty": penalty, "C": C, "lambda": 1 / C,
                         "train_pr_auc": average_precision_score(y_trf, s_tr),
                         "val_pr_auc": average_precision_score(y_va, s_va),
                         "nonzero_weights": int(np.sum(np.abs(w) > 1e-8)), "n_weights": len(w),
                         "sum_abs_weights": float(np.abs(w).sum()), "sum_sq_weights": float((w ** 2).sum())})
            if penalty == "l1":
                coefs.append(pd.Series(w, index=names, name=C))
            log(f"  E9 {penalty} C={C:g}: train PR-AUC {rows[-1]['train_pr_auc']:.4f}, "
                f"val {rows[-1]['val_pr_auc']:.4f}, non-zero {rows[-1]['nonzero_weights']}/{len(w)}")
    res = pd.DataFrame(rows)
    res["train_minus_val"] = res["train_pr_auc"] - res["val_pr_auc"]
    res.to_csv(S.results_dir / "e9_regularisation_path.csv", index=False)
    path = pd.concat(coefs, axis=1)
    path.to_csv(S.results_dir / "e9_l1_coefficients.csv", index_label="feature")

    # ---- choose on validation, score once on test -------------------------------
    sel = []
    for penalty in ["l2", "l1"]:
        r = res[res.penalty == penalty].sort_values("val_pr_auc", ascending=False).iloc[0]
        clf = LogisticRegression(penalty=penalty, C=r["C"], solver="liblinear", max_iter=2000, random_state=0).fit(X_tr, y_tr)
        s_te = clf.decision_function(X_te)
        sel.append({"penalty": penalty, "C": r["C"], "val_pr_auc": r["val_pr_auc"],
                    "nonzero_weights": r["nonzero_weights"], **ranking_metrics(y_te, s_te, S.fpr_target)})
    pd.DataFrame(sel).to_csv(S.results_dir / "e9_selected.csv", index=False)

    # ---- figures ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for penalty, color in [("l2", "#3b6ea5"), ("l1", "#c0392b")]:
        d = res[res.penalty == penalty]
        axes[0].plot(d["C"], d["val_pr_auc"], "o-", color=color, label=f"{penalty.upper()} validation")
        axes[0].plot(d["C"], d["train_pr_auc"], "o--", color=color, alpha=0.5,
                     label=f"{penalty.upper()} training months 0-4")
        axes[1].plot(d["C"], d["nonzero_weights"], "o-", color=color, label=penalty.upper())
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("C = 1 / lambda  (left = stronger penalty)")
    axes[0].set_ylabel("PR-AUC"); axes[0].legend(fontsize=7)
    axes[0].set_title("Under- vs good fit: training vs validation PR-AUC", fontsize=9)
    axes[1].set_ylabel("Non-zero weights"); axes[1].legend(fontsize=8)
    axes[1].set_title("L1 sets weights to exactly zero; L2 does not", fontsize=9)
    savefig(fig, S.fig_dir / "e9_regularisation.png")

    big = path.abs().max(axis=1).sort_values(ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for f in big:
        ax.plot(path.columns, path.loc[f], marker=".", label=f)
    ax.set_xscale("log"); ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("C = 1 / lambda"); ax.set_ylabel("Coefficient")
    ax.set_title("Lasso (L1) path: the 10 largest coefficients", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    savefig(fig, S.fig_dir / "e9_lasso_path.png")
    log("E9 done")
