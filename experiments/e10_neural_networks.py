"""E10 - Deep learning for fraud detection (Lecture 4, slides 73-75).

Two neural approaches from the lecture, compared with the E1 models on the same
out-of-time protocol (train months 0-4, choose on month 5, test months 6-7):

A. Supervised feedforward classifier (MLP): ReLU hidden layers, sigmoid output, log
   loss, mini-batch training, dropout. Width, depth and dropout are chosen on the
   validation month (Lecture 4 slides 35-38 show that the widest / deepest network
   is not the one that generalises best), with early stopping on validation PR-AUC.
   It trains on the undersampled training months (all frauds + `undersample_ratio`
   legitimate applications per fraud), the imbalance treatment E4 finds as good as
   any other and much faster.

B. Unsupervised autoencoder: trained to reconstruct LEGITIMATE applications only (no
   fraud labels used in training). Reconstruction error is the anomaly score. The
   alert threshold is set from the error distribution of legitimate applications in
   the validation month (95th percentile = 5% false-positive rate), then applied to
   the test months. High error means "unusual", not proven fraud (Lecture 4 trap).

C. Hybrid: the autoencoder's error is added as one extra input to LightGBM, to see if
   the unsupervised signal adds anything to the supervised model.

Inputs for both networks: numeric columns median-imputed (+ missing flags) and
standardised, categories one-hot encoded; all fitted on training rows only.

Outputs: results/e10_mlp_tuning.csv, results/e10_summary.csv, results/e10_mlp_history.csv,
         figures/e10_mlp_tuning.png, figures/e10_roc.png, figures/e10_autoencoder_errors.png
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET
from src.data import months, undersample, xy
from src.metrics import at_threshold, ranking_metrics, threshold_at_fpr
from src.models import fit_predict
from src.nn import MLP
from src.utils import load_json, log, plt, savefig, strategy_for

MLP_GRID = [  # (hidden layer sizes, dropout rate)
    ((16,), 0.0), ((64,), 0.0), ((256,), 0.0),
    ((64, 32), 0.0), ((64, 32), 0.2), ((128, 64, 32), 0.2),
]
AE_HIDDEN = (32, 8, 32)   # 8-unit bottleneck
AE_TRAIN_ROWS = 300_000


def dense_preprocessor(nums, cats):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ], sparse_threshold=0)


def _clip(X):
    # heavy tails would let a few extreme rows dominate the squared-error loss
    return np.clip(X, -5, 5).astype(np.float32)


def run(df, nums, cats, S):
    best = load_json(S.results_dir / "e1_best_params.json")
    tr_full, va, te = months(df, S.train_months), months(df, S.val_months), months(df, S.test_months)
    y_va, y_te = va[TARGET].to_numpy(), te[TARGET].to_numpy()
    epochs = 8 if S.seeds == [0] and S.sample_frac < 1 else 30

    prep = dense_preprocessor(nums, cats).fit(xy(tr_full, nums, cats)[0])
    X_va, X_te = _clip(prep.transform(xy(va, nums, cats)[0])), _clip(prep.transform(xy(te, nums, cats)[0]))
    log(f"E10: {X_va.shape[1]} network inputs")

    # ---- A. MLP: choose width / depth / dropout on validation -------------------
    tr = undersample(tr_full, S.undersample_ratio, seed=0)
    X_tr, y_tr = _clip(prep.transform(xy(tr, nums, cats)[0])), tr[TARGET].to_numpy()
    val_ap = lambda m: average_precision_score(y_va, m.predict_proba(X_va)[:, 1])  # noqa: E731
    tuning = []
    for hidden, dropout in MLP_GRID:
        m = MLP(hidden=hidden, dropout=dropout, lr=1e-3, batch_size=256, epochs=epochs, l2=1e-5,
                patience=4, seed=0).fit(X_tr, y_tr, val_score=val_ap)
        tuning.append({"hidden_layers": str(hidden), "n_hidden_layers": len(hidden), "dropout": dropout,
                       "n_params": int(sum(w.size for w in m.W) + sum(b.size for b in m.b)),
                       "best_epoch": m.best_epoch,
                       "train_pr_auc_undersampled": average_precision_score(y_tr, m.predict_proba(X_tr)[:, 1]),
                       "val_pr_auc": val_ap(m)})
        log(f"  E10 MLP {hidden} dropout {dropout}: val PR-AUC {tuning[-1]['val_pr_auc']:.4f} "
            f"(best epoch {m.best_epoch})")
    tuning = pd.DataFrame(tuning)
    tuning.to_csv(S.results_dir / "e10_mlp_tuning.csv", index=False)
    top = tuning.sort_values("val_pr_auc", ascending=False).iloc[0]
    hidden, dropout = eval(top["hidden_layers"]), float(top["dropout"])  # noqa: S307 (own string)
    log(f"E10: chosen MLP {hidden}, dropout {dropout}")

    rows, curves = [], {}
    for seed in S.seeds:
        tr_s = undersample(tr_full, S.undersample_ratio, seed=seed)
        Xs, ys = _clip(prep.transform(xy(tr_s, nums, cats)[0])), tr_s[TARGET].to_numpy()
        m = MLP(hidden=hidden, dropout=dropout, lr=1e-3, batch_size=256, epochs=epochs, l2=1e-5,
                patience=4, seed=seed).fit(Xs, ys, val_score=val_ap)
        if seed == S.seeds[0]:
            pd.DataFrame(m.history).to_csv(S.results_dir / "e10_mlp_history.csv", index=False)
        s_va, s_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
        at = at_threshold(y_te, s_te, threshold_at_fpr(y_va, s_va, S.fpr_target))
        rows.append({"model": f"mlp {hidden} dropout {dropout}", "family": "supervised neural network",
                     "seed": seed, **ranking_metrics(y_te, s_te, S.fpr_target),
                     "valthr_recall": at["recall"], "valthr_fpr": at["fpr"], "valthr_precision": at["precision"]})
        curves.setdefault("MLP", (y_te, s_te))

    # ---- B. autoencoder on legitimate applications only ----------------------------
    legit = tr_full[tr_full[TARGET] == 0]
    legit = legit.sample(n=min(AE_TRAIN_ROWS, len(legit)), random_state=0)
    X_leg = _clip(prep.transform(xy(legit, nums, cats)[0]))
    va_legit = X_va[y_va == 0]
    ae = MLP(hidden=AE_HIDDEN, output="identity", dropout=0.0, lr=1e-3, batch_size=512, epochs=epochs,
             patience=3, seed=0).fit(X_leg, X_val=va_legit[:50_000])
    e_va, e_te = ae.reconstruction_error(X_va), ae.reconstruction_error(X_te)
    thr = float(np.quantile(e_va[y_va == 0], 1 - S.fpr_target))  # from the NORMAL error distribution
    at = at_threshold(y_te, e_te, thr)
    rows.append({"model": f"autoencoder {AE_HIDDEN}", "family": "unsupervised (legit-only) neural network",
                 "seed": 0, **ranking_metrics(y_te, e_te, S.fpr_target),
                 "valthr_recall": at["recall"], "valthr_fpr": at["fpr"], "valthr_precision": at["precision"]})
    curves["Autoencoder"] = (y_te, e_te)
    log(f"E10 autoencoder: test PR-AUC {rows[-1]['pr_auc']:.4f}, recall@5%FPR {rows[-1]['recall_at_5fpr']:.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bins = np.linspace(0, np.quantile(e_te, 0.995), 80)
    ax.hist(e_te[y_te == 0], bins=bins, density=True, alpha=0.6, color="#9bb8d6", label="legit (test)")
    ax.hist(e_te[y_te == 1], bins=bins, density=True, alpha=0.6, color="#c0392b", label="fraud (test)")
    ax.axvline(thr, color="black", ls="--", lw=1, label="threshold (95th pct of legit, month 5)")
    ax.set_xlabel("Reconstruction error"); ax.set_ylabel("Density"); ax.legend(fontsize=7)
    ax.set_title("Autoencoder trained on legitimate applications only", fontsize=10)
    savefig(fig, S.fig_dir / "e10_autoencoder_errors.png")

    # ---- C. hybrid: autoencoder error as an extra LightGBM input --------------------
    e_tr = ae.reconstruction_error(_clip(prep.transform(xy(tr_full, nums, cats)[0])))
    tr_h, va_h, te_h = tr_full.assign(ae_error=e_tr), va.assign(ae_error=e_va), te.assign(ae_error=e_te)
    for name, cols in [("lightgbm (E1 inputs)", nums), ("lightgbm + autoencoder error", nums + ["ae_error"])]:
        _, (s_va, s_te), _ = fit_predict("lightgbm", tr_h, [va_h, te_h], cols, cats, S,
                                         strategy_for(best, "lightgbm"), seed=0, params=best["params"]["lightgbm"])
        at = at_threshold(y_te, s_te, threshold_at_fpr(y_va, s_va, S.fpr_target))
        rows.append({"model": name, "family": "gradient-boosted trees", "seed": 0,
                     **ranking_metrics(y_te, s_te, S.fpr_target),
                     "valthr_recall": at["recall"], "valthr_fpr": at["fpr"], "valthr_precision": at["precision"]})
        if name.startswith("lightgbm (E1"):
            curves["LightGBM"] = (y_te, s_te)

    # logistic regression scores from E1 for the ROC comparison
    try:
        sc = pd.read_csv(S.results_dir / "e1_scores.csv.gz")
        d = sc[(sc.model == "logreg") & (sc.split == "test")]
        curves["Logistic regression"] = (d[TARGET].to_numpy(), d["score"].to_numpy())
    except FileNotFoundError:
        pass

    res = pd.DataFrame(rows)
    res.to_csv(S.results_dir / "e10_results.csv", index=False)
    summ = res.groupby(["model", "family"], sort=False).agg(
        n_seeds=("seed", "size"), pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        roc_auc=("roc_auc", "mean"), recall_at_5fpr=("recall_at_5fpr", "mean"),
        recall_at_5fpr_std=("recall_at_5fpr", "std"), valthr_recall=("valthr_recall", "mean"),
        valthr_fpr=("valthr_fpr", "mean")).reset_index()
    summ.to_csv(S.results_dir / "e10_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, (yy, ss) in curves.items():
        fpr, tpr, _ = roc_curve(yy, ss)
        ax.plot(fpr, tpr, lw=1.2, label=f"{name} (AUC {ranking_metrics(yy, ss)['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="#999", label="random (AUC 0.5)")
    ax.axvline(S.fpr_target, ls="--", lw=0.8, color="#666")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC on test months", fontsize=10); ax.legend(fontsize=7, loc="lower right")
    savefig(fig, S.fig_dir / "e10_roc.png")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    lbl = [f"{h}\ndrop {d:g}" for h, d in zip(tuning["hidden_layers"], tuning["dropout"])]
    ax.bar(range(len(tuning)), tuning["val_pr_auc"], color="#3b6ea5")
    ax.set_xticks(range(len(tuning))); ax.set_xticklabels(lbl, fontsize=7)
    ax.set_ylim(tuning["val_pr_auc"].min() * 0.9, tuning["val_pr_auc"].max() * 1.03)
    ax.set_ylabel("Validation PR-AUC"); ax.set_title("MLP width / depth / dropout (month 5)", fontsize=10)
    savefig(fig, S.fig_dir / "e10_mlp_tuning.png")
    log("E10 done")
