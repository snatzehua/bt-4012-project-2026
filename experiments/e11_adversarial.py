"""E11 - Adversarial evasion and defences (Lecture 7, slides 46-87, applied to tabular data).

Lecture 7 shows that small, targeted input changes can fool a model. A fraudster
filling in an account application cannot change platform-wide velocity counts or the
bank's credit score, but can cheaply change what they type or how they connect:
the e-mail they use, phone numbers, declared income / housing / employment, the
device's reported OS, the channel, the session, and so on (CONTROLLABLE below).

Attack (non-targeted "dodging"): take every fraud the model catches on the test months
at the deployable threshold (5% FPR, set on validation), and let the attacker change
up to B controllable fields, greedily picking at each step the change that lowers the
fraud score most (a query-based black-box attack: it only needs the score). Changed
values are always values that legitimate applicants commonly use. We report the share
of caught frauds that then slip under the threshold (evasion rate) for B = 1, 2, 3.

Transferability (black-box via a proxy, slides 58-60): craft the changes against
logistic regression (the attacker's own proxy model) and apply them, unchanged, to
LightGBM, which the attacker never queries.

Defences (slides 79-87):
  robust features  - proactive design choice: drop every controllable field, so the
                     attack has nothing to change; cost = lower clean recall
  adversarial training - proactive: add attacked copies of the training frauds (still
                     labelled fraud) to the training data and refit; then attack the
                     NEW model again (an adaptive attacker)

Outputs: results/e11_evasion.csv, results/e11_defences.csv, results/e11_changed_fields.csv,
         figures/e11_evasion.png
"""
import numpy as np
import pandas as pd

from src.config import TARGET
from src.data import months, xy
from src.features import add_features
from src.metrics import at_threshold, ranking_metrics, threshold_at_fpr
from src.models import fit_predict
from src.utils import load_json, log, plt, savefig, strategy_for

# Fields an applicant can choose or fake when filling in the form / connecting.
CONTROLLABLE_NUMERIC = ["name_email_similarity", "income", "intended_balcon_amount",
                        "session_length_in_minutes", "current_address_months_count",
                        "prev_address_months_count"]
CONTROLLABLE_BINARY = ["email_is_free", "phone_home_valid", "phone_mobile_valid",
                       "foreign_request", "keep_alive_session"]
CONTROLLABLE_CATEGORICAL = ["housing_status", "employment_status", "device_os", "source", "payment_type"]
CONTROLLABLE = CONTROLLABLE_NUMERIC + CONTROLLABLE_BINARY + CONTROLLABLE_CATEGORICAL
BUDGETS = [1, 2, 3]


def _candidates(train_legit):
    """Values the attacker may switch a field to: common values among legitimate applicants."""
    cand = {}
    for c in CONTROLLABLE_NUMERIC:
        if c in train_legit:
            cand[c] = sorted(set(np.round(train_legit[c].dropna().quantile([0.1, 0.5, 0.9]).to_numpy(), 4)))
    for c in CONTROLLABLE_BINARY:
        if c in train_legit:
            cand[c] = [0, 1]
    for c in CONTROLLABLE_CATEGORICAL:
        if c in train_legit:
            vc = train_legit[c].value_counts(normalize=True)
            cand[c] = list(vc[vc >= 0.01].index)
    return cand


def _score(model, frame, nums, cats):
    return model.predict_proba(xy(add_features(frame), nums, cats)[0])[:, 1]


def greedy_attack(model, frame, cand, nums, cats, budget):
    """Return the attacked frame after each step, the score after each step, and the fields changed.
    Each row independently picks, at every step, the single (field, value) change that
    lowers its score most among fields it has not changed yet."""
    cur = frame.copy()
    changed = np.zeros((len(cur), len(cand)), dtype=bool)
    fields = list(cand)
    history, frames, picks = [], [], []
    for _ in range(budget):
        best_s = _score(model, cur, nums, cats)
        best_opt = np.full(len(cur), -1)
        options = [(fi, v) for fi, f in enumerate(fields) for v in cand[f]]
        for oi, (fi, v) in enumerate(options):
            trial = cur.copy()
            trial[fields[fi]] = pd.Series(v, index=trial.index).astype(cur[fields[fi]].dtype)
            s = _score(model, trial, nums, cats)
            better = (s < best_s) & ~changed[:, fi]
            best_s = np.where(better, s, best_s)
            best_opt = np.where(better, oi, best_opt)
        for oi in np.unique(best_opt[best_opt >= 0]):
            fi, v = options[oi]
            rows = np.where(best_opt == oi)[0]
            col = fields[fi]
            cur.iloc[rows, cur.columns.get_loc(col)] = pd.Series([v] * len(rows), dtype=cur[col].dtype).to_numpy()
            changed[rows, fi] = True
            picks.extend([col] * len(rows))
        history.append(best_s)
        frames.append(cur.copy())
    return frames, history, picks


def run(df, nums, cats, S):
    best = load_json(S.results_dir / "e1_best_params.json")
    tr, va, te = months(df, S.train_months), months(df, S.val_months), months(df, S.test_months)
    y_va, y_te = va[TARGET].to_numpy(), te[TARGET].to_numpy()
    cand = _candidates(tr[tr[TARGET] == 0])
    log(f"E11: attacker controls {len(cand)} fields: {', '.join(cand)}")

    def fit(name, train, use_nums, use_cats):
        model, (s_va, s_te), _ = fit_predict(name, train, [va, te], use_nums, use_cats, S,
                                             strategy_for(best, name), seed=0, params=best["params"][name])
        thr = threshold_at_fpr(y_va, s_va, S.fpr_target)
        return model, thr, s_te

    models = {name: fit(name, tr, nums, cats) for name in ["logreg", "lightgbm"]}

    # ---- 1. direct (query) attacks and the transfer attack ----------------------
    ev, fields_used = [], []
    attacked_by_proxy = None
    for name, (model, thr, s_te) in models.items():
        caught = (s_te > thr) & (y_te == 1)
        frame = te[caught]
        adv, hist, picks = greedy_attack(model, frame, cand, nums, cats, max(BUDGETS))
        for b in BUDGETS:
            ev.append({"target_model": name, "attack": "query (black-box, score only)", "budget_fields": b,
                       "frauds_caught_before": int(caught.sum()),
                       "evasion_rate": float(np.mean(hist[b - 1] <= thr)),
                       "recall_before": float(caught.sum() / y_te.sum()),
                       "recall_after": float((caught.sum() - np.sum(hist[b - 1] <= thr)) / y_te.sum())})
        fields_used += [{"target_model": name, "field": f} for f in picks]
        log(f"E11 {name}: {int(caught.sum())} caught frauds; evasion with 1/2/3 fields: "
            + ", ".join(f"{r['evasion_rate']:.0%}" for r in ev[-len(BUDGETS):]))
        if name == "logreg":
            attacked_by_proxy = (frame.index, {b: adv[b - 1] for b in BUDGETS})

    # transfer: changes crafted on logistic regression, applied to LightGBM
    gbm, thr_g, s_te_g = models["lightgbm"]
    caught_g = pd.Series((s_te_g > thr_g) & (y_te == 1), index=te.index)
    idx, adv_frames = attacked_by_proxy
    for b in BUDGETS:
        # only frauds caught by BOTH models (the proxy attack starts from LR's caught set)
        both = caught_g.loc[idx].to_numpy()
        s_adv = _score(gbm, adv_frames[b][both], nums, cats)
        ev.append({"target_model": "lightgbm", "attack": "transfer from logistic-regression proxy",
                   "budget_fields": b, "frauds_caught_before": int(both.sum()),
                   "evasion_rate": float(np.mean(s_adv <= thr_g))})

    # ---- 2. defences ------------------------------------------------------------------
    rows = []
    def clean_row(label, model_thr_s, use_nums, use_cats):
        model, thr, s_te = model_thr_s
        at = at_threshold(y_te, s_te, thr)
        return {"defence": label, "n_inputs": len(use_nums) + len(use_cats),
                **ranking_metrics(y_te, s_te, S.fpr_target), "valthr_recall": at["recall"], "valthr_fpr": at["fpr"]}

    base = clean_row("none (E1 LightGBM)", models["lightgbm"], nums, cats)
    base["evasion_rate_budget3"] = [r for r in ev if r["target_model"] == "lightgbm"
                                    and r["attack"].startswith("query") and r["budget_fields"] == 3][0]["evasion_rate"]
    rows.append(base)

    # (a) robust features: drop everything the attacker controls (and engineered features built from them)
    derived = {"n_missing", "has_prev_address", "total_address_months", "phones_valid"}
    r_nums = [c for c in nums if c not in CONTROLLABLE and c not in derived]
    r_cats = [c for c in cats if c not in CONTROLLABLE]
    robust = fit("lightgbm", tr, r_nums, r_cats)
    row = clean_row("robust features (controllable fields removed)", robust, r_nums, r_cats)
    row["evasion_rate_budget3"] = 0.0  # nothing the attacker controls reaches the model
    rows.append(row)

    # (b) adversarial training against an adaptive attacker
    frauds_tr = tr[tr[TARGET] == 1]
    adv_tr = greedy_attack(gbm, frauds_tr, cand, nums, cats, max(BUDGETS))[0][-1]
    tr_aug = pd.concat([tr, add_features(adv_tr)], ignore_index=True)  # recompute engineered columns
    hardened = fit("lightgbm", tr_aug, nums, cats)
    model_h, thr_h, s_te_h = hardened
    caught_h = (s_te_h > thr_h) & (y_te == 1)
    _, hist_h, _ = greedy_attack(model_h, te[caught_h], cand, nums, cats, max(BUDGETS))
    row = clean_row("adversarial training (+ attacked training frauds)", hardened, nums, cats)
    row["evasion_rate_budget3"] = float(np.mean(hist_h[-1] <= thr_h))
    rows.append(row)
    for b in BUDGETS:
        ev.append({"target_model": "lightgbm (adversarially trained)", "attack": "query (black-box, score only)",
                   "budget_fields": b, "frauds_caught_before": int(caught_h.sum()),
                   "evasion_rate": float(np.mean(hist_h[b - 1] <= thr_h)),
                   "recall_after": float((caught_h.sum() - np.sum(hist_h[b - 1] <= thr_h)) / y_te.sum())})
    for r in rows:
        r["recall_after_budget3_attack"] = r["valthr_recall"] * (1 - r["evasion_rate_budget3"])

    ev = pd.DataFrame(ev)
    ev.to_csv(S.results_dir / "e11_evasion.csv", index=False)
    pd.DataFrame(rows).to_csv(S.results_dir / "e11_defences.csv", index=False)
    pd.DataFrame(fields_used).value_counts().rename("times_changed").reset_index().to_csv(
        S.results_dir / "e11_changed_fields.csv", index=False)
    log("E11 defences: " + "; ".join(f"{r['defence']}: clean recall {r['valthr_recall']:.3f}, "
                                      f"evasion {r['evasion_rate_budget3']:.0%}" for r in rows))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    styles = {("logreg", "query"): ("#9bb8d6", "o-"), ("lightgbm", "query"): ("#3b6ea5", "o-"),
              ("lightgbm", "transfer"): ("#3b6ea5", "s--"),
              ("lightgbm (adversarially trained)", "query"): ("#2e8b57", "o-")}
    for (target, kind), (color, style) in styles.items():
        d = ev[(ev.target_model == target) & ev.attack.str.startswith(kind)]
        if len(d):
            ax.plot(d.budget_fields, 100 * d.evasion_rate, style, color=color, label=f"{target}: {kind}")
    ax.set_xticks(BUDGETS); ax.set_xlabel("Fields the fraudster may change")
    ax.set_ylabel("Caught frauds that now evade (%)")
    ax.set_title("Evasion attack on the application form (test months)", fontsize=10)
    ax.legend(fontsize=7)
    savefig(fig, S.fig_dir / "e11_evasion.png")
    log("E11 done")
