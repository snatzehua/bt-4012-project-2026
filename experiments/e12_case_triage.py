"""E12 - From score to investigation: a case-triage workflow (Lectures 5, 9, 10, 11).

How the model would sit inside a bank's application-review process, following the
design rules of the agentic-investigation lectures:

* The score, the thresholds and the escalation decision are deterministic CODE, not
  an LLM (Lecture 9: arithmetic, thresholds and lookups belong in tools; Lecture 10:
  the escalation rule is code). Thresholds are set on the validation month from the
  review capacity: top 1% -> "escalate", next 4% -> "review", rest -> "clear".
* Each flagged application gets a structured case brief that must validate against a
  JSON Schema (Lecture 5: schema-constrained structured output), with reason codes
  from the model's own per-feature contributions (LightGBM pred_contrib = TreeSHAP).
* The brief lists what was NOT checked (Lecture 11 report contract `not_checked`), and
  any irreversible action (declining the application) requires human approval
  (Lecture 10 human-in-the-loop).
* An LLM is optional on top: it may turn the validated JSON into prose for the analyst
  (prompt template in docs/llm_case_summary_prompt.md). It never decides the band.

Outputs: results/e12_queue_summary.csv, results/e12_queue_by_month.csv,
  results/e12_reason_frequency.csv, results/e12_case_brief_schema.json,
  results/e12_case_briefs_sample.jsonl, figures/e12_queue.png
"""
import json

import joblib
import jsonschema
import numpy as np
import pandas as pd

from src.config import MONTH, TARGET
from src.data import months, xy
from src.utils import log, plt, save_json, savefig

CAPACITY_ESCALATE, CAPACITY_REVIEW = 0.01, 0.05

DESCRIBE = {  # plain-language names for reason codes
    "name_email_similarity": "e-mail address does not resemble the applicant's name",
    "email_is_free": "free web-mail address",
    "phone_home_valid": "home phone number validity",
    "phone_mobile_valid": "mobile phone number validity",
    "phones_valid": "number of valid phone numbers",
    "housing_status": "declared housing status",
    "device_os": "device operating system",
    "credit_risk_score": "internal credit risk score",
    "has_other_cards": "has other cards with the bank",
    "proposed_credit_limit": "credit limit requested",
    "income": "declared income decile",
    "customer_age": "applicant age band",
    "keep_alive_session": "kept the session logged in",
    "prev_address_months_count": "months at previous address",
    "current_address_months_count": "months at current address",
    "total_address_months": "total months at current + previous address",
    "has_prev_address": "previous address given",
    "date_of_birth_distinct_emails_4w": "e-mails seen with the same date of birth (4 weeks)",
    "device_distinct_emails_8w": "e-mails seen from the same device (8 weeks)",
    "payment_type": "credit payment plan",
    "employment_status": "declared employment status",
    "bank_months_count": "months with previous bank",
    "session_length_in_minutes": "session length",
    "zip_count_4w": "applications from the same zip code (4 weeks)",
    "velocity_6h": "application volume, last 6 hours",
    "velocity_24h": "application volume, last 24 hours",
    "velocity_4w": "application volume, last 4 weeks",
    "bank_branch_count_8w": "applications at the same branch (8 weeks)",
    "intended_balcon_amount": "initial transfer amount",
    "source": "application channel",
    "foreign_request": "request from a foreign country",
    "days_since_request": "days since the request",
    "n_missing": "number of optional fields left empty",
}

ACTIONS = {
    "escalate": "Hold the application. Senior analyst verifies identity documents and contact details "
                "before any decision. Declining requires analyst approval.",
    "review": "Step-up verification: one-time code to the mobile number and proof of address. "
              "Proceed automatically only if both pass.",
    "clear": "Proceed with standard onboarding.",
}

NOT_CHECKED = [
    "authenticity of identity documents (no document images in the BAF data; Lecture 7 image checks not possible)",
    "links to other applications sharing a device, phone or address (no shared identifiers in BAF; "
    "Lecture 8 network analysis not possible)",
    "external watchlists, credit bureau and sanctions screening",
]

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Application fraud case brief",
    "type": "object",
    "additionalProperties": False,
    "required": ["case_id", "month", "risk_percentile", "band", "escalation_rule", "top_reasons",
                 "recommended_action", "requires_human_approval", "not_checked", "model"],
    "properties": {
        "case_id": {"type": "string", "pattern": "^APP-[0-9]{7}$"},
        "month": {"type": "integer", "minimum": 0, "maximum": 7},
        "risk_percentile": {"type": "number", "minimum": 0, "maximum": 100},
        "band": {"enum": ["escalate", "review", "clear"]},
        "escalation_rule": {"type": "string"},
        "top_reasons": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["feature", "description", "value", "typical_legit_value", "contribution"],
            "properties": {"feature": {"type": "string"}, "description": {"type": "string"},
                           "value": {"type": ["string", "number", "null"]},
                           "typical_legit_value": {"type": ["string", "number", "null"]},
                           "contribution": {"type": "number"}}}},
        "recommended_action": {"type": "string"},
        "requires_human_approval": {"type": "boolean"},
        "not_checked": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "model": {"type": "string"},
    },
}


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.floating, float)):
        return round(float(v), 3)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def run(df, nums, cats, S):
    gbm = joblib.load(S.results_dir / "models" / "lightgbm.joblib")
    tr, va, te = months(df, S.train_months), months(df, S.val_months), months(df, S.test_months)
    X_va, X_te = xy(va, nums, cats)[0], xy(te, nums, cats)[0]
    s_va, s_te = gbm.predict_proba(X_va)[:, 1], gbm.predict_proba(X_te)[:, 1]
    y_te = te[TARGET].to_numpy()

    # ---- deterministic escalation rule (thresholds from the validation month) -----
    t_esc = float(np.quantile(s_va, 1 - CAPACITY_ESCALATE))
    t_rev = float(np.quantile(s_va, 1 - CAPACITY_REVIEW))
    band = np.where(s_te > t_esc, "escalate", np.where(s_te > t_rev, "review", "clear"))
    rule = (f"escalate if score > {t_esc:.4f} (top {CAPACITY_ESCALATE:.0%} of month-5 scores); "
            f"review if score > {t_rev:.4f} (top {CAPACITY_REVIEW:.0%}); else clear")
    sorted_va = np.sort(s_va)
    pct = 100 * np.searchsorted(sorted_va, s_te, side="right") / len(sorted_va)

    q = pd.DataFrame({"band": band, "fraud": y_te, MONTH: te[MONTH].to_numpy()})
    summ = q.groupby("band").agg(applications=("fraud", "size"), frauds=("fraud", "sum")).reindex(
        ["escalate", "review", "clear"])
    summ["share_of_applications"] = summ["applications"] / len(q)
    summ["precision"] = summ["frauds"] / summ["applications"]
    summ["share_of_all_fraud"] = summ["frauds"] / q["fraud"].sum()
    summ["lift_vs_base_rate"] = summ["precision"] / q["fraud"].mean()
    summ.to_csv(S.results_dir / "e12_queue_summary.csv")
    q.groupby([MONTH, "band"]).agg(applications=("fraud", "size"), frauds=("fraud", "sum")).to_csv(
        S.results_dir / "e12_queue_by_month.csv")
    log("E12 queue: " + "; ".join(f"{b} {r.share_of_applications:.1%} of apps, precision {r.precision:.1%}, "
                                   f"{r.share_of_all_fraud:.0%} of fraud" for b, r in summ.iterrows()))

    # ---- reason codes from per-feature contributions (TreeSHAP) --------------------
    flagged = np.where(band != "clear")[0]
    contrib = gbm.predict(X_te.iloc[flagged], pred_contrib=True)[:, :-1]  # last column = bias
    feats = list(X_te.columns)
    legit_typ = {c: (tr.loc[tr[TARGET] == 0, c].mode().iloc[0] if c in cats
                     else tr.loc[tr[TARGET] == 0, c].median()) for c in feats}
    top_idx = np.argsort(-contrib, axis=1)[:, :3]
    reason_counts = pd.Series([feats[j] for row in top_idx for j in row]).value_counts()
    pd.DataFrame({"feature": reason_counts.index, "times_in_top3": reason_counts.values,
                  "share_of_flagged_cases": reason_counts.values / len(flagged),
                  "description": [DESCRIBE.get(f, f) for f in reason_counts.index]}).to_csv(
        S.results_dir / "e12_reason_frequency.csv", index=False)

    # ---- structured case briefs, validated against the schema ----------------------
    save_json(SCHEMA, S.results_dir / "e12_case_brief_schema.json")
    validator = jsonschema.Draft202012Validator(SCHEMA)
    rng = np.random.default_rng(0)
    pick = {b: rng.choice(np.where(band == b)[0], size=min(n, int((band == b).sum())), replace=False)
            for b, n in [("escalate", 8), ("review", 6), ("clear", 4)]}
    pos = {r: i for i, r in enumerate(flagged)}
    n_valid = 0
    with open(S.results_dir / "e12_case_briefs_sample.jsonl", "w", encoding="utf-8") as fh:
        for b, rows in pick.items():
            for r in rows:
                rec = te.iloc[r]
                reasons = []
                if b != "clear":
                    c = contrib[pos[r]]
                    for j in top_idx[pos[r]]:
                        f = feats[j]
                        reasons.append({"feature": f, "description": DESCRIBE.get(f, f), "value": _fmt(rec[f]),
                                        "typical_legit_value": _fmt(legit_typ[f]),
                                        "contribution": round(float(c[j]), 3)})
                brief = {"case_id": f"APP-{int(te.index[r]):07d}", "month": int(rec[MONTH]),
                         "risk_percentile": round(float(pct[r]), 2), "band": b, "escalation_rule": rule,
                         "top_reasons": reasons, "recommended_action": ACTIONS[b],
                         "requires_human_approval": b == "escalate", "not_checked": NOT_CHECKED,
                         "model": "LightGBM (E1, seed 0), trained on months 0-4"}
                errors = list(validator.iter_errors(brief))
                if errors:
                    raise ValueError(f"case brief failed schema validation: {errors[0].message}")
                n_valid += 1
                # the true label is written NEXT to the brief, for evaluation only; an analyst never sees it
                fh.write(json.dumps({"brief": brief, "evaluation_only_true_label": int(rec[TARGET])}) + "\n")
    log(f"E12: {n_valid} sample case briefs written, all valid against the schema")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    order = ["escalate", "review", "clear"]
    axes[0].bar(order, 100 * summ.loc[order, "precision"], color=["#c0392b", "#e59866", "#9bb8d6"])
    axes[0].set_ylabel("Fraud among cases (%)"); axes[0].set_title("Precision per queue", fontsize=10)
    axes[1].bar(order, 100 * summ.loc[order, "share_of_all_fraud"], color=["#c0392b", "#e59866", "#9bb8d6"])
    axes[1].set_ylabel("Share of all test fraud (%)"); axes[1].set_title("Where the fraud ends up", fontsize=10)
    for ax, col in zip(axes, ["share_of_applications", "share_of_applications"]):
        for i, b in enumerate(order):
            ax.annotate(f"{summ.loc[b, col]:.1%} of apps", (i, 0), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color="white" if b != "clear" else "black")
    savefig(fig, S.fig_dir / "e12_queue.png")
    log("E12 done")
