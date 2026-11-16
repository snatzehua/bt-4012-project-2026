# Results summary (plain language)

Full run on all 1,000,000 applications of `Base.csv`, 3 seeds, engineered features
(`full_run.log`, 37 minutes on a 20-thread laptop; E9 re-run alone after a fix, `e9_rerun.log`).
Train = months 0–4 (675,666 rows), validation = month 5 (119,323), test = months 6–7 (205,011).
Every number below is on the **test months** unless it says otherwise, and every threshold was
set on the validation month.

## One-paragraph story

Fraud is rare (1.1% of applications, rising to 1.5% in month 7). LightGBM ranks applications best,
but only by a little over logistic regression. With enough data, **more history beats fresher
history**. Recency only helps when training data is scarce. Drift is real but sits mostly in
features the model hardly uses. The practical risks are **threshold drift**, **age bias** and
**fraudsters changing what they type on the form**. The model works as a triage tool: reviewing
the top 1% of applications finds about 26 frauds per 100 reviews, roughly 19 times the base rate.

## Key numbers

| Question | Answer | Evidence |
|---|---|---|
| Which model is best? | LightGBM, by a small margin | Recall at 5% FPR: LightGBM **53.4%** (95% CI 52.2–55.7), random forest 52.0%, logistic regression 51.0%, MLP 51.5%, dummy 5%. PR-AUC 0.184 vs 0.169 (LR); base rate 0.014 (`e1_summary.csv`, `e10_summary.csv`) |
| Which imbalance treatment? | Undersampling (10 legit : 1 fraud), chosen on validation | Same ranking quality as no treatment, 2× faster; class weighting slightly worse for LightGBM (52.2% vs 53.6%) (`e1_best_params.json`, `e4_summary.csv`) |
| Do models overfit? | Logistic regression: no. Trees: somewhat | LR training PR-AUC 0.14 vs validation 0.165. A very strong penalty underfits (`e9_regularisation.png`). LightGBM training PR-AUC 0.26 vs test 0.18 (`e1_summary.csv`) |
| L1 vs L2 | Same accuracy, L1 is sparser | Best L1 keeps 86 of 99 weights with the same validation PR-AUC (0.165) as L2 (`e9_selected.csv`, `e9_lasso_path.png`) |
| Neural networks (L4)? | MLP ≈ logistic regression; autoencoder ≈ random | MLP (one layer of 256, chosen on validation; the deeper nets were worse): 51.5% recall at 5% FPR. Autoencoder trained on legit only: ROC-AUC 0.51. Simple "how unusual is this row" scores also reach only 0.54–0.56, so **fraud in BAF is not unusual-looking**. It must be learned from labels (`e10_*`) |
| Random vs time split too optimistic? | Barely | Recall gap +0.2 pp (LightGBM), +1.0 pp (LR). PR-AUC looks *lower* on the random split only because the test months have more fraud (`e2_summary.csv`) |
| Rolling window vs all history? | All history wins | Expanding beats rolling by 0.9–4.4 pp recall. Against a size-matched random sample of history the gap is about 0, so the loss comes from less data, not from recency (`e3_summary.csv`) |
| Same size K: recent vs old data? (TA feedback) | Recent helps only when K is small | At K = 25k–50k, recent data beats the oldest data by about 1.7–2.0 pp (p < 0.03). From K = 100k up, the difference stays within ±0.6 pp and changes sign (`e3b_summary.csv`, `e3b_recency_effect.png`) |
| Does feature engineering help? | Only logistic regression, slightly | LR PR-AUC +0.007 (CI +0.002 to +0.011). LightGBM PR-AUC −0.005, recall change not significant. Velocity ratios are far more stable over time than raw velocities (PSI 0.19 vs 2.4–6.2) (`e7_summary.csv`, `e7_ratio_stability.csv`) |
| What drives predictions? | Stable application fields, not the drifting velocities | Top SHAP: `device_os`, `housing_status`, `has_other_cards`, `current_address_months_count`, `phone_home_valid`. The most drifting features (`velocity_4w` PSI 6.2) rank low (`e6_shap_importance.csv`, `e0_covariate_drift.csv`) |
| Review capacity | Top 1% → 26 frauds per 100 reviews; top 5% → 54% of fraud | `e5_capacity.csv` |
| Accuracy is misleading (L3) | 94–98% accuracy while recall is 24–56% | `e5_confusion_matrices.csv` |
| Which threshold? Depends on cost (L3) | Top-1% wins if false alarms are expensive; 5%-FPR wins if misses are expensive | `e5_lecture_cost_table.csv`. Cost-optimal alert rate goes from 3.7% (10:1) to 21.7% (100:1) (`e5_costs.csv`) |
| Threshold drift | Recalibrate monthly | A 5%-FPR threshold set in month 5 gives 6.7% FPR in month 6 and 4.6% in month 7 (`e5_fpr_stability.csv`) |
| Fairness | Applicants 50+ are wrongly flagged about 3× as often | LightGBM FPR 12.8% vs 4.3% (ratio 0.34). Removing `customer_age` narrows it (10.7% vs 4.8%, ratio 0.45) but does not close it, because other fields stand in for age (`e5_fairness*.csv`) |
| Can fraudsters dodge it? (L7) | Yes, easily | Changing 2 form fields lets 87% of caught frauds evade LightGBM (3 fields: 96%). Changes crafted on a logistic-regression proxy transfer: 74% (2 fields) evade LightGBM without querying it. Most-used fields: previous-address months, device OS, housing status (`e11_*`) |
| Which defence works? | Trade-off | Adversarial training keeps clean recall (53% vs 56%) and cuts 3-field evasion to 82%. Removing all controllable fields stops the attack but clean recall falls to 33% (`e11_defences.csv`) |
| How is it used? (L5, L9–L11) | Three queues with code-set thresholds | "Escalate" = 1.3% of applications, 26% are fraud, 24% of all fraud. "Review" = 4.6%, 9% fraud, 31% of fraud. "Clear" = 94%, 0.7% fraud. Every case brief validates against a JSON schema and lists what was not checked (`e12_*`) |

## Statistical EDA highlights (Lecture 2, training months only)

* 27 of 29 variables are "significantly" related to fraud (p < 0.05), which shows the large-sample
  trap. Ranking by information gain instead puts `housing_status`, `device_os`, `proposed_credit_limit`,
  `credit_risk_score` and `customer_age` at the top (`e8_bivariate_tests.csv`).
* Outliers matter. Applications that are boxplot outliers on `customer_age` are fraud 5.3× as often as
  average, and on `credit_risk_score` 4.0× as often (`e8_outliers.csv`).
* Redundancy is low. Only `credit_risk_score` and `proposed_credit_limit` correlate above 0.5 (r = 0.62).
* Benford's law. No suitable variable conforms well (MAD mostly > 0.015), and fraud and legit deviate by
  similar amounts. This is expected for synthetic data and for counts like months at an address that
  are not "natural" amounts. Benford is therefore not a useful fraud signal here, which is a valid
  Lecture 2 conclusion. Capped or bucketed variables (income, age, credit limit) were excluded, as
  slide 85 says.

## Suggested report storyline (8–12 pages)

1. **Problem**: application fraud at a digital bank, rare and changing. Why recall at a fixed FPR and
   review capacity matter.
2. **Data**: BAF Base, 1M rows × 32 columns, 8 months, variable table (`e0_variable_summary.csv`), L2
   profile (`e8_univariate.csv`), missing-value sentinels, drift (E0).
3. **Methodology**: out-of-time split; model ladder dummy → LR (L1/L2) → RF / LightGBM → MLP / autoencoder;
   per-model preprocessing and feature engineering; imbalance choice on validation.
4. **Experiments and results**: E1/E9/E10 model comparison; E2–E3b time design; E4 imbalance; E5 operations
   (capacity, cost table, drift, fairness); E6 explanations; E11 robustness; E12 triage workflow.
5. **Conclusion and limitations**: more data beats fresher data; triage not automation; synthetic data;
   no links between applications (no L8 network analysis); cost assumptions; assumed attacker fields.
