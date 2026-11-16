# Findings: BAF Base variant

Full run: 1,000,000 applications, 8 months, 3 seeds for tree models. Train months 0–4,
validation month 5 (tuning and thresholds only), test months 6–7. Runtime about 40 minutes.
All numbers below come from the csv files in this folder.

## E0 · Data and drift

* Fraud rate is 1.10% overall but **rises over time**: 0.87% in month 2 → 1.34% in month 6 → 1.47% in month 7.
* **Strong covariate drift** in the velocity features: largest PSI vs month 0 is 6.24 for `velocity_4w`
  (η² = 0.75 across months), 3.35 for `velocity_24h`, 2.42 for `velocity_6h`, 1.55 for `zip_count_4w`.
  With 1M rows every ANOVA/chi-square p-value is ~0, so effect sizes are what matter.
* The **link between features and fraud is fairly stable**: the largest month-to-month range in univariate
  AUC is 0.11 (`prev_address_months_count`). The inputs move a lot; the fraud pattern moves little.

## E1 · Models (test months 6–7)

| Model | PR-AUC | Recall @ 5% FPR (95% CI) | Precision @ top 1% | Fit time |
|---|---|---|---|---|
| LightGBM | 0.178 | **0.523** (0.509–0.542) | 0.276 | 20 s |
| Random forest | 0.167 | 0.517 (0.504–0.540) | 0.262 | 147 s |
| Logistic regression | 0.160 | 0.500 (0.483–0.518) | 0.256 | 6 s |

* LightGBM is best, but the margin is small and the confidence intervals overlap.
  Logistic regression is within ~2 points and fully interpretable; the random forest costs 7× LightGBM's training time for no gain.
* Seed variation is small (std of recall ≈ 0.005).

## E2 · Random vs out-of-time split

* A random split is **barely optimistic** on this data: recall @ 5% FPR is +0.4 points (LightGBM) and +0.9 points (LR) higher.
* PR-AUC is actually *higher* on the temporal test (0.181 vs 0.164 for LightGBM) because the test months have
  more fraud (≈1.4% vs ≈1.1%). **PR-AUC depends on the base rate**; ROC-AUC and recall at a fixed FPR do not.
  Compare periods with prevalence-independent metrics.

## E3 · Rolling vs expanding training window

* **Expanding history beats every rolling window in all 36 paired comparisons** (4 test months × 3 seeds × 3 window sizes).
  Loss in recall @ 5% FPR: k = 1 month −5.5 points, k = 2 −2.7, k = 3 −1.9.
* Against an expanding window **cut to the same size**, rolling windows differ by only −0.1 to −0.3 points.
  So the deficit is **entirely due to having less data; recency adds nothing**.
* This matches E0: the features drift, but the fraud pattern is stable, so more history helps more than recent history.

## E4 · Class imbalance

* LightGBM recall @ 5% FPR: no treatment 0.539, undersampling 0.533, class weighting 0.523.
  Logistic regression: all within 0.8 points.
* Undersampling trains **~6× faster** (3 s vs 19 s) for almost the same ranking quality.
* Treatments mainly change score calibration, so thresholds set on validation drift more
  (realised test FPR 5.9% for none/undersample vs 4.9% with weighting).
* Note: E1 used class weighting for every model (fixed in advance); E4 suggests no weighting is slightly better for LightGBM.

## E5 · Operating the model (LightGBM)

**Review capacity** (threshold set on validation):

| Analysts review | Precision | Recall |
|---|---|---|
| Top 1% | 26.7% | 20.3% |
| Top 2% | 21.1% | 31.7% |
| Top 5% | 13.8% | 50.1% |

**Threshold drift:** a threshold giving 5% FPR in month 5 gives **6.0% FPR in month 6 and 3.8% in month 7**.
Thresholds need regular recalibration even when ranking quality holds.

**Costs** (missed fraud : false alarm), cost-optimal threshold chosen on validation vs the fixed 5%-FPR threshold:

| Ratio | Cost-optimal | 5% FPR threshold | Saving | Recall | Applications flagged |
|---|---|---|---|---|---|
| 10 | 22,779 | 23,724 | 4% | 34.6% | 2.4% |
| 25 | 44,038 | 44,289 | 1% | 53.4% | 5.9% |
| 50 | 65,955 | 78,564 | 16% | 70.6% | 12.6% |
| 100 | 94,096 | 147,114 | 36% | 82.0% | 21.7% |

**Fairness:** at the 5%-FPR threshold, applicants aged 50+ have a **false-positive rate of 11.9% vs 3.6%**
for under-50s (LightGBM): about 3.3× the burden. Predictive-equality ratio 0.30 (LR 0.28, RF 0.28).
All three models show it, so it comes from the data, not one model.

## E6 · What drives predictions

* Largest mean |SHAP|: `device_os`, `phone_home_valid`, `prev_address_months_count` (mostly missing),
  `housing_status`, `name_email_similarity`, `current_address_months_count`, `email_is_free`.
  `housing_status` alone is 33% of LightGBM's total gain.
* Logistic regression agrees: `housing_status = BA` (+0.79), a missing previous address (+0.72) and
  Windows devices (+0.56) raise fraud odds; Linux devices (−0.81) lower them.
* **Missed frauds look legitimate**: higher name–email similarity (median 0.36 vs 0.24 for caught fraud),
  lower credit-risk score (160 vs 215) and shorter time at current address (78 vs 108 months).
  The miss rate rises from 45% in month 6 to 50% in month 7.

## Limitations to state

Synthetic data (privacy-preserving generation), only 8 months (4 test months in E3), costs are assumptions,
one dataset variant. Optional robustness check: `python run_all.py --data "data/Variant I.csv" --only e1 e2`.
