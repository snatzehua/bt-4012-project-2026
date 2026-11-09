# BAF exploratory study protocol v1

Written before any model evaluation on 10 October 2026. This is an analyst-selected first study under the user's request to perform EDA and test the proposal's theories; it is not a claim that the group has approved every setting. The executable settings are in `configs/study.json`.

## Questions and scope

1. Does recent training improve next-month detection compared with the same number of historical applications and the same class counts?
2. Does using all eligible history beat smaller training sets, and do conclusions depend on K or model family?
3. How does class weighting change ranking and the precision/recall trade-off?
4. How do logistic regression, random forest and gradient boosting compare under common chronological evaluation?
5. Does a random-split diagnostic give different performance estimates? Such a gap alone is not evidence of temporal drift.

Core work is the matched-K recency comparison, EDA and chronological model/weighting comparisons. Random splitting is a labelled secondary diagnostic restricted to development data. Resampling, SHAP, additional BAF variants, delay interventions and extensive hyperparameter optimization are future extensions, not silently implemented experiments.

## Data and information boundary

Use the downloaded **BAF Base, Kaggle version 2**, with the verified SHA-256 in the config. It has 1,000,000 synthetic rows, 32 columns and eight months. Base avoids the deliberately injected biases of other variants; this does not imply it is unbiased. The current file has no index column: `fraud_bool` is the first column and must not be dropped as an index.

Exploratory summaries and feature choices use months **0-6** only (903,157 applications). Month **7** is excluded from development EDA, random diagnostics and tuning. Its row-level scores and labels are used only after the selection manifest is written. The initial download's structural counts are not model performance evaluation.

Assume all labels from previous months are available by the cutoff. Actual reporting delays are absent, so this is a retrospective benchmark, not a deployment simulation. Existing aggregate features are supplied synthetic inputs; applicant/account identifiers and exact event timestamps are absent, so we cannot reconstruct or independently audit their historical computation. Do not create graph or rolling features from row order.

Sources: [official BAF repository](https://github.com/feedzai/bank-account-fraud), [datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf), [dataset version](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/versions/2). Variable descriptions below are short paraphrases; the downloaded file determines the actual column names.

## Variable inventory

| Variable | Role / meaning |
|---|---|
| fraud_bool | Target; fraud indicator |
| month | Split only; application month |
| income | Income decile |
| name_email_similarity | Name/email similarity |
| prev_address_months_count | Previous residence duration |
| current_address_months_count | Current residence duration |
| customer_age | Rounded age |
| days_since_request | Application elapsed days |
| intended_balcon_amount | Intended initial transfer |
| payment_type | Payment category |
| zip_count_4w | Postcode application count |
| velocity_6h | Short application velocity |
| velocity_24h | Daily application velocity |
| velocity_4w | Monthly application velocity |
| bank_branch_count_8w | Branch application count |
| date_of_birth_distinct_emails_4w | Birth-date email count |
| employment_status | Employment category |
| credit_risk_score | Internal risk score |
| email_is_free | Free-email flag |
| housing_status | Housing category |
| phone_home_valid | Home-phone validity |
| phone_mobile_valid | Mobile-phone validity |
| bank_months_count | Existing account duration |
| has_other_cards | Other-card flag |
| proposed_credit_limit | Requested credit limit |
| foreign_request | Foreign-origin flag |
| source | Application channel |
| session_length_in_minutes | Session duration |
| device_os | Device operating system |
| keep_alive_session | Session persistence flag |
| device_distinct_emails_8w | Device email count |
| device_fraud_count | Device fraud count; excluded |

Treat `-1` as missing for address durations, bank duration, session duration and device email count; negative intended-transfer amounts are missing. Preserve negative `credit_risk_score` and `velocity_6h`, which are not documented missing codes. `device_fraud_count` is constant zero across development data and is excluded; its deployment-time label availability would otherwise need scrutiny. `month` is never a predictor. All remaining 29 fields are candidates.

## Splits and training sets

| Stage | Eligible training | Evaluation |
|---|---|---|
| Validation 1 | Months 0-3 (547,975 rows) | Month 4 |
| Validation 2 | Months 0-4 (675,666 rows) | Month 5 |
| Validation 3 | Months 0-5 (794,989 rows) | Month 6 |
| Final | Months 0-6 (903,157 rows) | Month 7 |

For each validation cutoff, use **K = 100,000; 250,000; 500,000**. All are feasible by row count in every fold. Compare:

- Expanding: every eligible historical application.
- Recent K: take latest eligible months first. Within the boundary month, use a fixed target-independent pseudorandom priority (seed 2026). There is no claim of intramonth ordering.
- Matched historical K: sample without replacement from the full eligible pool, including recent rows, matching both K and the exact fraud/nonfraud counts in recent K. Repeat with seeds **17, 29, 43**. Overlap with recent K is allowed; this is historical retention versus recency, not an older-only population experiment. Matching controls class composition, but conditions on observed training labels and is not a claim of causal identification.

Record selected row-ID hashes, size, fraud count, training-month composition and seed. Reject infeasible requests rather than silently changing class counts or K. The evaluation cases and preprocessing policy are identical within each comparison. Training-set size differs only in the explicit expanding-history comparison.

## Preprocessing and models

Fit all learned transformations on each training subset. Replace documented sentinels, median-impute numeric inputs with missingness indicators, and one-hot encode five categorical fields with unknown-category handling. Logistic regression additionally standardizes numeric inputs. Trees keep numerical scale. No test-fitted vocabularies, imputation or scaling; the missing-code rules themselves come from documentation.

Use logistic regression and LightGBM for all matched-K comparisons, initially unweighted to isolate training strategy. On expanding history, compare unweighted and balanced class weighting for logistic regression, LightGBM and random forest. This is a controlled weighting ablation, not the full cross-product of every factor. Models and their exact fixed parameters are in the config. LightGBM is justified by nonlinear tabular interactions; logistic regression provides an interpretable linear reference; random forest is a bagged-tree reference. Random forest is not part of every K/seed experiment to control runtime.

There is **no hyperparameter search** in v1: one preset per model, applied consistently across strategies; random forest has 150 trees and LightGBM 200 boosting rounds, which are not equal computational budgets. Record fit time and convergence warnings. The conclusions are about these model configurations, not the best achievable family performance. Run sequential fits with six threads at most; do not automatically alter presets after seeing final-test outcomes.

## Metrics, selection and final test

Primary: recall at the top **1%** review capacity. Sensitivities: **0.5%, 2%, 5%**. Select `ceil(q * n)` applications, ordered by decreasing predicted score, then ascending stable row ID for score ties. Report capacity count, precision, recall, TP/FP/FN/TN and missed fraud. Also report average precision (non-interpolated scikit-learn AP, not trapezoidal PR-AUC), ROC-AUC and precision/recall/F1 at threshold 0.5. Accuracy is descriptive only.

Validation chooses K separately for each of logistic regression and LightGBM using recent-strategy macro-mean recall at 1%; ties favor smaller K. Fix that K for its final recent-versus-three-matched comparisons. Also evaluate both weighting policies on expanding history for all three model families. Choose an overall candidate among these predeclared final comparators using validation macro-mean recall at 1%, then AP, then deterministic config ordering. Macro-average seeds within a month before averaging months, so historical sampling gets no extra month weight.

Choose each final run's F1 threshold from the median of its corresponding validation-optimal thresholds, including its seed where applicable. Select the largest threshold in an exact validation F1 tie. This is a secondary operating rule; fixed-capacity metrics do not use this threshold. The 0.5 results illustrate sensitivity to weighting and are not the deployment recommendation.

Write a selection manifest containing frozen configuration, source code and data hashes before final scoring. Final evaluation reports every predeclared comparator (14 model fits) once; do not retune or choose K from month 7. Further studies after accessing month 7 must call it an already-consumed holdout and use new evidence for confirmatory claims.

Use **500 paired row-bootstrap replicates** on final predictions for recent minus mean matched-historical recall at 1%, and recent minus expanding recall at 1%. Apply the same resampled rows to all compared scores and recompute the capacity in each resample. Report percentile 95% intervals, sampling-seed spread and validation-month variation. These conditional intervals capture test-row variation, not retraining, temporal or generative-model uncertainty; they are descriptive and unadjusted for multiple comparisons.

## Secondary diagnostic and outputs

Use a seed-2026 stratified random 80/20 split of development months 0-6, fitting unweighted logistic regression, LightGBM and random forest. Exclude month 7. This mixes months in train and evaluation and is a diagnostic comparison with different populations/training sizes, not a controlled drift experiment.

EDA exports development-month prevalence, schema, duplicate/null/sentinel/constant checks, distributions, categorical composition, numerical month-0-versus-month-6 KS effect sizes and categorical total-variation distances. Large sample sizes make p-values unhelpful for practical drift; emphasize effect sizes. Save run-level metrics, sample audits, final predictions, fit times, frozen decisions, paired differences, plots and a readable research summary. Raw inputs and bulky outputs remain ignored; retain small results and the summary under `reports/`.

## Acceptance check and limitations

Before final scoring, verify hash identity, strict train-month-before-evaluation boundaries, exact matched sizes/class counts, deterministic ties, training-only preprocessing and budget accounting. A second group member must be able to run the README commands without choosing missing settings. Automated checks exercise those failure modes on small examples; full run audits establish the real-data boundaries.

Eight synthetic months, one variant, coarse time, unknown reporting delays, only three forward folds, one preset per model and the absence of realized fraud-loss amounts limit generalization. Distribution changes do not by themselves establish changing fraud mechanisms. No monetary savings or deployable bank performance will be claimed.

AI acknowledgment: Codex assisted with protocol design, code, verification and interpretation. Cite BAF and the used software libraries in the final report; the group must review and explain all submitted material. This protocol records an executable first study, not a completed course deliverable.
