# Presentation skeleton and speaker outline

Draft, not submission-ready: replace Group XX and member placeholders. Rename the final deck presentationXX.pptx after confirming the group number. Target recording: **13:10**, including title and references; hard cap 15:00. A–D are provisional balanced sections for four speakers; reassign for actual group membership. Aesthetics and final recording remain outstanding.

All slide text is editable. Figures listed below are source assets for the design pass; no unverified chart is invented. Speaker notes live in this outline, not embedded in the PPTX.

## 1. Robust bank account fraud detection

**Timing:** 00:00–00:08 (8s). **Speaker:** A.

- Temporal drift, training history and class imbalance
- BT4012 Fraud Analytics | Group XX
- [Insert all group members and student IDs]

**Speaking notes:** Introduce the title and group. Leave the title readable for eight seconds. Replace the placeholders before recording.

**Evidence / figure candidates:** Project Guidelines, pp. 3–6

## 2. The decision: which applications should be reviewed?

**Timing:** 00:08–00:50 (42s). **Speaker:** A.

- One row is an account application, not a payment transaction.
- Fraud is rare, review capacity is limited, and populations change over time.
- We study a ranking and human-review pipeline; labels do not measure fraud prevented.

**Speaking notes:** Set the business context. Ask whether prioritising recent training data actually improves future detection, and distinguish modelling from intervention outcomes.

**Evidence / figure candidates:** reports/baf-study-v1/findings.md

## 3. Three questions drive the experiments

**Timing:** 00:50–01:40 (50s). **Speaker:** A.

- Does recent history beat equally sized historical training data?
- How do model family and class weighting affect detection?
- Can feature engineering and ensembles improve useful operating points?

**Speaking notes:** Explain why a model leaderboard alone cannot answer the recency question. Each question has a controlled comparison; later improvement work is exploratory.

**Evidence / figure candidates:** docs/experiment-protocol.md; docs/improvement-protocol.md

## 4. BAF Base v2: rare fraud across eight months

**Timing:** 01:40–02:35 (55s). **Speaker:** A.

- 1,000,000 applications; 32 source columns; months 0–7.
- 29 predictors after excluding target, split key and constant device_fraud_count.
- Month 7: 96,843 applications and 1,428 fraud labels (1.47%).
- Development prevalence changes from 0.87% to 1.34%; missingness uses encoded sentinels.

**Speaking notes:** Show the saved monthly prevalence figure when aesthetics are added. Explain synthetic data, monthly timing and why zero literal nulls does not mean complete observations.

**Evidence / figure candidates:** reports/baf-study-v1/monthly.png; reports/baf-study-v1/findings.md

## 5. Chronological evaluation keeps future scoring separate

**Timing:** 02:35–03:25 (50s). **Speaker:** B.

- Train months 0–3 → validate 4; train 0–4 → validate 5; train 0–5 → validate 6.
- Original final model trains 0–6 and scores month 7 after a selection freeze.
- Fit imputers, encoders and scalers on each eligible training subset only.
- Later month-7 comparisons are exploratory: the holdout has already been consumed.

**Speaking notes:** Use the chronological split diagram in the final design. Model selection uses development data; later threshold policies also learn only from earlier development months.

**Evidence / figure candidates:** docs/experiment-protocol.md; reports/baf-improvements-v1/findings.md

## 6. Match size and class counts to isolate recency

**Timing:** 03:25–04:20 (55s). **Speaker:** B.

- K = 100,000; 250,000; 500,000.
- Recent K: latest eligible months, with random selection at the boundary month.
- Historical K: same K and exact fraud/nonfraud counts; seeds 17, 29 and 43.
- All strategies predict identical future applications; expanding history is a separate size comparison.

**Speaking notes:** Historical controls may include recent observations. Do not call this recent-versus-oldest. There are no within-month timestamps, so row order is not treated as time.

**Evidence / figure candidates:** docs/experiment-protocol.md; reports/baf-study-v1/findings.md

## 7. Preprocessing follows the model

**Timing:** 04:20–05:10 (50s). **Speaker:** B.

- All models: documented sentinel handling and training-only transformations.
- Logistic regression: median imputation, missing indicators, scaling and one-hot categories.
- Original LightGBM / random forest: imputation, indicators and one-hot categories; no scaling.
- Improvement study: separately declared native-category and engineered-feature recipes.

**Speaking notes:** Explain the linear baseline and nonlinear tree comparators. Do not retroactively describe the original LightGBM as the later native-categorical pipeline.

**Evidence / figure candidates:** docs/experiment-protocol.md; docs/improvement-protocol.md

## 8. Metrics describe different operating decisions

**Timing:** 05:10–06:10 (60s). **Speaker:** B.

- Average precision summarises ranking under class imbalance.
- Exact 1% review: choose ceil(0.01 × N) rows using label-free score ordering.
- F1 / F2 thresholds are learned on eligible development predictions.
- F2 weights recall more heavily; its search imposes no review quota or precision floor.

**Speaking notes:** Explicitly distinguish fixed capacity, fixed threshold and retrospective recall-at-FPR. Weighted scores and ensemble margins are not claimed calibrated probabilities.

**Evidence / figure candidates:** docs/experiment-protocol.md; reports/baf-bagged-anchor-fusion-v1/findings.md

## 9. Result 1: no reliable general recency advantage

**Timing:** 06:10–07:10 (60s). **Speaker:** C.

- Month-7 recent-minus-matched recall: logistic +0.09 percentage points.
- Conditional 95% interval: −0.61 to +0.89 percentage points.
- LightGBM difference: +0.14 points; interval −0.81 to +0.89 points.
- Both intervals include zero; these settings do not establish a reliable recency advantage.

**Speaking notes:** Use final_recency.png in the designed deck. Explain conditional uncertainty: fixed fitted models, not all retraining or future-month uncertainty. Null evidence is not proof of exact equivalence.

**Evidence / figure candidates:** reports/baf-study-v1/findings.md; reports/baf-study-v1/final_recency.png

## 10. Result 2: weighting changes the trade-off

**Timing:** 07:10–08:10 (60s). **Speaker:** C.

- At equal month-7 1% review: unweighted LightGBM recall 22.34%; balanced 21.64%.
- Unweighted random forest: 21.08%; unweighted logistic regression: 19.40%.
- Weighting raises default-0.5 recall, but changes score scale and false-positive burden.
- Compare models under the same operating policy before claiming improvement.

**Speaking notes:** The original validation-selected candidate remains balanced LightGBM; the higher observed unweighted final result does not retroactively replace it. These are fixed presets, not universal model-family rankings.

**Evidence / figure candidates:** reports/baf-study-v1/findings.md; reports/baf-study-v1/weighting_and_models.png

## 11. Result 3: targeted features help the linear model

**Timing:** 08:10–09:10 (60s). **Speaker:** C.

- Mean forward logistic AP: 0.1542 → 0.1674.
- Recall at 1% review: 19.84% → 21.00%; precision: 24.14% → 25.55%.
- A fixed-C diagnostic separates engineered-feature effects from regularisation changes.
- Feature engineering is model-specific; gains should not be assumed for tree models.

**Speaking notes:** Show the saved improvement ablation figure later. This is averaged development evidence over months 4–6, not a direct comparison to month-7 AP.

**Evidence / figure candidates:** reports/baf-improvements-v1/findings.md; reports/baf-improvements-v1/ablations.png

## 12. Exploratory ensembles: small operational gains

**Timing:** 09:10–10:10 (60s). **Speaker:** C.

- Capacity-selected 25% LightGBM / 75% CatBoost blend: 334 frauds caught versus 319.
- Same 969 reviews: recall 23.39% versus 22.34%; precision 34.47% versus 32.92%.
- 15 fewer missed frauds and 15 fewer legitimate applications reviewed at the same budget.
- Month 7 was already consumed; this is exploratory evidence, not fresh confirmation.

**Speaking notes:** Do not mix the AP-selected 50/50 blend with the capacity-selected 25/75 blend. Explain that most fraud still remains outside the small review queue.

**Evidence / figure candidates:** reports/baf-improvements-v1/findings.md; reports/baf-improvements-v1/exploratory_policies.csv

## 13. Further F2 search did not deliver a clear final win

**Timing:** 10:10–11:10 (60s). **Speaker:** D.

- Selected bagged-anchor fusion: development F2 0.330881 versus balanced 0.327802.
- Consumed month-7 F2: fusion 0.372240 versus unweighted reference 0.376884.
- The development winner remains the selected strategy; no retrospective winner replacement.
- Additional model complexity and search do not guarantee better future detection.

**Speaking notes:** Distinguish original fixed-capacity study from later F2 optimisation. F2 results have different review loads; do not present them as a 1% capacity comparison. Unfinished context ablation is excluded.

**Evidence / figure candidates:** reports/baf-bagged-anchor-fusion-v1/findings.md; reports/baf-shallow-catboost-v1/findings.md

## 14. Business integration: a measured human-review queue

**Timing:** 11:10–12:05 (55s). **Speaker:** D.

- Validate fields → score applications → apply a declared review policy → human investigation.
- Record realised alert volume, precision, recall, missed fraud and legitimate cases flagged.
- Monitor population/prevalence shifts and threshold drift; plan controlled retraining and fairness checks.
- Human review and outcome logging are proposed deployment steps, not evaluated interventions.

**Speaking notes:** Use exact capacity if operational budget is fixed. Fixed score thresholds may create variable review volume. The dataset lacks realised investigation cost, label delays and monetary losses, so do not claim savings.

**Evidence / figure candidates:** docs/experiment-protocol.md; reports/baf-improvements-v1/findings.md

## 15. Conclusion and limits

**Timing:** 12:05–12:50 (45s). **Speaker:** D.

- Retain history unless controlled forward evidence supports a recency benefit.
- Choose preprocessing and operating policies to match the model and review objective.
- Useful improvements are modest; strong simple references remain essential.
- Synthetic data, eight months, unknown label delays and adaptive search limit generalisation.

**Speaking notes:** Add the fixed-model bootstrap and consumed-holdout caveats verbally. New independent data are needed to confirm future performance. Complete member review and verify the reproducible code bundle before submission.

**Evidence / figure candidates:** reports/baf-study-v1/findings.md; reports/baf-improvements-v1/findings.md

## 16. Sources, reproducibility and acknowledgement

**Timing:** 12:50–13:10 (20s). **Speaker:** D.

- Data: Jesus et al. (2022), Bank Account Fraud Dataset Suite, NeurIPS Datasets and Benchmarks.
- Repository: github.com/snatzehua/bt-4012-project-2026
- Protocols, configurations and checked results accompany the code; see submission-code-guide.md.
- ChatGPT/Codex assisted with design, code, audits and writing; all submitted work requires group review.

**Speaking notes:** Mention any external branch material actually adopted in the final version and its assistance attribution. Replace the group placeholders; keep the references slide available after the conclusion.

**Evidence / figure candidates:** README.md; docs/submission-code-guide.md

## Recording checklist

- Replace group number, members and student IDs. Confirm names match the report.
- Rehearse all transitions and keep the actual recording under 15 minutes.
- Ensure every member speaks meaningfully and distribute time reasonably.
- Verify all numbers against generated evidence and keep exploratory labels visible.
- Add acknowledgements for any additional adopted external/AI-assisted material.
- Save final slides as presentationXX.pptx and recording as videoXX.mp4.
- Test video audio, readability and playback before submission.
