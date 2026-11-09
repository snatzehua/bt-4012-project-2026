## Context

See proposal.md for motivation. The current improvement report records macro cross-fitted F2 of 0.3278015781 for balanced LightGBM, 0.3226536849 for unweighted LightGBM and 0.3243332661 for blend50. These use month 5/6 predictions and thresholds selected from earlier development months. Those scores are the relevant starting benchmarks; recall at 1% is not the new objective.

All eight months have been inspected. Both completed studies have frozen source/config/software identities. Reusing their training-only preprocessing and eligible cached predictions can reduce unnecessary fitting, but new artifacts must never overwrite their records.

## Goals / Non-Goals

**Goals:** Increase macro cross-fitted F2 against the strongest previous F2 benchmark, distinguish training gains from operating-policy gains, and measure whether the frozen selection transfers to the consumed month 7. Provide a concrete precision/recall/workload interpretation and honest uncertainty.

**Non-Goals:** Promise 90% recall at an unspecified acceptable precision, impose a 1% quota, optimize support-weighted F1, reconstruct unavailable entity histories, or treat a reused period as independent confirmation. A finite search's completion alone does not satisfy a performance-improvement goal.

## Decisions

1. **Metric semantics.** Use F2=5PR/(4P+R)=5TP/(5TP+4FN+FP), with zero for a zero denominator. Beta=2 is an analyst-selected interpretation of the user's greater emphasis on recall, not a claim of group approval. F1 and F3 at the same selected decisions describe sensitivity; do not select a different beta after seeing results. Support-weighted F1 would be dominated by legitimate applications and does not implement the requested preference.

2. **Data and boundaries.** Base v2 retains the existing checksum and sentinel decoding. Fit models on all months before each evaluation month 4, 5 or 6. Exclude fraud_bool, month and constant device_fraud_count from predictors. Threshold/policy evaluation on month 5 uses month 4 predictions only; month 6 uses months 4/5 only. Both macro month averages and pooled confusion counts are reported separately. All development observations are reused exploratory evidence, not a newly untouched test.

3. **Finite model search.** In the new config declare native LightGBM leaves {15,31,63} crossed with positive weights {1,3,10}: nine presets, all 1,200 rounds, learning rate .025, min_child_samples=150, reg_lambda=10, colsample_bytree=.9, deterministic seed 2026. This permits matched capacity and weight contrasts with all other settings fixed. Add engineered CatBoost depth 6, 1,000 iterations, learning rate .05, l2_leaf_reg=5, border_count=128, max_ctr_complexity=1, positive weights {1,3,10}; reuse exact eligible old presets when identity and parameters match. There are at most 36 new development fits before reuse. No outer-fold early stopping. Compare against unchanged cached unweighted/balanced LightGBM and prior native31/CatBoost engineered references on identical cases. Do not add presets after month-7 inspection under the same identity.

4. **Feature justification.** Native LightGBM uses 29 raw eligible predictors, numerical missingness and training-only categorical levels with unseen-to-missing fallback. Engineered CatBoost uses the existing 65-predictor row-local bundle, training-only numerical medians and category strings. These adapters are imported read-only with hash checks. Existing results favor keeping raw LightGBM and testing the engineered CatBoost bundle rather than assuming extra features help every model. No new external dependency is initially needed.

5. **Operating-policy search.** Compare (a) maximum pooled earlier-month F2 threshold, reproducing the old policy; (b) maximum F2 threshold from the latest earlier prediction month; and (c) an earlier-prediction F2-selected review fraction applied by score rank, with ceil(q*n) rounding and stable row-ID ties. The third policy tests robustness to score-scale shifts and searches all attainable fractions, not a manually imposed quota. Raw-threshold ties choose the largest threshold; fraction ties choose the smallest reviewed count. Training weights and F2 beta are distinct choices: changing weights does not itself guarantee improved recall or F2.

6. **Blends and chronology.** Add LightGBM/CatBoost blends with LightGBM weights {.25,.5,.75}; choose each family's member by macro earlier-month F2 under its pooled-threshold policy, tie by candidate ID. For month 5 this uses month 4 only; for month 6 it uses 4/5 only. Historical member selection and policy fitting both use earlier labels, so member scores for those earlier months must be available. State the resulting adaptive development-selection optimism. Blends average raw scores; weighted members are not calibrated probability estimates.

7. **Selection and preservation.** Rank candidate/policy pairs by macro cross-fitted F2, then recall, precision and deterministic ID. Freeze the selected configuration, dependency members, all policies fitted from 4/5/6, and old/new source/config/data/software identities before optional month-7 scoring. Report new models under the original pooled policy and old models under the new policies, so policy-only gains are distinguished from better modelling. Keep the old F2 winner and unweighted baseline regardless of the new selection.

8. **Verification and reporting.** Recompute confusion counts/F2 directly from saved predictions, verify identical cases and eligible threshold-source months, and test ties, degenerate labels, quota rounding, preservation failures and stale-cache rejection. Use 500 paired row bootstrap draws, seed 1027, for F2 differences at fixed policies; use the same sampled evaluation rows per comparator. Report each month, macro performance, conditional intervals and selection limitations. A positive macro development difference is exploratory evidence; a fresh generalization claim requires new data. A positive point estimate with an interval spanning zero is a small uncertain gain, not a confirmed improvement.

## Risks / Trade-offs

- Repeated development selection and known month 7 → preserve identities, declare finite search, report exploratory status and failed transfer honestly.
- Higher recall may create excessive workload → always show precision, FP, FN and flagged fraction; do not hide costs behind F2.
- Threshold instability may dominate apparent model improvement → include policy-only and training-only contrasts on identical cases.
- Cached files may drift or lack full provenance → validate identities and row/label alignment; recompute in the new directory when reuse is not defensible.
- F2 gains may remain small or absent → report the evidence, keep the performance goal unachieved, and describe the next evidence-based change separately.

## Migration Plan

Add the new modules, config and protocol; preserve historical run files. Run development search, audit and freeze. Score only selected dependencies and baselines on month 7 if reporting that exploratory comparison. Publish findings and reproduction steps; synchronize current specs only after actual outputs are verified. New outputs can be removed without altering either completed study.
