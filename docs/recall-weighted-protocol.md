# Exploratory recall-weighted BAF study

The user's objective is improved precision/recall performance with greater emphasis on recall. This study uses **positive-class F2**, not support-weighted F1. F2=5TP/(5TP+4FN+FP): a missed fraud contributes four times the denominator penalty of a false alarm. Beta=2 is an analyst-selected initial preference; it is not an approved bank cost ratio. There is **no imposed 1% review quota or precision floor**. Precision, recall, workload, missed fraud, false positives, F1/F3, AP and ROC AUC accompany F2.

The exact finite search is [configs/recall-weighted.json](../configs/recall-weighted.json). It is declared before new fits and uses separate ignored outputs. Preserve both completed studies' configuration, prediction sources, software identity and selections. This extends the revised proposal's class-imbalance/threshold questions and the feedback's demand for model-specific justification. All available periods have already been examined, so evidence remains exploratory.

## Data and comparisons

Use the same SHA-256-verified Base v2. Models predict months 4,5,6 using all earlier rows; score all candidates on identical cases. Training-only sentinel decoding, categorical handling and exclusion of target/month/constant device_fraud_count follow the completed protocols. Previous-month labels are assumed available; reporting delays and applicant identities are unavailable. No random split, graph reconstruction or future target-derived feature is introduced.

There are 15 base candidates per month: nine native LightGBM presets (leaves 15/31/63 crossed with positive weights 1/3/10; 1,200 rounds, learning rate .025, minimum child samples 150, L2 10, column fraction .9), three engineered CatBoost presets (depth 6, 1,000 rounds, learning rate .05, L2 5, positive weights 1/3/10), and the original unweighted/balanced LightGBM and native31 references. Seed 2026; six threads; sequential fits; no outer-fold early stopping. Five presets have exact prior equivalents, producing 15 verified reuses and **30 new development fits**. Incompatible or absent cache entries are rejected or fitted separately under the new study rather than modifying old artifacts.

Native LightGBM retains 29 original predictors and numeric NaNs; categorical levels are learned from training and unseen categories become missing. CatBoost uses the 65-predictor row-local engineered bundle, training numerical medians and categorical strings. Existing ablations motivated these family-specific feature choices. This search isolates training weight and leaf capacity within the native LightGBM grid, and weight within CatBoost; it is not a fully controlled comparison of model families. Positive training weights change learning, whereas beta defines evaluation preference.

## Policies and chronological selection

For month 5, policies learn from month 4 predictions; for month 6, from months 4/5. Compare pooled maximum-F2 threshold, latest-month maximum-F2 threshold, and pooled maximum-F2 review fraction applied by score rank. Threshold ties favor the largest threshold; learned rank-fraction ties favor the smallest number flagged. Rank ties use ascending stable row ID. The fraction uses exact rational ceiling on the future population; it is learned from F2, not fixed at 1%. No-fraud fitting populations choose no flags.

Add three raw-score LightGBM/CatBoost blends, weights .25/.5/.75 for LightGBM. For each scored fold select each family's member only from earlier predictions, using macro earlier-month F2 at its pooled earlier-month threshold, then candidate ID. Fit blend policies on those same earlier prediction months, never on the scored fold. Their development selection is still adaptive and can be optimistic. Weighted member scores and blends are not asserted to be calibrated probabilities.

Select by macro F2 over months 5/6, then recall, precision and candidate/policy ID. Also report pooled confusion-count F2, since a macro mean is a different estimand. Preserve existing balanced LightGBM pooled-policy macro F2 0.3278015781 and unweighted 0.3226536849 as verified references. Compare new models under the original policy and old models under each new policy to distinguish modelling gains from policy gains. No candidate is selected on AP or recall at 1%.

## Freeze and exploratory transfer

After development checks, freeze selection, dependencies, policies fitted on months 4/5/6 and source/config/software/data identity before further month-7 access. Fit or reuse only selected dependencies and the two baseline comparators. Record access separately without mutating selection. **Month 7 is already consumed**: any selected-model comparison is exploratory, and thresholds are not reselected from month 7.

Report 500 paired row-bootstrap draws (seed 1027) for F2 differences at fixed decisions, against the balanced baseline's original pooled policy; show forward-month variation separately. For rank-fraction policies the interval is conditional on the realized decisions, not a bootstrap of the whole ranking policy. These intervals exclude training/search/temporal/generative uncertainty and multiplicity correction. A positive point estimate is a measured gain on the named scope; an interval spanning zero does not support a clear gain. No gain means the performance goal remains unachieved.

## Outputs and reproduction

Run the README sequence with the existing pinned improvement environment. Bulk predictions, models and checkpoints live in `outputs/baf-recall-weighted-v1/`; findings, exact config/selection, audits, figures and small metric tables in `reports/baf-recall-weighted-v1/`. Validate chronological sources, training populations, row/label equality, cached fingerprints, metric arithmetic, preservation and freeze ordering. Inspect rendered figures. Acknowledge Codex assistance with design, implementation, verification and interpretation, plus BAF, scikit-learn, LightGBM and CatBoost.
