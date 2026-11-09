# Adaptive recall-weighted F2 extension

The first dedicated F2 search failed to improve macro development F2 0.3278015781. This separate extension pursues the same user objective: emphasize recall in positive-class F2=5TP/(5TP+4FN+FP), without an imposed review quota or precision floor. It does not optimize support-weighted F1 or change beta after seeing results. Settings are analyst-selected, not group-approved. All evidence is exploratory, including reused development periods and already-consumed month 7.

The exact finite grid is [configs/recall-weighted-extension.json](../configs/recall-weighted-extension.json), declared before new fits. Preserve the three completed studies' sources/config/software/selections and negative findings. Reuse frozen preprocessing and metric helpers read-only; write new data only beneath `outputs/baf-recall-weighted-extension-v1/`.

## Data, preprocessing and training

Use the same checksum-verified BAF Base v2, development months 0-6. Models predict 4/5/6 using earlier months only. Predictors exclude fraud_bool, month and constant device_fraud_count. Numeric missing-code decoding follows the first protocol. Native LightGBM has 29 original predictors, numeric NaNs and training-only categorical levels, with unknown categories mapped to missing. Reused CatBoost references retain the documented 65-feature row-local bundle and training numeric medians. Original window models retain train-fitted imputation and one-hot categoricals; their comparison with native models does not isolate preprocessing.

Twelve new native LightGBM presets cross leaves {15,31}, positive weights {1,30,100}, rounds {200,600}; learning rate .05, minimum child samples 100, L2 10, column fraction .9, seed 2026, six threads. There are 36 new development fits with existing caches, sequential and checkpointed. No outer-fold early stopping. Weight-1 controls and crossed settings support matched weight/round/leaf contrasts. Training weights are different from evaluation beta.

Retain all 15 previous F2 base presets and add 12 original unweighted LightGBM window recipes: recent K and three-seed class/size-matched historical K for K 100000/250000/500000 and seeds 17/29/43. The original sampling logic reconstructs exact training-row hashes and fraud counts; recorded AP/ROC and evaluation IDs/labels are checked against saved scores. Missing caches are refitted into extension directories using the exact frozen recipe, with new provenance and software identity. Reuse never follows a filename alone. Prior-month labels are assumed available; no unavailable entity history or reporting-delay simulation is invented.

## Derived strategies and timing

Average each K's three historical seed scores, creating three fixed ensembles. Add six earlier-selected rank ensembles anchored on balanced LightGBM: best eligible window/seed-mean member and best eligible native LightGBM member, each with anchor weight .25/.5/.75. Choose the second member by macro earlier-month F2 at its pooled earlier threshold, tie by ID. Normalize scores within the same unlabeled month to `(average_rank(score)-.5)/n` before blending. This is a batch rule; it requires the scored application batch and does not produce calibrated fraud probabilities. Current-month labels never enter normalization or member selection.

There are 39 base models, three seed means and six rank strategies: 48 strategies and 288 operating evaluations (three policies, months 5/6). For month 5, member/policy fitting uses month 4 predictions; for month 6 it uses 4/5 only. Policies are pooled maximum-F2 threshold, latest-month maximum-F2 threshold, and pooled maximum-F2 review fraction applied by rank. Threshold ties favor the largest threshold; fraction ties the smallest reviewed count. Rank ties use ascending stable row ID; quotas use exact rational ceiling. No-fraud fitting populations flag no cases.

Select by macro F2, then recall, precision and deterministic strategy/policy ID. Report pooled confusion-count F2 separately. All candidates score identical future applications. Retain the previous balanced/unweighted LightGBM F2 policies and reproduce their original development metrics. Export negative contrasts too: weight/round/leaf, window versus matched controls, seed averaging and rank blending versus their constituents. Repeated selection across reused periods adds optimism; no comparison establishes causal recency effects.

## Freeze, transfer and evidence

Freeze strategy, recursive dependencies, policy parameters learned from 4/5/6, and source/config/software/data identity before further month-7 access. Score only selected dependencies and both baselines; reuse exact eligible predictions or refit the selected recipe in the new directory. Record access separately and preserve the selection hash. **Month 7 is consumed** and cannot confirm a newly tuned protocol independently.

Report F2, F1/F3, precision/recall, AP/ROC, TP/FP/FN/TN, workload, MCC/balanced accuracy and conditional paired F2 intervals (500 row-bootstrap draws, seed 1027). Bootstrap fixed decisions on identical paired rows, with within-month resampling for macro development differences. These intervals exclude retraining, selection, temporal and generative uncertainty and multiplicity correction. Rank-policy intervals condition on realized decisions, not a reselected ranking workload. A point gain is limited to its named scope; intervals including zero preclude clear-gain claims. No measured gain leaves the performance goal unachieved.

## Reproduction and assistance

Use the README command sequence with the pinned improvement environment. Fresh-clone steps restore exported frozen manifests and allow absent predictions to be refitted. Share small configs/selections/tables/audits/inspected figures/findings in `reports/baf-recall-weighted-extension-v1/`, never raw data or bulk predictions. Checks cover preservation, cache identity, original training reconstruction/class matches, earlier-only sources, batch ranks, saved-score hashes and direct metric recomputation.

Codex assisted with design, implementation, execution, verification and interpretation. Acknowledge BAF, scikit-learn, LightGBM and CatBoost in the eventual course report. This extension does not constitute the final 8-12 page deliverable.
