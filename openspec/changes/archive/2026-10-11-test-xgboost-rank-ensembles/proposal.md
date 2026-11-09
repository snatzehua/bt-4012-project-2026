## Why

Eight completed studies leave balanced LightGBM's macro chronological F2 of 0.327802 unbeaten; the latest direct-ranking study's best ranker scored 0.321311. Test whether a previously untested tree implementation, alone or combined with the balanced baseline's ranking, improves actual recall-weighted fraud decisions.

This serves the user's performance goal and the revised BAF proposal's explicit XGBoost/LightGBM comparison, class-imbalance and operating-threshold questions. Lecturer feedback motivates explicit model-specific preprocessing justification. Different models can make complementary errors, but neither diversity nor a ranking gain guarantees an F2 gain.

## What Changes

- Declare a separate finite study retaining positive-class F2 beta=2, no review quota or precision floor, and all eight earlier frozen studies and their negative results.
- Test four categorical XGBoost presets crossing depth 4/6 and positive weight 1/5 on the existing eligible original predictors, with fixed remaining parameters and strictly earlier training.
- Compare seven raw model strategies, six label-free percentile representations and twelve fixed percentile blends with balanced LightGBM. Include raw and percentile single-member controls to distinguish blending from representation and policy changes.
- Reuse the three earlier-only empirical F2 policies and select actual macro month-5/6 F2. Expose precision, recall, false positives, missed fraud and workload alongside the objective.
- Verify the new package/native categorical/save-load interface without changing old software identities. Freeze selected strategy, all required controls and their dependency hashes before any further consumed-month access.
- Export checked comparisons, model diversity diagnostics, conditional paired uncertainty, negative results, reproduction commands and assistance attribution. No optional extension is part of this grid.

## Capabilities

### New Capabilities

- `baf-xgboost-rank-ensemble-study`: Preserved categorical model experiments and label-free rank ensembles with matched single-member controls and chronological actual-F2 evidence.

### Modified Capabilities

None. The existing eight studies' requirements and frozen source/config/software/selection records remain authoritative.

## Impact

Plan additive `requirements-xgboost.txt`, `configs/xgboost-rank-ensembles.json`, `docs/xgboost-rank-ensembles-protocol.md`, model/score/runner/report modules and methodological tests. Pin XGBoost 3.4.1, available for the current Python/macOS platform, while retaining all current dependency versions. Planned destinations are ignored `outputs/baf-xgboost-rank-ensembles-v1/` and checked `reports/baf-xgboost-rank-ensembles-v1/`.

Four presets need 12 full development fits and eight separate miniature integration fits. Three old references supply nine verified prediction results, with missing caches reproducible under frozen recipes in new destinations. Twenty-five strategies across three months yield 75 score results; three policies on two forward evaluation folds yield 150 operating rows and 75 macro summaries. Only the frozen winner and specified controls reach already-consumed month 7; at most one new XGBoost final fit is required. All settings are analyst-selected research choices. No new dataset, learned combiner, independent holdout or performance improvement is assumed.
