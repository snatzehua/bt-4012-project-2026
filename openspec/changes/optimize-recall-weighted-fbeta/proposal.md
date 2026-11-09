## Why

The user wants to improve a precision/recall score that gives recall greater weight. The previous studies measured F2 but searched broader objectives; a dedicated, reproducible F2 search can test whether training choices and threshold transfer improve that operating trade-off without imposing the analyst-selected 1% review quota.

This extends the revised BAF proposal's imbalance and operating-threshold questions and respects the lecturer's request for justified, controlled model comparisons. It is a separate exploratory study because all available months have already been inspected.

## What Changes

- Interpret recall-weighted F1 as F-beta with beta=2 (F2), explicitly distinct from class-support-weighted F1. Report beta=1 and beta=3 as descriptive sensitivities.
- Compare existing F2 baselines, a finite grid of native LightGBM and engineered CatBoost training presets, and threshold-transfer policies using earlier development predictions only.
- Select by chronological macro F2 on identical applications; leave review volume unconstrained and report precision, recall, false positives, missed fraud and workload alongside F2.
- Freeze new source/config/software identity and selections separately. Preserve both completed studies and keep month 7 out of search; any subsequent selected-model comparison there is exploratory.
- Produce runnable code, exact configuration, findings, paired F2 uncertainty, preservation audits and reproduction instructions. Retain unsuccessful experiments honestly; an improved score must be measured, not presumed.

## Capabilities

### New Capabilities

- `baf-recall-weighted-study`: Dedicated chronological optimization and verified reporting of recall-weighted F-beta performance.

### Modified Capabilities

None. Existing completed-study requirements and records remain in force.

## Impact

New `configs/recall-weighted.json`, `docs/recall-weighted-protocol.md`, additive `baf/recall_weighted*.py` modules, methodological tests, ignored `outputs/baf-recall-weighted-v1/` and small `reports/baf-recall-weighted-v1/` exports. Reuse existing feature adapters and verified development predictions read-only with identity checks. Update README, AGENTS and OpenSpec context to distinguish planned and implemented work. No new dataset, raw-data commit or submission artifact is required.
