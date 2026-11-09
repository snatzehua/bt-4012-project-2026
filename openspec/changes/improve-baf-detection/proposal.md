## Why

The user requests tested strategies and feature engineering that improve fraud detection, including F1 and recall rather than precision alone. This extends the revised BAF proposal's model/imbalance scope after the first study; the original protocol and consumed month-7 holdout must remain intact.

## What Changes

- Add a separate exploratory study comparing model capacity, categorical handling, row-local engineered features, moderate class weights, recency weights, undersampling and ensembles.
- Evaluate chronological validation with AP, ROC AUC, F1/F2, precision, recall, MCC, balanced accuracy, confusion counts and fixed review capacities.
- Freeze validation-selected candidates and operating thresholds before descriptive evaluation on already-consumed month 7; quantify gains and trade-offs against unchanged baselines.
- Deliver reproducible configuration, tests, small tables, plots and findings, including unsuccessful strategies and uncertainty.

## Capabilities

### New Capabilities
- `baf-improvements`: Reproducible exploratory improvement experiments with multiple operating objectives and leakage-safe feature engineering.

### Modified Capabilities
None. The original study requirements and frozen implementation remain unchanged.

## Impact

New modules, configuration, supplementary dependencies, tests and reports. Existing `baf/common.py`, `baf/study.py`, first-study configuration and selection records are preserved. No raw data or bulk predictions are committed. Exhaustive optimization, production deployment, fairness certification and fresh confirmatory evidence are outside scope.
