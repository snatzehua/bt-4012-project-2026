## 1. Implement the exploratory study

- [x] 1.1 Add separate configuration, pinned supplementary dependencies, feature engineering and model adapters; verify original-study hashes remain unchanged and feature/preprocessing tests pass.
- [x] 1.2 Implement multi-metric evaluation, temporal threshold cross-fitting, blends, resumable validation and selection guards; verify metrics, threshold selection and leakage tests pass.

## 2. Execute and evaluate

- [x] 2.1 Run all specified forward candidates and blends; verify complete records, identical evaluation cases, temporal boundaries and per-month comparisons.
- [x] 2.2 Freeze objective-specific selections and thresholds, then evaluate selected models and unchanged baselines on consumed month 7; verify frozen identity, preservation hashes and prediction records.
- [x] 2.3 Calculate paired uncertainty, compare feature/training ablations and inspect multiple operating policies; verify computations against saved predictions and confusion counts.
- [x] 2.4 Run the adaptive raw-logistic C=0.1 diagnostic on development folds; verify fixed-C feature contrasts, identical cases and explicit exclusion from month-7 selection.

## 3. Deliver reproducible findings

- [x] 3.1 Generate findings, small tables and charts with gains, failures and exploratory limitations; inspect figures and cross-check every headline against measured outputs.
- [x] 3.2 Update README, project context and main specifications; verify methodological tests, original-study preservation, documentation links and strict OpenSpec validation.
