# Exploratory BAF improvement protocol

This extends the revised BAF proposal and first study in response to the request to improve precision/recall and evaluate F1 and other metrics. These are analyst-selected settings, not group-approved final methodology. The complete finite search is in [configs/improvements.json](../configs/improvements.json), declared before its model scoring. The original [protocol](experiment-protocol.md), code, configuration and selection record remain preserved.

## Data and evaluation

Use the same verified Base v2 CSV. Development folds predict months 4, 5 and 6 using only earlier months. All candidates predict identical applications; undersampling affects training only. Predictors exclude target, month and constant-zero device fraud count. Missing codes and label-availability assumptions follow the original protocol.

**Month 7 is already consumed.** This study may describe its performance after new development selections are frozen, but cannot produce fresh confirmatory evidence. Prior knowledge of the dataset, repeated development selection and a finite search all limit inference. Nothing can guarantee higher precision and recall at every operating point.

## Candidate strategies

Seventeen base presets yield 51 fits across three forward months:

| Strategy | Purpose |
|---|---|
| Original LightGBM unweighted/balanced and logistic | Preserve exact first-study baselines |
| Native categorical small LightGBM | Compare native missing/categorical processing with the small baseline pipeline |
| Native LightGBM 31/63 leaves | Test longer boosting with stronger capacity and explicit regularization |
| Engineered LightGBM 31/63 leaves | Isolate the engineered feature bundle at each matching preset |
| Engineered LightGBM positive weights 3/10 | Test moderate imbalance weighting at a fixed model/feature preset |
| Engineered LightGBM recency half-life 2 months | Downweight old training cases smoothly without throwing them away |
| Engineered LightGBM negative:positive ratio 20:1 | Test training-only random undersampling with prior-odds correction |
| Engineered logistic, C=0.1 | Test nonlinear transformations/categorical crosses plus stronger regularization; this contrast does not isolate features alone |
| CatBoost depth 6/8 | Test native categorical modelling at two fixed capacity presets |
| Engineered CatBoost depth 6, unweighted/positive weight 3 | Isolate engineered features and then moderate weighting |

Iteration counts are fixed (see configuration). Outer validation labels are not used for early stopping. Models run sequentially with six threads and checkpoint by source/config/software identity. Seeds are 2026. CatBoost uses CPU Plain boosting with internal permutation-based categorical target statistics from training cases only; no target encodings are constructed from validation labels.

Select the best LightGBM and CatBoost preset by mean forward average precision (ties by candidate name), then compare probability blends with LightGBM weights 0.25/0.5/0.75. These add nine prediction comparisons, not new base fits. Choosing blend members with the same development folds introduces search optimism, which is reported explicitly.

Blends average the raw `predict_proba` scores. If a selected member uses class weights, its score scale can differ from an unweighted member; the nominal blend weights do not turn these into calibrated fraud probabilities.

## Engineered feature bundle

All additions are computed independently for each row; no future population aggregates, synthetic entity IDs or target-derived group rates are introduced. All original eligible predictors are retained.

| Additions | Definition and interpretation |
|---|---|
| Signed logs | `sign(x)*log1p(abs(x))` for the three velocity fields, zip/branch counts, request delay, session length, intended transfer, credit limit and risk score; preserves valid negative values and missingness |
| Velocity contrasts | 6h/(24h+1), 24h/(4w+1), 6h/(4w+1), 6h−24h, 24h−4w; global application velocities share hourly units |
| Activity normalization | Zip count/(4w hourly velocity×24×28+1), branch count/(4w velocity+1), DOB emails/(zip count+1); exploratory normalization signals, not estimated fraud rates or exact branch shares (branch and velocity windows differ) |
| Missing indicators | Prior/current address, bank tenure, session length, device emails and intended amount |
| Tenure | Prior+current address months (missing if either missing), bank tenure/(current address tenure+1) |
| Identity/contact signals | Number of valid phones, neither phone valid, free email×(1−name similarity), device email count×(1−name similarity) |
| Numeric interactions | Credit limit×income decile, credit limit×age bucket, risk score×income decile; income and age are coarse fields, and these are not actual monetary affordability or age measurements |
| Categorical crosses | Source/device OS, payment/employment, housing/home-phone validity |

There are 29 original predictors and 36 additional predictors (65 total). Exact names are exported in the findings. Feature engineering is a bundled ablation; a successful bundle does not establish that every added variable helps.

## Model preprocessing and imbalance

The original baselines keep training-median numeric imputation with missing indicators and one-hot categorical encoding; logistic numeric inputs are scaled. Engineered logistic additionally one-hot encodes the three categorical crosses.

Native LightGBM keeps numerical NaNs and learns categorical levels from training only; unseen categories become missing. CatBoost receives category strings (unseen strings supported) and numerical medians learned on training only. No transforms learn from future rows. Native missing handling and categorical encoding change together in the native-small ablation.

The undersampled model retains all eligible frauds and at most 20 randomly sampled negatives per positive. Convert sample-posterior odds to population odds using the observed eligible training class prior; this is an approximation assuming class-conditional sampling invariance and does not guarantee calibration. Evaluation prevalence remains unchanged. Recency weights are `2**((training_month−latest_training_month)/2)` normalized to mean one, and month is used only for these training weights and split construction.

Recency weighting also changes effective class composition when monthly prevalence differs, so this ablation tests the practical weighting strategy rather than isolating a pure recency mechanism. Fit times are single-run elapsed measurements on the current host and are affected by load and power state; they are not a controlled speed benchmark.

## Operating policies and selection

Rankings report non-interpolated AP, ROC AUC, Brier score and precision/recall/F1/F2/confusion counts at ceiling-rounded 0.5%, 1%, 2% and 5% review capacities. Equal scores are broken by ascending original row ID. Brier scores of weighted models are descriptive; weighted scores are not claimed to be calibrated.

Four threshold policies are learned from development predictions: maximize F1; maximize F2 (emphasizes recall); maximize recall subject to precision at least 30%; maximize precision subject to recall at least 50%. Ties use the highest threshold. If the precision constraint is infeasible, flag no cases and report that the target is unmet. Targets apply to validation and may fail out of time. Threshold metrics also include MCC, balanced accuracy, false-positive rate, accuracy and flagged fraction; accuracy alone is misleading at this prevalence.

Cross-fitting evaluates month 5 using thresholds learned on month 4, and month 6 using thresholds learned on months 4/5. Selection for F1 and F2 uses their respective cross-fitted macro averages; rank objectives use macro AP and recall at 1% across all three folds. An overall ranking objective and a recall-focused objective are therefore not silently substituted for the user's F1 objective.

Freeze four objective-specific candidates, thresholds learned from pooled months 4/5/6, all needed blend dependencies and source/config/data/software identities before further month-7 evaluation. Evaluate only these candidates plus the three original baselines; model dependencies of selected blends are fitted but not promoted to additional post-hoc candidates. Compare every baseline with its own validation-selected policies, so threshold optimization is not misattributed to modelling improvements.

## Uncertainty and reproducibility

Use 500 paired test-row bootstrap replicates with seed 1027 for differences in AP, recall at 1% review and F1 under fixed validation-selected F1 thresholds. Recompute review quotas within each bootstrap sample and aggregate equal-score groups for AP. Intervals are conditional on fitted models and thresholds, do not include tuning/retraining/time uncertainty, have no multiple-comparison adjustment and cannot repair a consumed holdout. Forward-month variation and ablations remain separate evidence.

Store bulk predictions, models and checkpoints in ignored `outputs/baf-improvements-v1/`. Export small tables, preservation audits and inspected figures to `reports/baf-improvements-v1/`. Reproduction commands are in README. Codex assisted in protocol, implementation, execution and interpretation; acknowledge it and external software in the course submission.

## Adaptive diagnostic amendment

After observing a first-fold improvement for engineered logistic regression, add three development-only fits of raw logistic at C=0.1. This separates regularization effects (raw C=1 to raw C=0.1) from the engineered bundle at fixed C=0.1 on identical folds. The diagnostic uses [a separate configuration](../configs/logistic-ablation.json) and ignored output directory. It is excluded from the main candidate competition and month-7 evaluation. Its adaptive origin and all measured per-month contrasts will be reported; it does not retroactively make the original engineered-logistic comparison a pure feature ablation.

Sources: [official BAF datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf), [LightGBM tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html), [CatBoost categorical features](https://catboost.ai/docs/en/features/categorical-features), [CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/common).
