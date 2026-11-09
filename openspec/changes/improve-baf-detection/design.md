## Context

See proposal.md for motivation. Base v2 has a fixed local checksum and eight coarse months. Study v1 has consumed month 7. Reuse its input/missing conventions and baseline constructors without modifying frozen source files.

## Goals / Non-Goals

**Goals:** Make the precision/recall trade-off explicit and test genuine ranking gains, nonlinear features and training strategies under a finite reproducible search.

**Non-Goals:** Claim fresh confirmation, assert simultaneous improvement at every threshold, or silently expand to an unlimited hyperparameter search.

## Decisions

- Use development forward months 4/5/6, each trained only on earlier months. Compare 17 specified model presets: three original baselines; native-categorical LightGBM small/31/63-leaf versions; engineered 31/63-leaf versions; engineered versions with positive weights 3/10, two-month recency half-life and 20:1 negative undersampling; engineered logistic regression; CatBoost depth 6/8, engineered depth 6 and engineered depth 6 with positive weight 3. Fixed iteration counts avoid early stopping on outer evaluation labels. Use six CPU threads and resumable source/config-hashed records.
- Engineer row-local velocity ratios/differences, signed logs, tenure/identity indicators and categorical crosses. Do not infer monetary income from deciles, manufacture entity histories or use month as a model predictor. Native categorical vocabularies and CatBoost numeric medians are training-only; unknown categories have explicit fallbacks.
- Preserve original one-hot/median pipelines for exact baselines. Native LightGBM uses missing values and training-only categorical levels. CatBoost handles categorical inputs internally and uses training medians for numerical missingness. Logistic engineered preprocessing scales numeric inputs and one-hot encodes all categorical crosses.
- Undersample training negatives without changing future evaluation prevalence; correct probability odds using the eligible population versus sampled class prior. Recency weights use only training-month age relative to the latest eligible month, normalized to mean one.
- Compare three probability blends with LightGBM fractions 0.25/0.5/0.75, using the best validation-AP LightGBM and CatBoost base presets. This adds no base-model fits. Report this development selection and its optimistic search bias explicitly.
- Select ranking candidates by macro forward AP and recall at 1% review. Select F1/F2 candidates by cross-fitted forward threshold metrics: evaluate month 5 with thresholds from month 4, and month 6 with thresholds from months 4/5. Freeze final thresholds from pooled development predictions for F1, F2, precision at least 30%, and recall at least 50%; policy targets are validation targets, not future guarantees. Fixed review budgets remain 0.5/1/2/5%.
- Freeze selected candidates, blend dependencies, thresholds, all source/config/data/software identities and v1 preservation hashes before further month-7 scoring. Evaluate only selected candidates plus original baselines on month 7. Reuse identical evaluation row IDs and report conditional paired bootstrap differences in AP, fixed-quota recall and F1 (fixed validation-selected thresholds); no retuning from this comparison.
- Report forward performance and month-7 exploratory results separately, ablations, confusion counts, runtime, constraints and software/AI assistance. Include machine-readable metrics and figures under a new report directory.
- After the first forward month showed a gain for engineered logistic regression, add an explicitly adaptive diagnostic: raw logistic with C=0.1 on the same three development folds. This separates regularization (raw C=1 versus raw C=0.1) from features at fixed C=0.1. Keep its three fits in a separate output directory; do not promote it into the predeclared candidate competition or evaluate it on month 7. Report its adaptive origin and measured contrasts.

## Risks / Trade-offs

- Known month-7 outcomes and repeated development selection -> all improvements are exploratory, with paired intervals conditional on fixed fitted models and no search/multiplicity adjustment.
- Sensitive/coarse fields and generated data -> explain features accurately and retain synthetic-data/fairness limitations.
- Increased model capacity overfits -> expose per-month results, fixed presets and model-specific ablations rather than only the best row.
- Weighting distorts score scale -> evaluate rankings at identical quota and thresholds separately; report Brier score without presenting weighted scores as calibrated probabilities.
- Long CPU runtime -> checkpoint each run; never restart a live process solely because a polling timeout occurs.
