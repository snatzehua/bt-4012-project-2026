## Context

See proposal.md for motivation and specs/baf-xgboost-rank-ensemble-study/spec.md for the observable contract. The eight frozen studies and their checked findings remain authoritative. Balanced/unweighted pooled macro month-5/6 F2 are 0.32780157809618937/0.3226536849212962. Their consumed-month F2 are 0.3644085570469799/0.3768844221105528. The latest direct-ranking grid failed to beat balanced: best ranker 0.321311, best added binary control 0.323151. These are adaptively reused development observations and exploratory consumed-period evidence.

All settings below are analyst-selected research choices, not approved group or bank operating policy. No independent evaluation month remains. This design changes the available model predictions and tests fixed combinations; it does not change beta to manufacture a gain.

## Goals / Non-Goals

**Goals:** Add a separately identified categorical tree model, distinguish model/representation/blend/policy contributions, and measure actual F2 on identical chronological cases with auditable inference and uncertainty.

**Non-Goals:** Another data download, engineered-feature search, learned meta-model, calibration search, random-split experiment, early stopping on the scored month, quota/floor constraint, losing-setting sweep on month 7, or claim of fresh confirmation. No optional extension belongs to this finite run.

## Decisions

### 1. Preserve the environment and add one verified dependency

Plan `requirements-xgboost.txt` including `-r requirements-improvements.txt` and `xgboost==3.4.1`. The live [PyPI release](https://pypi.org/project/xgboost/3.4.1/) provides Python >=3.12 and macOS ARM64/x86-64 wheels, compatible with the current Python 3.12.9 environment. XGBoost is not currently installed. Stable documentation currently identifies itself as 3.4.2, so it is guidance rather than proof of the installed 3.4.1 contract.

During implementation, install the single pinned package with `--no-deps`, verify its requirements/import/native interface, and compare all existing package versions before/after. Do not upgrade old dependencies or overwrite old fingerprints. The new identity adds XGBoost to its software record and hashes the new dependency file and the two existing requirement files. Preserve all current source/config/software identities recursively.

The new config copies the existing 16 preservation entries, adds direct-ranking config and its original/exported selection pair, yielding 19 entries. Direct-ranking config SHA-256 is `80bcfc3a82c61a5b1fbb6042c92463033c32b4e22f8ed4194377cc4bebf1ef0c`; its selection SHA-256 is `186391cd2936986dfa7f176c29ca01e043a5dde1782dd9073f3e8b91394060e1`. All eight exported selection records must be restorable on a fresh clone without running old fitting stages. Exact old-reference cache reproduction stays in the new destination.

Alternative: modifying an existing study or upgrading the environment would destroy comparison identities. The additive package and recursive guards retain them.

### 2. Fixed native-categorical model grid

Use verified Base v2 `data/raw/baf/Base.csv`, SHA-256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. Use the existing decoded 29 original eligible predictors; exclude fraud_bool, month and constant device_fraud_count. No new feature engineering. Fit the existing NativeInputs adapter only on training rows: categorical levels are training-fitted strings; missing and unseen categories become categorical missing; numeric NaNs remain missing. Keep original column order and stable IDs. This matches the available native tree representation, while old references retain their exact own frozen preprocessing. Cross-family preset differences cannot identify a universal algorithm effect.

Four new IDs cross max_depth {4,6} with scale_pos_weight {1,5}: `xgb_d4_w1`, `xgb_d4_w5`, `xgb_d6_w1`, `xgb_d6_w5`. All share n_estimators=800, learning_rate=0.03, min_child_weight=10, reg_lambda=5, reg_alpha=0, gamma=0, subsample=0.8, colsample_bytree=0.9, tree_method=hist, grow_policy=depthwise, max_bin=256, max_cat_to_onehot=4, max_cat_threshold=64, enable_categorical=True, device=cpu, objective=binary:logistic, base_score=0.5, eval_metric=logloss, random_state=2026 and n_jobs=6. Use no evaluation-set early stopping, class-prior correction or additional sample weights. Weight 5 is a declared training setting, unrelated to beta or an estimated bank cost.

Depth limits express a small controlled capacity comparison; moderate positive weighting tests a different training emphasis without estimating it from future labels. Fixed regularization and subsampling limit this first grid. Native hist categories avoid an unrelated one-hot search. The [official parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html) describes these controls and explicit base_score disabling automatic intercept estimation. These parameter choices are experimental, not claimed optimal or computationally matched to old models.

Save native models as JSON/UBJSON plus a separately hashed adapter/metadata artifact; the [categorical documentation](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) requires this serialization for category information. Retain training feature names/types, parameters, fitted round count, ordered training/evaluation IDs and model/prediction hashes. Scores are finite positive-class logistic outputs, not assumed calibrated; obtain the same score after reloading. Keep weighted scores and percentile/blend scores out of probability-loss comparisons.

Before full fitting, perform two deterministic miniature fits of each of the four presets, eight fits total with shortened rounds counted separately. Verify training-only categories, unknown/missing inference, feature types, class-column orientation, saved/reloaded predictions (absolute/relative tolerance 1e-12), fixed parameters, repeated predictions and unchanged old software. Reject warnings needing investigation, nonfinite predictions, changed schemas/identities and partial caches. Record all actual warnings and fitted rounds; never silently substitute settings.

### 3. Label-free percentile composition and explicit controls

Reuse exact original `baseline_balanced`, `baseline_lgbm` and `native31` reference recipes. Seven raw strategies are these three plus four new XGBoost models. Six percentile strategies are the four XGBoost models and balanced/unweighted references; native31 is a fixed raw reference only.

For a nonempty aligned unlabeled batch of n finite scores, define p_i = average ascending rank(score_i)/n. Ties get their average rank; no label or original-ID tie breaking enters this transform. A singleton has percentile 1. All-equal scores share (n+1)/(2n). Reject nonfinite or empty batches. The transform has no fitted parameters, but depends on the entire current application batch; it is not a per-row calibrated probability. Verify invariance to label changes, strictly increasing score transforms and row permutation with IDs restored. Test explicit ties, singletons, empty/nonfinite rejection and mismatched member IDs/labels.

Twelve blends use each XGBoost percentile member and balanced-LightGBM percentile member with fixed XGBoost weights alpha {0.25,0.5,0.75}: score = alpha*p_xgb + (1-alpha)*p_balanced. Membership and weights are declared before fitting; do not pick members on the evaluation fold. Save both member prediction hashes, transform version, weight and derived score hash. Use stable original row ID only for final operating-rule ties. Percentile single-member controls reveal whether gains arise from batch scale stabilization rather than combining rankings. The direct score average alternative mixes potentially incompatible weight/model score scales; a learned combiner was already tested unsuccessfully and adds another fitted stage.

Total strategies: 7 raw + 6 percentile + 12 blends = 25. Rank-based transformations can leave rankings unchanged while changing the transfer behavior of numerical thresholds. F2, rather than apparent score calibration or AP, determines selection.

### 4. Chronological policies and finite table counts

For each model, predict month m in {4,5,6} after fitting only months 0 through m-1. Every model and adapter sees the same eligible training rows for its month. Models predict identical future IDs/labels; score composition accesses only the current unlabeled batch. Four new presets need 12 full fits; three references supply nine results. With no reference caches, reproduce those nine exact old recipes in the new destination. Twenty-one base results plus 18 percentile and 36 blend results yield 75 score results.

Reuse unchanged pooled maximum-F2 threshold, latest-month maximum-F2 threshold and pooled maximum-F2 rank-fraction policies. Month-5 policy parameters use month 4 only; month-6 policies use months 4/5 (latest uses 5). Threshold ties favor the largest threshold; rank-fraction optimum ties favor the smallest number flagged; operating score ties use stable row ID. Rank fractions use exact rational ceilings. The review fraction is learned from F2, not fixed at 1%. No future labels enter policy fitting.

Export all 25*3*2=150 operating rows, 75 macro summaries and 75 pooled-confusion summaries. Rank by actual macro month-5/6 F2, then recall, precision and deterministic strategy/policy ID. Show both monthly results and pooled-confusion F2, distinguishing the estimands. Retain old pooled benchmarks exactly. Report precision/recall, flagged count/fraction, TP/FP/FN/TN, F1/F3, AP and ROC AUC; no probability-only losses for compositions.

Export 302 named contrasts: 150 against both old pooled references (25 strategies*3 policies*2 references); 50 non-pooled-minus-own-pooled (25*2); 18 percentile-minus-raw at the same policy (6*3); 72 blends-minus-each-percentile-member at the same policy (12*3*2); six depth-6-minus-4 raw comparisons (2 weights*3 policies); six weight-5-minus-1 raw comparisons (2 depths*3 policies). Include zero/self and unsuccessful comparisons rather than removing them. Recompute each difference from its named rows and verify every matched factor.

Export 75 per-month score diagnostics (finite ranges, counts and ties) and 12 label-free diversity rows (4 XGBoost members*3 months) showing Spearman correlation with balanced LightGBM and tied-score fractions. Diversity is descriptive, not a selection objective or proof of useful error complementarity. No new fixed workload quota is needed for these diagnostics.

### 5. Freeze exactly specified transfer roles

Always freeze roles `selected`, `balanced` (raw baseline_balanced/pooled), `unweighted` (raw baseline_lgbm/pooled) and `same_model` (the selected score strategy under pooled policy, including composite strategies). Determine extra roles solely from the selected strategy type:

- Blend: `xgb_member` and `balanced_member`, each percentile member under the selected policy; `raw_xgb`, the same raw XGBoost member under that policy.
- Percentile: `raw_member`, its raw source under the selected policy.
- Raw strategy with a declared percentile counterpart: `rank_member`, that counterpart under the selected policy.
- Raw native31: no extra role because its percentile variant is outside this declared grid.

Do not choose a different control from consumed-period outcomes. These rules yield four to seven role rows, deduplicating base model fits but retaining all role comparisons. Every role learns its final operating rule from eligible months 4/5/6 (latest uses 6). Freeze all selected/control strategy parameters, policy rules, dependencies, source/config/software hashes and development tables before access.

Only dependencies of these frozen roles may reach month 7. This requires at most one new XGBoost model fit on months 0-6, plus two old references (or three if raw native31 wins). There is no losing-XGBoost sweep. Batch percentile transforms use unlabeled month-7 scores only. Freeze hash and access timestamp must reconstruct; a changed/partial cache or late choice is an error.

### 6. Conditional uncertainty and performance interpretation

Use 500 paired fixed-decision row-bootstrap replicates, seed 1027, against every frozen reference/control role. On development, resample months 5 and 6 separately with the same row draws shared across roles; average monthly F2 differences for macro intervals. On consumed month 7, pair decisions on the same sampled applications. Keep models, selection, percentiles and realized operating decisions fixed; do not recompute ranks or reoptimize thresholds inside draws. Tests must independently reproduce unequal-month aggregation, pairing and exported quantiles.

Intervals condition on fitted strategies and realized batches and exclude adaptive search/selection, retraining, batch-composition, temporal/generative uncertainty and multiplicity correction. Measured improvement requires positive F2 differences on the scope claimed. A clear/confirmed gain claim also requires its conditional interval to exclude zero, but significance is not a replacement performance objective. A development gain that fails against an old baseline on transfer is reported as failed transfer; finishing the grid never proves the original goal. Keep both balanced and unweighted comparisons visible and leave the goal unachieved without a transferable measured gain.

### 7. Additive implementation and checked exports

Plan additive `baf/xgboost_inputs.py`, `baf/rank_ensemble_scores.py`, `baf/xgboost_ensemble.py` and `baf/xgboost_ensemble_report.py`; keep all 20 earlier frozen sources unchanged. The new configuration/protocol capture the complete declared grid and identity before full fitting. The runner offers sequential `validate`, `freeze`, `exploratory` stages and a separate report command, with resumable exact-identity checkpoints and no overwrite of frozen selections.

Use `outputs/baf-xgboost-rank-ensembles-v1/` for bulk models/predictions and `reports/baf-xgboost-rank-ensembles-v1/` for checked small exports. Export full development tables, contrasts, diversity and score diagnostics, base/dependency provenance, exact config/freeze/access records, software/data/source identity, all paired draws/intervals, findings and readable inspected operating/diversity figures. An independent audit reconstructs saved models, composed scores, actual decisions/confusion/F2, selection ordering and frozen role inclusion. Record new-fit versus reference-reuse/refit counts and integration fits separately.

README will state pinned install, verified Base access/checksum, eight-record restoration, new-destination reproduction of missing references, sequential commands, seeds, outputs and consumed-holdout limitations. Update AGENTS and project context only as actual implementation/evidence becomes verified. Acknowledge Codex's design, code, execution, tests, figures and interpretation plus official XGBoost documentation/package sources. No public notebook is being copied. Finish full methodological tests, compileall and strict OpenSpec validation before syncing implemented requirements and archiving.

## Risks / Trade-offs

- Repeated development reuse can select noise: preserve the full finite grid, conditional limitations and consumed-period failures; do not claim independent generalization.
- A new tree implementation may be highly correlated with old predictions or worse: retain all raw/member controls and failed comparisons, and select on actual F2.
- Percentile composition changes with the application batch: expose batch inference requirements and condition uncertainty on realized decisions; never call ranks probabilities.
- New package behavior may differ from newer online docs: verify the pinned installed native/save-load contract before full fits without changing frozen dependencies.
- Larger depth/weight can increase false positives or runtime: report actual workload, fitted rounds, warnings and time; keep finite declared settings rather than adjust after outcomes.
- Three policies and 25 strategies add adaptive-selection uncertainty: report every comparison and no multiple-comparison-adjusted confidence claim.

## Migration Plan

Implementation is additive and local. Verify old identities, add/install the single pinned dependency, pass miniature/methodological gates, execute stages sequentially, audit and publish the result, then sync implemented requirements and archive only after verified tasks. If integration fails, retain its diagnostics and stop before full fitting; resolve changed methodology through a revised prefit plan or a new study after results. Reverting new unconsumed artifacts does not authorize altering earlier records; once scored, this study's identity is preserved too.
