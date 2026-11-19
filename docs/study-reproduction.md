# Later-study reproduction recipes

Start with the [code and evidence guide](submission-code-guide.md#reproduce-the-models) for Python, dependencies, data verification and replay isolation. This reference retains study-specific restore steps and ordered stages that previously lived in the main README. It is for reproducing existing recipes; it does not propose further experiments.

## Before running a recipe

Study-specific configurations and measured conclusions are linked from the [report index below](#study-reports). The shell blocks retain the existing numerical recipes; they were inspected during cleanup but full training was not rerun.

Run shell blocks from the repository root in Bash on Linux or WSL, with Python 3.12.9. Install `requirements-xgboost.txt` for the full historical suite. The base and improvements dependency files alone do not cover that suite. Tests may read Base.csv, check restored ancestor records and fit small synthetic models. Full scientific training is a separate stage.

Use a disposable copy with original LF bytes and no current-study outputs. Reporting can overwrite tracked report files, and several runners write receipts to fixed names. For each study restore only its declared ancestors, use fresh test/log paths where supported, and retain the original reports outside the replay copy. Do not remove a selection in the working research checkout. The filesystem-copy blocks below assume an existing pinned environment and verified data in the source checkout; they do not install either prerequisite.

The older `cp -n` blocks leave existing files untouched; an unequal existing record is an error to investigate, not permission to continue. Later Python restore blocks verify all declared records before copying. Caches must be wholly absent or pass the study's individual identity/inference checks; partial or incompatible caches are not interchangeable with missing ones.

The later full-suite gates below are historical recipes, not base-only checks. The current expanded suite also checks descendant declarations and may require all nineteen ancestor selections from `configs/context-feature-ablation.json`, including the target study's original selection. Run broad historical verification in a separate test-only copy with those records restored; keep it separate from a fresh target-study training replay. Do not remove a selection in the research checkout or present scoped tests as satisfying a complete-suite gate. The existing study reproduction receipts document the source/test versions actually checked; full replay of those gates against today's expanded suite has not been verified by this cleanup.

All follow-up month-7 evaluations are exploratory. F2 uses beta=2 without a review quota or precision floor. Run only the chosen study's stages, sequentially, and stop at a failed gate. Historical test totals and resource measurements remain in the linked study reports; these recipes are not new replication receipts.

## Recall-weighted F2 study

Use the existing Python 3.12.9 environment and `requirements-improvements.txt`. From a fresh clone, download Base v2 and install the pinned dependencies first. Restore the small exported frozen records below if the ignored local runs are absent; `cp -n` preserves any existing records. The runner reuses only eligible verified predictions and refits missing predictions into the new output directory. With the historical completed caches, the search needs 30 new development fits and 15 verified reuses; without them it needs 45 fits.

```sh
mkdir -p outputs/baf-study-v1 outputs/baf-improvements-v1
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-improvements-v1/validation_summary.csv outputs/baf-improvements-v1/validation_summary.csv
.venv/bin/python -m pytest -q tests/test_recall_weighted.py
.venv/bin/python -m baf.recall_weighted validate
.venv/bin/python -m baf.recall_weighted freeze
.venv/bin/python -m baf.recall_weighted exploratory
.venv/bin/python -m baf.recall_weighted_report
```

Run these stages in order. Resume unfinished validation with unchanged source/config/software; freeze refuses to overwrite a selection and validation refuses a frozen study. Selected-model transfer on month 7 is exploratory on an already-consumed holdout. New bulk artifacts remain in ignored `outputs/baf-recall-weighted-v1/`; checked findings are exported to `reports/baf-recall-weighted-v1/`. Policy changes, modelling changes, precision/recall, false positives, missed fraud, workload and conditional F2 uncertainty are reported separately. Use a new configuration/output directory for any further search.

## Adaptive F2 extension

Use Python 3.12.9 and `requirements-improvements.txt`. A fresh clone needs the verified Base CSV and these small exported freeze records. Existing files are preserved by `cp -n`; incompatible identities are rejected. Missing model predictions are refitted into extension outputs, including exact original-window recipes, without writing old study directories.

```sh
mkdir -p outputs/baf-study-v1 outputs/baf-improvements-v1 outputs/baf-recall-weighted-v1
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-recall-weighted-v1/selection.json outputs/baf-recall-weighted-v1/selection.json
.venv/bin/python -m pytest -q tests/test_recall_extension.py
.venv/bin/python -m baf.recall_extension validate
.venv/bin/python -m baf.recall_extension freeze
.venv/bin/python -m baf.recall_extension exploratory
.venv/bin/python -m baf.recall_extension_report
```

Run stages sequentially and wait for each to finish. With existing caches the search needs 36 new development fits and 81 verified reuses; without caches it refits 117 base candidate/month combinations. Re-run unfinished validation with unchanged config/source/software to resume. A frozen study refuses further validation or selection overwrite. Final transfer uses only selected dependencies and the two baselines on already-consumed month 7, so it remains exploratory. Ignored bulk outputs go to `outputs/baf-recall-weighted-extension-v1/`; small checked reports go to `reports/baf-recall-weighted-extension-v1/`.

## Chronological score stacking

Use Python 3.12.9 and the pinned `requirements-improvements.txt` environment. Download the verified Base v2 data first. From a fresh clone, restore the four small exported freeze records; missing predictions are fitted into the new output directory, preserving all old studies:

```sh
mkdir -p outputs/baf-study-v1 outputs/baf-improvements-v1 outputs/baf-recall-weighted-v1 outputs/baf-recall-weighted-extension-v1
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-recall-weighted-v1/selection.json outputs/baf-recall-weighted-v1/selection.json
cp -n reports/baf-recall-weighted-extension-v1/selection.json outputs/baf-recall-weighted-extension-v1/selection.json
.venv/bin/python -m pytest -q tests/test_stacking.py
.venv/bin/python -m baf.stacking validate
.venv/bin/python -m baf.stacking freeze
.venv/bin/python -m baf.stacking exploratory
.venv/bin/python -m baf.stacking_report
```

Run stages in order and wait for each to finish. Four fixed base members produce out-of-time scores for months 1-6; the unweighted reference is additionally scored for 4-6. With current caches this needs 12 early base fits and 15 verified reuses; without caches it needs 27 base fits. Twenty-four combiner/control presets produce 72 fits for months 4/5/6. Their preprocessing and training use only earlier out-of-time base scores, and policies use still-earlier out-of-time combiner scores. There are 156 forward policy rows. Score ranks use the unlabeled application batch, so the percentile variants require batch inference.

Unfinished validation resumes with unchanged source/config/software. A frozen study refuses retuning or selection overwrite. Freeze fits any selected final combiner and matched control on out-of-time scores from 1-6 before further month-7 access. Transfer evaluates only frozen selected/control dependencies and both old baselines on consumed month 7. Bulk outputs remain ignored at `outputs/baf-chronological-stacking-v1/`; checked reports go to `reports/baf-chronological-stacking-v1/`. Further tuning needs another configuration/output identity.

## Calibrated F2 batch policies

Use Python 3.12.9 and `requirements-improvements.txt` in `.venv`, then obtain verified Base v2 as described in the code and evidence guide. A fresh clone restores five small exported freeze records; missing score caches are reproduced under the frozen recipes in the new destination:

```sh
mkdir -p outputs/baf-study-v1 outputs/baf-improvements-v1 outputs/baf-recall-weighted-v1 outputs/baf-recall-weighted-extension-v1 outputs/baf-chronological-stacking-v1
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-recall-weighted-v1/selection.json outputs/baf-recall-weighted-v1/selection.json
cp -n reports/baf-recall-weighted-extension-v1/selection.json outputs/baf-recall-weighted-extension-v1/selection.json
cp -n reports/baf-chronological-stacking-v1/selection.json outputs/baf-chronological-stacking-v1/selection.json
.venv/bin/python -m pytest -q tests/test_calibrated_policy.py
.venv/bin/python -m baf.calibrated_policy validate
.venv/bin/python -m baf.calibrated_policy freeze
.venv/bin/python -m baf.calibrated_policy diagnostics
.venv/bin/python -m baf.calibrated_policy exploratory
.venv/bin/python -m baf.calibrated_policy_report
```

Run stages sequentially. Current caches supply nine fixed source predictions without new base/combiner fits; dependencies are separately verified. With no caches, up to 27 frozen-recipe base results and three fits of the one fixed stack are needed. These go only to `outputs/baf-calibrated-f2-policy-v1/`, never old study directories. Unfinished validation resumes with unchanged sources/config/software; frozen selection refuses retuning. Calibration uses only earlier out-of-time predictions, and the chooser uses the current unlabeled batch. The surrogate is a ratio of estimated counts, not exact expected or achieved F2. Equal calibrated probabilities use raw score and then stable ID to break ties.

Freeze stores the selected final calibrator and any necessary final stack before further month-7 access. The six label-aware development prefix optima are generated only after freeze and excluded from selection; no month-7 hindsight optimum is computed. Consumed-period transfer evaluates only the selected policy, its same-model pooled comparator and both old raw baselines, not every losing calibration policy. Conditional bootstrap intervals keep realized batch decisions fixed and exclude model/calibration/selection/time uncertainty. Checked reports go to `reports/baf-calibrated-f2-policy-v1/` after all stages pass.

## Focal-loss score learning

Use Python 3.12.9 and the pinned `requirements-improvements.txt` environment, and download verified Base v2 as described in the code and evidence guide. A fresh clone restores six small exported selection records; absent scores are reproduced in the new destination under frozen recipes:

```sh
mkdir -p outputs/baf-study-v1 outputs/baf-improvements-v1 outputs/baf-recall-weighted-v1 outputs/baf-recall-weighted-extension-v1 outputs/baf-chronological-stacking-v1 outputs/baf-calibrated-f2-policy-v1
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-recall-weighted-v1/selection.json outputs/baf-recall-weighted-v1/selection.json
cp -n reports/baf-recall-weighted-extension-v1/selection.json outputs/baf-recall-weighted-extension-v1/selection.json
cp -n reports/baf-chronological-stacking-v1/selection.json outputs/baf-chronological-stacking-v1/selection.json
cp -n reports/baf-calibrated-f2-policy-v1/selection.json outputs/baf-calibrated-f2-policy-v1/selection.json
.venv/bin/python -m pytest -q tests/test_focal_loss.py
.venv/bin/python -m baf.focal_study validate
.venv/bin/python -m baf.focal_study freeze
.venv/bin/python -m baf.focal_study exploratory
.venv/bin/python -m baf.focal_report
```

Run commands sequentially and wait for completion. Validation checkpoints each saved model/adapter, raw margin, sigmoid score, numeric trace and prediction population; resuming requires unchanged code/config/software. Four miniature integration fits verify custom/built-in ordinary-loss equivalence before the 24 full development fits; they are reported separately. The loss callback receives raw margins, and inference applies exactly one sigmoid. Focal negative/small Hessians receive an explicit 1e-6 curvature floor, with per-call statistics. This is approximate curvature optimization; ordinary gamma=0 controls use actual logistic Hessians. Objective calls and fitted tree counts are recorded separately because a boosting attempt may add no tree. Scores are not assumed calibrated.

Model/preprocessing fitting uses strictly earlier months; month-5 policies use month 4 predictions and month-6 policies use earlier 4/5 or 5. Selection uses actual macro F2. Freeze records selected/control recipes and policies before further consumed-month access. Transfer includes only selected/required ordinary controls, same-model pooled policy and both old pooled references; losing focal presets and month-7 hindsight thresholds are not swept. This run required two exploratory baseline reuses and no new month-7 model fits. Selected F2 remains **0.364409**, below the unweighted reference's **0.376884**. Bootstrap draws keep realized decisions fixed and exclude model-selection, retraining and time uncertainty. Outputs remain ignored in `outputs/baf-focal-loss-v1/`; the checked report includes 106 contrasts, 14,400 numerical trace rows, inspected figures and a [verification record](../reports/baf-focal-loss-v1/verification.md). All six earlier studies remain frozen, and further tuning needs another identity.

## Direct fraud-ranking study

Use Python 3.12.9, the pinned `requirements-improvements.txt` environment and verified Base v2. Restore the six exported selection records using the focal-study instructions in the code and evidence guide, then restore the seventh and run stages sequentially:

```sh
mkdir -p outputs/baf-focal-loss-v1
cp -n reports/baf-focal-loss-v1/selection.json outputs/baf-focal-loss-v1/selection.json
.venv/bin/python -m pytest -q tests/test_direct_ranking.py
.venv/bin/python -m baf.direct_ranking validate
.venv/bin/python -m baf.direct_ranking freeze
.venv/bin/python -m baf.direct_ranking exploratory
.venv/bin/python -m baf.direct_ranking_report
```

Wait for each command to finish. Validation checkpoints models, adapters, training/group metadata, raw predictions and exact identities; resuming requires unchanged code/config/software. Four miniature fits verify installed initialization/repeatability before the 18 full fits. The group seed and model/objective seed are 2026; paired bootstrap seed is 1027 with 500 draws. Current reference caches are verified before reuse; absent scores are refitted under frozen recipes only in `outputs/baf-direct-ranking-v1/`. Never rerun earlier validate/freeze commands to populate old directories. After selection freezes, only its selected/matched dependencies and old references reach already-consumed month 7, with no losing-preset sweep or hindsight threshold optimum. Conditional intervals keep decisions fixed and exclude selection, retraining, temporal/generative uncertainty and multiplicity correction. Checked exports go to `reports/baf-direct-ranking-v1/`; all seven earlier studies remain frozen.

## XGBoost and rank ensembles

Use Python 3.12.9 and the pinned requirements-xgboost.txt environment, verified Base v2 and the same existing dependencies. For the existing .venv, `uv pip install --python .venv/bin/python --no-deps xgboost==3.4.1` adds only the new package; `uv pip check --python .venv/bin/python` verifies compatibility. A fresh environment can use `uv pip install --python .venv/bin/python -r requirements-xgboost.txt`. Existing versions must match the frozen identities. Restore all eight small exported selections, then run commands sequentially:

```sh
mkdir -p outputs/baf-study-v1 outputs/baf-improvements-v1 outputs/baf-recall-weighted-v1 outputs/baf-recall-weighted-extension-v1 outputs/baf-chronological-stacking-v1 outputs/baf-calibrated-f2-policy-v1 outputs/baf-focal-loss-v1 outputs/baf-direct-ranking-v1
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-recall-weighted-v1/selection.json outputs/baf-recall-weighted-v1/selection.json
cp -n reports/baf-recall-weighted-extension-v1/selection.json outputs/baf-recall-weighted-extension-v1/selection.json
cp -n reports/baf-chronological-stacking-v1/selection.json outputs/baf-chronological-stacking-v1/selection.json
cp -n reports/baf-calibrated-f2-policy-v1/selection.json outputs/baf-calibrated-f2-policy-v1/selection.json
cp -n reports/baf-focal-loss-v1/selection.json outputs/baf-focal-loss-v1/selection.json
cp -n reports/baf-direct-ranking-v1/selection.json outputs/baf-direct-ranking-v1/selection.json
.venv/bin/python -m pytest -q tests/test_xgboost_ensemble.py
.venv/bin/python -m baf.xgboost_ensemble validate
.venv/bin/python -m baf.xgboost_ensemble freeze
.venv/bin/python -m baf.xgboost_ensemble exploratory
.venv/bin/python -m baf.xgboost_ensemble_report
```

Wait for every stage to finish. Missing old reference caches are reproduced under frozen recipes only in outputs/baf-xgboost-rank-ensembles-v1/, without rerunning earlier studies' fitting stages. Model checkpoints save native UBJSON, training-fitted categorical adapters, feature/parameter identities, fitted rounds, warnings and predictions. Exact unchanged source/config/software is required for resumption. Seeds are 2026 for models and 1027 for 500 paired uncertainty draws; models use at most six threads. Percentiles use average ascending rank/n within the current unlabeled batch, and blends have fixed weights 0.25/0.5/0.75. These representations require batch inference and are not calibrated fraud probabilities.

Freeze includes exactly the selected strategy, same-strategy pooled policy, both old pooled references and the specified raw/percentile/member controls. Only those dependencies reach already-consumed month 7, requiring at most one new XGBoost final fit. No losing-setting sweep or hindsight cutoff is permitted. Exported 302 contrasts distinguish model, representation, blending and policy effects. Conditional draws keep realized batches and decisions fixed, excluding adaptive selection, retraining, batch-composition, temporal/generative uncertainty and multiplicity correction. Checked reports go to reports/baf-xgboost-rank-ensembles-v1/ after actual verification.

## Fixed raw-score and log-odds fusion

Use Python 3.12.9 and the unchanged pinned `requirements-xgboost.txt`; a fresh environment can use `uv venv --python 3.12.9 .venv` followed by `uv pip install --python .venv/bin/python -r requirements-xgboost.txt`. Run `uv pip check --python .venv/bin/python` and verify frozen versions before reuse. Download and verify Base v2 using the data instructions in the code and evidence guide. Restore all nine small selections, then run stages sequentially from the repository root:

```sh
mkdir -p outputs/baf-study-v1 outputs/baf-improvements-v1 outputs/baf-recall-weighted-v1 outputs/baf-recall-weighted-extension-v1 outputs/baf-chronological-stacking-v1 outputs/baf-calibrated-f2-policy-v1 outputs/baf-focal-loss-v1 outputs/baf-direct-ranking-v1 outputs/baf-xgboost-rank-ensembles-v1
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-recall-weighted-v1/selection.json outputs/baf-recall-weighted-v1/selection.json
cp -n reports/baf-recall-weighted-extension-v1/selection.json outputs/baf-recall-weighted-extension-v1/selection.json
cp -n reports/baf-chronological-stacking-v1/selection.json outputs/baf-chronological-stacking-v1/selection.json
cp -n reports/baf-calibrated-f2-policy-v1/selection.json outputs/baf-calibrated-f2-policy-v1/selection.json
cp -n reports/baf-focal-loss-v1/selection.json outputs/baf-focal-loss-v1/selection.json
cp -n reports/baf-direct-ranking-v1/selection.json outputs/baf-direct-ranking-v1/selection.json
cp -n reports/baf-xgboost-rank-ensembles-v1/selection.json outputs/baf-xgboost-rank-ensembles-v1/selection.json
.venv/bin/python -m pytest -q tests/test_score_fusion.py
.venv/bin/python -m baf.score_fusion_study validate
.venv/bin/python -m baf.score_fusion_study freeze
.venv/bin/python -m baf.score_fusion_study exploratory
.venv/bin/python -m baf.score_fusion_report
```

Wait for every stage to finish. Verify complete existing model/prediction caches before copying; reject partial or incompatible ones. Wholly absent caches are reproduced only in ignored `outputs/baf-xgboost-score-fusion-v1/` under exact frozen recipes, without rerunning older studies' fitting commands. Existing reference development predictions have no saved models: separately counted exact-recipe verification fits must match those predictions and save the missing models. A wholly absent reference prediction/model pair is fitted together once. Actual fits, reuses, warnings, timings and provenance remain recorded.

Freeze includes the actual winner, its pooled-policy control, both original pooled references and predetermined member/representation controls. Only their dependencies may reach already-consumed month 7. Never sweep losing settings or compute a hindsight cutoff. Exact source/config/software is required to resume; frozen selections cannot be overwritten. Checked small exports go to `reports/baf-xgboost-score-fusion-v1/`. Conditional uncertainty excludes adaptive search/selection, retraining, temporal/generative uncertainty and multiplicity correction. Consumed month 7 cannot provide independent confirmation, and completing a search alone does not achieve the performance goal. The [verification record](../reports/baf-xgboost-score-fusion-v1/verification.md) covers all 366 operating rows, 866 contrasts, 165 derived dependency records, 28 interval summaries and all 1,500 paired draws. The figure was inspected. The run reused 12 development XGBoost models and nine reference predictions, fitted nine absent reference models for verification, and needed one final XGBoost fit plus two final reference reuses. All ten studies are now frozen; further tuning requires another study identity.

## Additive spline-logistic attempt (stopped at convergence gate)

Use Python 3.12.9 and the unchanged pinned environment: `uv venv --python 3.12.9 .venv`, then `uv pip install --python .venv/bin/python -r requirements-xgboost.txt` and `uv pip check --python .venv/bin/python`. Download Base v2 using the data-access instructions in the code and evidence guide and check its declared checksum. The commands below reproduce the stopped attempt under its original recipe. They are not an instruction to bypass the failed gate. Restore the ten small frozen selections and run stages sequentially from the repository root:

```bash
mkdir -p outputs/baf-calibrated-f2-policy-v1 outputs/baf-chronological-stacking-v1 outputs/baf-direct-ranking-v1 outputs/baf-focal-loss-v1 outputs/baf-improvements-v1 outputs/baf-recall-weighted-extension-v1 outputs/baf-recall-weighted-v1 outputs/baf-study-v1 outputs/baf-xgboost-rank-ensembles-v1 outputs/baf-xgboost-score-fusion-v1
cp -n reports/baf-calibrated-f2-policy-v1/selection.json outputs/baf-calibrated-f2-policy-v1/selection.json
cp -n reports/baf-chronological-stacking-v1/selection.json outputs/baf-chronological-stacking-v1/selection.json
cp -n reports/baf-direct-ranking-v1/selection.json outputs/baf-direct-ranking-v1/selection.json
cp -n reports/baf-focal-loss-v1/selection.json outputs/baf-focal-loss-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-recall-weighted-extension-v1/selection.json outputs/baf-recall-weighted-extension-v1/selection.json
cp -n reports/baf-recall-weighted-v1/selection.json outputs/baf-recall-weighted-v1/selection.json
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-xgboost-rank-ensembles-v1/selection.json outputs/baf-xgboost-rank-ensembles-v1/selection.json
cp -n reports/baf-xgboost-score-fusion-v1/selection.json outputs/baf-xgboost-score-fusion-v1/selection.json
.venv/bin/python -m baf.spline_study miniature
.venv/bin/python -m baf.spline_study validate
.venv/bin/python -m baf.spline_study freeze
.venv/bin/python -m baf.spline_study exploratory
.venv/bin/python -m baf.spline_report
```

The miniature gate checks convergence, deterministic repeats and saved inference on a synthetic fixture, separately from BAF fits. Full-data convergence warnings or invalid inference stop execution before selection; do not silently change iterations, C, solver or representation. Failures are retained under the new output root and cannot be treated as a completed search. Any protocol remedy needs a separate study identity. Freeze is immutable; only the predetermined five/seven roles can reach consumed month 7. Never delete an existing selection to rerun tuning.

## Newton spline-logistic successor (complete)

Use Python 3.12.9 and the environment/data setup in the code and evidence guide. From the repository root restore the ten small selections, then run stages sequentially; do not run freeze unless validate has completed successfully:

```bash
mkdir -p outputs/baf-calibrated-f2-policy-v1 outputs/baf-chronological-stacking-v1 outputs/baf-direct-ranking-v1 outputs/baf-focal-loss-v1 outputs/baf-improvements-v1 outputs/baf-recall-weighted-extension-v1 outputs/baf-recall-weighted-v1 outputs/baf-study-v1 outputs/baf-xgboost-rank-ensembles-v1 outputs/baf-xgboost-score-fusion-v1
cp -n reports/baf-calibrated-f2-policy-v1/selection.json outputs/baf-calibrated-f2-policy-v1/selection.json
cp -n reports/baf-chronological-stacking-v1/selection.json outputs/baf-chronological-stacking-v1/selection.json
cp -n reports/baf-direct-ranking-v1/selection.json outputs/baf-direct-ranking-v1/selection.json
cp -n reports/baf-focal-loss-v1/selection.json outputs/baf-focal-loss-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-recall-weighted-extension-v1/selection.json outputs/baf-recall-weighted-extension-v1/selection.json
cp -n reports/baf-recall-weighted-v1/selection.json outputs/baf-recall-weighted-v1/selection.json
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-xgboost-rank-ensembles-v1/selection.json outputs/baf-xgboost-rank-ensembles-v1/selection.json
cp -n reports/baf-xgboost-score-fusion-v1/selection.json outputs/baf-xgboost-score-fusion-v1/selection.json
.venv/bin/python -m baf.newton_spline_audit miniature
.venv/bin/python -m baf.newton_spline_audit validate
.venv/bin/python -m baf.newton_spline_audit freeze
.venv/bin/python -m baf.newton_spline_audit exploratory
.venv/bin/python -m baf.newton_spline_audit_report
```

Outputs use ignored `outputs/baf-newton-spline-logistic-v1/` and tracked `reports/baf-newton-spline-logistic-v1/`. Development declares 36 new fits, nine checked reference base results and three derived fusions: 48 total scores, 45 selectable results, 90 forward rows and 237 matched contrasts. All original predictors, earlier-only preprocessing and policies are retained. The fixed representation changes smooth columns to additive cubic bases with observed-training knots, without interactions. Model seed is 2026, threads six, bootstrap seed 1027 and 500 paired fixed-decision draws per evaluation month.

Wholly absent required reference caches are reproduced under their original frozen recipes only in this destination; partial/incompatible required caches are rejected. Ignored stopped L-BFGS model caches are optional: verify any available artifacts against the recorded stopped hashes, snapshot their presence and retain them. A fresh clone need not rerun the failed attempt; unavailable old-solver training-objective diagnostics are disclosed. They do not enter F2 selection. New predictions always require genuine Newton fits.

A full-data gate failure retains numerical evidence and stops completion without changing the recipe. A successful development run freezes only the predetermined five/seven transfer roles; month 7 remains already consumed and cannot provide independent confirmation. Completing this study alone does not achieve the broader performance goal. All settings remain analyst-selected and open to group review.

## Spline-score fusion study

Use Python 3.12.9 and the unchanged pinned requirements in `requirements.txt`, `requirements-improvements.txt` and `requirements-xgboost.txt`; the existing environment suffices. Download Base v2 to `data/raw/baf/Base.csv` and check SHA-256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. Model seed 2026, six threads, bootstrap seed 1027, 500 paired fixed-decision draws per month. On a fresh clone restore these eleven small frozen selections before running the stages in order:

```bash
mkdir -p outputs/baf-calibrated-f2-policy-v1 outputs/baf-chronological-stacking-v1 outputs/baf-direct-ranking-v1 outputs/baf-focal-loss-v1 outputs/baf-improvements-v1 outputs/baf-recall-weighted-extension-v1 outputs/baf-recall-weighted-v1 outputs/baf-study-v1 outputs/baf-xgboost-rank-ensembles-v1 outputs/baf-xgboost-score-fusion-v1 outputs/baf-newton-spline-logistic-v1
cp -n reports/baf-calibrated-f2-policy-v1/selection.json outputs/baf-calibrated-f2-policy-v1/selection.json
cp -n reports/baf-chronological-stacking-v1/selection.json outputs/baf-chronological-stacking-v1/selection.json
cp -n reports/baf-direct-ranking-v1/selection.json outputs/baf-direct-ranking-v1/selection.json
cp -n reports/baf-focal-loss-v1/selection.json outputs/baf-focal-loss-v1/selection.json
cp -n reports/baf-improvements-v1/selection.json outputs/baf-improvements-v1/selection.json
cp -n reports/baf-recall-weighted-extension-v1/selection.json outputs/baf-recall-weighted-extension-v1/selection.json
cp -n reports/baf-recall-weighted-v1/selection.json outputs/baf-recall-weighted-v1/selection.json
cp -n reports/baf-study-v1/selection.json outputs/baf-study-v1/selection.json
cp -n reports/baf-xgboost-rank-ensembles-v1/selection.json outputs/baf-xgboost-rank-ensembles-v1/selection.json
cp -n reports/baf-xgboost-score-fusion-v1/selection.json outputs/baf-xgboost-score-fusion-v1/selection.json
cp -n reports/baf-newton-spline-logistic-v1/selection.json outputs/baf-newton-spline-logistic-v1/selection.json
.venv/bin/python -m baf.spline_fusion_study initialize
.venv/bin/python -m pytest -q tests/test_spline_fusion.py
.venv/bin/python -m baf.spline_fusion_study validate
.venv/bin/python -m baf.spline_fusion_report audit
.venv/bin/python -m baf.spline_fusion_study freeze
.venv/bin/python -m baf.spline_fusion_study exploratory
PYTHONPATH=. .venv/bin/python reports/baf-spline-score-fusion-v1/independent_score_audit.py
.venv/bin/python -m baf.spline_fusion_report report
```

Wait for every stage to finish. Existing required caches are verified individually, including saved inference and Newton objective/gradient/fallback evidence. Local development verified twelve Newton and nine tree base reuses and zero new fits. Wholly absent required caches reproduce only their exact frozen recipes in the new destination, up to twelve Newton and nine tree development fits; reject partial/incompatible caches. Optional historical bulk outputs are not required on fresh clones, but their initial presence is captured and available files must remain unchanged. Never rerun unrelated old full-grid loaders or the stopped L-BFGS attempt. Composition fixtures fit zero models. Two tiny reference-cache recipe fixtures were tested separately (two fits per targeted/full-suite run, four test-only fits total), outside the full-study fit counts.

New destinations are ignored `outputs/baf-spline-score-fusion-v1/` and exported `reports/baf-spline-score-fusion-v1/`. The independent audit checked all 72 development score artifacts and 433 contrasts before freeze; the supplementary auditor checked all 85 development/transfer scores, including 15 coefficient-to-sigmoid reconstructions, and all contrast factor sets. Only the predeclared five/seven/eight/ten winner-type roles may transfer, with deduplicated work and at most three new Newton fits plus required exact reference bases. Freeze/access records remain immutable. Findings and conditional intervals must disclose adaptive source/C choice and repeated search; the consumed month cannot restore independent confirmation. Preserve all eleven completed identities and the stopped attempt. Settings remain analyst-selected.

## Fixed spline interactions

Use Python 3.12.9, the unchanged pinned requirements-xgboost.txt environment and the verified Base v2 access/checksum instructions in the code and evidence guide for a clean reproduction in a fresh checkout. In a fresh clone, restore all twelve small original selection records before initializing; the new config's mandatory hashes identify the exact required files. Reproduce only wholly absent required individual model recipes in the new destination; partial/incompatible sets fail. Do not rerun or modify any historical grid. Keep this completed local run and its published reports intact: do not rerun fitting, audit-writing, freeze or reporting stages over it, because frozen selections reference the exact audit/report identities. The ordered commands below are for a clean reproduction. A retained numerical failure must not be bypassed.

Restore exactly those twelve selections, refusing any changed existing record:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
from baf.common import read_config, digest
c = read_config('configs/interaction-spline-logistic.json')
records = {p: h for p, h in c['preserve_sha256'].items()
           if p.startswith('outputs/') and p.endswith('/selection.json')}
assert len(records) == 12
for name, expected in records.items():
    target = Path(name)
    source = Path('reports') / target.relative_to('outputs')
    assert digest(source) == expected, source
    if target.exists():
        assert digest(target) == expected, target
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
PY
```

```bash
.venv/bin/python -m pytest -q tests --junitxml=reports/baf-interaction-spline-logistic-v1/tests.xml > reports/baf-interaction-spline-logistic-v1/tests.log 2>&1
.venv/bin/python -m baf.interaction_study record_checks
.venv/bin/python -m baf.interaction_study miniature
.venv/bin/python -m baf.interaction_study preflight
.venv/bin/python -m baf.interaction_study prepare_references
.venv/bin/python -m baf.interaction_study validate
.venv/bin/python -m baf.interaction_report audit
.venv/bin/python -m baf.interaction_study freeze
.venv/bin/python -m baf.interaction_study exploratory
.venv/bin/python -m baf.interaction_report report
```

All declared execution, reporting and completion-audit stages have finished. `freeze` requires the complete independent development audit and refuses an existing selection. Transfer uses only the frozen six/nine roles and cache/type-specific fit bounds. Reporting requires the full passing test receipt for the exact frozen source/config/software identity. Independent two-pass moments use the host's actual `longdouble` width (64 bits on this Mac); this does not imply extra precision over float64.

## Fixed interaction-score fusion

On a clean reproduction, wholly absent required individual caches use only exact frozen recipes in the new destination; partial/incompatible caches fail. The mandatory exported eight-fit miniature receipt is checked, and ignored historical miniature models are verified if present without being recreated. Four genuine tiny cache-format fits per full test invocation are counted separately from production fits and historical tests. Required new production fitting must pass the actual-host predictor-only preflight for all months-0–6 training rows and the unchanged numerical gates.

Use Python 3.12.9, the unchanged pinned requirements files and the Base v2 data-access/checksum instructions in the code and evidence guide. On a fresh clone, restore exactly thirteen original selection records, rejecting changed existing records:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
from baf.common import read_config, digest
c = read_config('configs/interaction-score-fusion.json')
records = {p: h for p, h in c['preserve_sha256'].items()
           if p.startswith('outputs/') and p.endswith('/selection.json')}
assert len(records) == 13
for name, expected in records.items():
    target = Path(name)
    source = Path('reports') / target.relative_to('outputs')
    assert digest(source) == expected, source
    if target.exists():
        assert digest(target) == expected, target
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
PY
```

Run these stages in order only in a clean reproduction destination. After a selection exists, do not overwrite its audit-writing, freeze or reporting stages; frozen records reference their exact hashes. Preserve any failure evidence and do not relax a recipe or gate after outcomes.

```bash
mkdir -p reports/baf-interaction-score-fusion-v1
.venv/bin/python -m pytest -q tests --junitxml=reports/baf-interaction-score-fusion-v1/tests.xml > reports/baf-interaction-score-fusion-v1/tests.log 2>&1
.venv/bin/python -m baf.interaction_fusion_study record_checks
.venv/bin/python -m baf.interaction_fusion_study preflight
.venv/bin/python -m baf.interaction_fusion_study prepare_development
.venv/bin/python -m baf.interaction_fusion_study validate
.venv/bin/python -m baf.interaction_fusion_study freeze
.venv/bin/python -m baf.interaction_fusion_study exploratory
.venv/bin/python -m baf.interaction_fusion_report report
```

`validate` includes the full independent development audit before freeze. A separate audit command, `.venv/bin/python -m baf.interaction_fusion_report audit_development`, is available only before selection. The complete development outputs are 156 physical scores, 153 selectable scores, 306 forward rows, 153 macro/pooled/score summaries, 1,386 contrasts, 33 clipping rows and 48 diversity rows. Transfer retains only six/eleven/fourteen predetermined winner-type roles with exact deduplicated dependencies and tighter cache-specific fit bounds. Source/score provenance, independent features/inference/gradients, operating decisions/metrics, all paired draws and figure values have passed independent checks. Month 7 remains consumed and cannot provide independent confirmation; completing the finite search alone does not establish improvement.

## Unweighted-reference fusion

Use Python 3.12.9 and the unchanged `requirements.txt`, `requirements-improvements.txt` and `requirements-xgboost.txt`; all 27 pins were verified. Obtain Base v2 using the data-access instructions in the code and evidence guide, with SHA256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. Model seed is 2026, numerical threads six, bootstrap seed 1027 with 500 paired draws per month. Required individual caches reproduce only exact frozen recipes when wholly absent; partial or changed sets fail. No unrelated historical grid is required. Actual-host predictor-only preflight checked 903157 rows/259 features and the unchanged 70%-of-host-RAM allowance.

Preserve existing frozen audit/selection/report evidence. To reproduce, work in a disposable filesystem copy excluding ignored outputs, then move that copy's tracked report export aside before running commands. The original checkout and report remain untouched. The following uses the already-verified environment/data; a new environment instead needs the pinned setup above:

```bash
set -e
baf_original_root="$PWD"
baf_replay_root="$(mktemp -d "${TMPDIR:-/tmp}/baf-fusion-replay.XXXXXX")"
rsync -a --exclude=.git --exclude=.venv --exclude=data --exclude=outputs --exclude=.pytest_cache --exclude=__pycache__ ./ "$baf_replay_root/"
ln -s "$baf_original_root/.venv" "$baf_replay_root/.venv"
mkdir -p "$baf_replay_root/data/raw/baf" "$baf_replay_root/replay-evidence"
ln -s "$baf_original_root/data/raw/baf/Base.csv" "$baf_replay_root/data/raw/baf/Base.csv"
mv "$baf_replay_root/reports/baf-unweighted-reference-fusion-v1" "$baf_replay_root/replay-evidence/"
cd "$baf_replay_root"
```

Restore exactly fourteen ancestor selections from their checked exports; this rejects changed existing records before copying missing ones:

```bash
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import shutil
from baf.common import read_config, digest
c = read_config('configs/unweighted-reference-fusion.json')
pairs = [(Path(p), Path(p.replace('outputs/', 'reports/', 1)), h)
         for p, h in c['preserve_sha256'].items()
         if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(pairs) == 14
for original, exported, expected in pairs:
    assert exported.is_file() and digest(exported) == expected, str(exported)
    if original.exists():
        assert digest(original) == expected, str(original)
for original, exported, expected in pairs:
    if not original.exists():
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(exported, original)
PYTHON
```

Then run the ordered stages. `validate` includes this grid's complete independent development audit before freeze. The full methodological suite includes four genuine tiny absent-cache formats and all 120 winner/policy cases; source miniature history is verified without production miniature refitting.

```bash
set -e
.venv/bin/python -m pytest -q --junitxml=reports/baf-unweighted-reference-fusion-v1/tests.xml
.venv/bin/python -m baf.unweighted_fusion_study record_checks
.venv/bin/python -m baf.unweighted_fusion_study preflight
.venv/bin/python -m baf.unweighted_fusion_study prepare_development
.venv/bin/python -m baf.unweighted_fusion_study validate
.venv/bin/python -m baf.unweighted_fusion_study freeze
.venv/bin/python -m baf.unweighted_fusion_study exploratory
.venv/bin/python -m baf.unweighted_fusion_report report
```

## Bagged LightGBM seed study

For reproduction use Python 3.12.9 and unchanged `requirements.txt`, `requirements-improvements.txt` and `requirements-xgboost.txt` (27 dependency pins). Obtain Base v2 with the data-access instructions in the code and evidence guide, verifying SHA256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. Model seed is 2026, threads six, bootstrap seed 1027 and 500 paired fixed-decision draws per month. Keep existing frozen evidence intact; run in a disposable copy excluding ignored outputs and move only that copy's current-study report export aside. The following links the already-verified environment and read-only dataset; a new environment instead uses the pinned installation above:

```bash
set -e
baf_original_root="$PWD"
baf_replay_root="$(mktemp -d "${TMPDIR:-/tmp}/baf-bagged-replay.XXXXXX")"
rsync -a --exclude=.git --exclude=.venv --exclude=data --exclude=outputs --exclude=.pytest_cache --exclude=__pycache__ ./ "$baf_replay_root/"
ln -s "$baf_original_root/.venv" "$baf_replay_root/.venv"
mkdir -p "$baf_replay_root/data/raw/baf" "$baf_replay_root/replay-evidence"
ln -s "$baf_original_root/data/raw/baf/Base.csv" "$baf_replay_root/data/raw/baf/Base.csv"
mv "$baf_replay_root/reports/baf-bagged-lightgbm-seeds-v1" "$baf_replay_root/replay-evidence/"
mkdir -p "$baf_replay_root/reports/baf-bagged-lightgbm-seeds-v1"
cd "$baf_replay_root"
```

Restore exactly fifteen ancestor selections; reject changed records before copying missing ones:

```bash
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import shutil
from baf.common import read_config, digest
c = read_config('configs/bagged-lightgbm-seeds.json')
pairs = [(Path(p), Path(p.replace('outputs/', 'reports/', 1)), h)
         for p, h in c['preserve_sha256'].items()
         if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(pairs) == 15
for original, exported, expected in pairs:
    assert exported.is_file() and digest(exported) == expected, str(exported)
    if original.exists():
        assert digest(original) == expected, str(original)
for original, exported, expected in pairs:
    if not original.exists():
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(exported, original)
PYTHON
```

Run stages sequentially. `validate` includes the complete grid-specific independent development audit before freeze; the explicit `audit_development` entry point is available before selection. The methodological suite verifies all 45 winner/policy cases and includes two genuine tiny new-format fits per full invocation. `preflight` checks preserved source-miniature ancestry without rerunning old gates; `miniature` runs nine new synthetic fits.

```bash
set -e
.venv/bin/python -m pytest -q tests --junitxml=reports/baf-bagged-lightgbm-seeds-v1/tests.xml
.venv/bin/python -m baf.bagged_boosting record_checks
.venv/bin/python -m baf.bagged_boosting preflight
.venv/bin/python -m baf.bagged_boosting miniature
.venv/bin/python -m baf.bagged_boosting prepare_development
.venv/bin/python -m baf.bagged_boosting validate
.venv/bin/python -m baf.bagged_boosting freeze
.venv/bin/python -m baf.bagged_boosting exploratory
.venv/bin/python -m baf.bagged_boosting_report report
```

Wholly absent references can require 15 development reproductions in addition to 18 new bagged fits. Frozen transfer permits seven reference-winner or fourteen new-winner roles, with at most eleven bases/seventeen scores/six new bagged fits and tighter cache/type-specific bounds. Missing final references can require two additive and three tree reproductions, counted separately. Numerical/resource failures retain evidence and stop completion. Conditional row intervals exclude training-seed/retraining, repeated search/selection, temporal/generative uncertainty and multiplicity correction. Further model or policy tuning requires a separate study.

## Bagged-ensemble additions to the anchor (complete)

For reproduction use Python 3.12.9 and the same 27 dependency pins in the three requirements files. Verify Base v2 SHA256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809` using the data-access instructions in the code and evidence guide. Model seed is 2026, threads six, bootstrap seed 1027 and 500 paired fixed-decision draws per month. Keep existing frozen evidence intact. Set up a disposable copy without ignored outputs; the following links the pinned environment and read-only dataset. A new environment instead uses the pinned installation above:

```bash
set -e
baf_original_root="$PWD"
baf_replay_root="$(mktemp -d "${TMPDIR:-/tmp}/baf-bagged-anchor-replay.XXXXXX")"
rsync -a --exclude=.git --exclude=.venv --exclude=data --exclude=outputs --exclude=.pytest_cache --exclude=__pycache__ ./ "$baf_replay_root/"
ln -s "$baf_original_root/.venv" "$baf_replay_root/.venv"
mkdir -p "$baf_replay_root/data/raw/baf" "$baf_replay_root/replay-evidence"
ln -s "$baf_original_root/data/raw/baf/Base.csv" "$baf_replay_root/data/raw/baf/Base.csv"
mv "$baf_replay_root/reports/baf-bagged-anchor-fusion-v1" "$baf_replay_root/replay-evidence/"
mkdir -p "$baf_replay_root/reports/baf-bagged-anchor-fusion-v1"
cd "$baf_replay_root"
```

Restore exactly sixteen ancestor selections, rejecting changed records before copying missing ones:

```bash
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import shutil
from baf.common import read_config, digest
c = read_config('configs/bagged-anchor-fusion.json')
pairs = [(Path(p), Path(p.replace('outputs/', 'reports/', 1)), h)
         for p, h in c['preserve_sha256'].items()
         if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(pairs) == 16
for original, exported, expected in pairs:
    assert exported.is_file() and digest(exported) == expected, str(exported)
    if original.exists():
        assert digest(original) == expected, str(original)
for original, exported, expected in pairs:
    if not original.exists():
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(exported, original)
PYTHON
```

Run stages in order. The test runner records actual classifier-fit calls, separating six current-study tiny format fits from historical test-module fits. `validate` performs the full independent development audit; the standalone `audit_development` entry point is also available. Preflight verifies preserved nine-fit and older miniature ancestry with zero new production miniature fits. Wholly absent required caches reproduce only their exact individual source recipes; partial/incompatible caches fail without fallback. Fresh development reproduction can require eighteen bagged, six additive and nine tree fits. Actual final bounds depend on the selected roles and source availability, with at most six bagged, two additive and three old-tree fits. No unrelated grid is rerun.

```bash
set -e
.venv/bin/python -m baf.bagged_anchor_study verify_declaration
.venv/bin/python scripts/check_bagged_anchor.py
.venv/bin/python -m baf.bagged_anchor_study record_checks
.venv/bin/python -m baf.bagged_anchor_study preflight
.venv/bin/python -m baf.bagged_anchor_study prepare_development
.venv/bin/python -m baf.bagged_anchor_study validate
.venv/bin/python -m baf.bagged_anchor_study freeze
.venv/bin/python -m baf.bagged_anchor_study exploratory
.venv/bin/python -m baf.bagged_anchor_report report
```

## Conditional LightGBM experts

Use the pinned environment, verified Base v2 and isolated replay copy described above. First restore the seventeen exact ancestor selections, checking all records before copying any:

```bash
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import shutil
from baf.common import read_config, digest
c = read_config('configs/conditional-lightgbm-experts.json')
pairs = [(Path(p), Path(p.replace('outputs/', 'reports/', 1)), h)
         for p, h in c['preserve_sha256'].items()
         if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(pairs) == 17
for original, exported, expected in pairs:
    assert exported.is_file() and digest(exported) == expected, str(exported)
    if original.exists():
        assert digest(original) == expected, str(original)
for original, exported, expected in pairs:
    if not original.exists():
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(exported, original)
PYTHON
```

Verify the declaration before running tests:

```sh
.venv/bin/python -m baf.conditional_expert_study verify-declaration
```

Then execute these stages in order. The runner uses fresh default `tests.xml` and `test_model_fits.json` paths before `record-checks`; retain existing receipts outside the replay copy. Changed declarations or reused ledger paths are rejected.

```bash
set -e
.venv/bin/python scripts/check_conditional_experts.py
.venv/bin/python -m baf.conditional_expert_study record-checks
.venv/bin/python -m baf.conditional_expert_study preflight
.venv/bin/python -m baf.conditional_expert_study miniature
.venv/bin/python -m baf.conditional_expert_study prepare-development
.venv/bin/python -m baf.conditional_expert_study validate
.venv/bin/python -m baf.conditional_expert_study freeze
.venv/bin/python -m baf.conditional_expert_study exploratory
.venv/bin/python -m baf.conditional_expert_report report
```

Preflight checks preserved sampling/miniature ancestry before the separate sixteen-fit production miniature. Wholly absent required references reproduce only their exact individual recipes in the new destination; partial or incompatible sets stop execution. Freeze limits transfer to the declared dependency roles. Retain the [miniature metadata qualification](../reports/baf-conditional-lightgbm-experts-v1/miniature-metadata-note.md) when interpreting historical gate evidence.

## Shallow CatBoost and incomplete work

Use the existing [shallow CatBoost reproduction guide](../reports/baf-shallow-catboost-v1/reproduction.md) for its eighteen-ancestor restore, resource gates and ordered stages. Preserve separate Newton fit and corrected analysis identities; the stopped L-BFGS attempt is not a completed experiment. Context-feature ablation remains incomplete at 3/14 verified tasks and contributes no measured submission result; its [active tasks](../openspec/changes/test-context-feature-ablation-f2/tasks.md) describe the remaining work.


For inspection of the incomplete context-feature study only, its implemented commands are:

```sh
.venv/bin/python -m baf.context_ablation_study verify_declaration
.venv/bin/python scripts/check_context_ablation.py --scope current --junit reports/baf-context-feature-ablation-v1/replay-current.xml --ledger reports/baf-context-feature-ablation-v1/replay-current-fits.json
```

These require the pinned environment, verified Base and nineteen original selections. Use fresh receipt paths. The current scope checks synthetic methodology; it does not satisfy the full-suite gate or produce a completed BAF study.

## Study reports

| Study | Findings and limitations |
|---|---|
| Original controlled study | [Findings](../reports/baf-study-v1/findings.md) |
| Feature and ensemble improvements | [Findings](../reports/baf-improvements-v1/findings.md) |
| Recall-weighted F2 | [Findings](../reports/baf-recall-weighted-v1/findings.md) |
| Extended F2 search | [Findings](../reports/baf-recall-weighted-extension-v1/findings.md) |
| Chronological stacking | [Findings](../reports/baf-chronological-stacking-v1/findings.md) |
| Calibrated policies | [Findings](../reports/baf-calibrated-f2-policy-v1/findings.md) |
| Focal loss | [Findings](../reports/baf-focal-loss-v1/findings.md) |
| Direct ranking | [Findings](../reports/baf-direct-ranking-v1/findings.md) |
| XGBoost rank ensembles | [Findings](../reports/baf-xgboost-rank-ensembles-v1/findings.md) |
| XGBoost score fusion | [Findings](../reports/baf-xgboost-score-fusion-v1/findings.md) |
| Stopped L-BFGS attempt | [Findings](../reports/baf-spline-logistic-v1/findings.md) |
| Newton splines | [Findings](../reports/baf-newton-spline-logistic-v1/findings.md) |
| Spline score fusion | [Findings](../reports/baf-spline-score-fusion-v1/findings.md) |
| Spline interactions | [Findings](../reports/baf-interaction-spline-logistic-v1/findings.md) |
| Interaction score fusion | [Findings](../reports/baf-interaction-score-fusion-v1/findings.md) |
| Unweighted reference fusion | [Findings](../reports/baf-unweighted-reference-fusion-v1/findings.md) |
| Bagged LightGBM seeds | [Findings](../reports/baf-bagged-lightgbm-seeds-v1/findings.md) |
| Bagged anchor fusion | [Findings](../reports/baf-bagged-anchor-fusion-v1/findings.md) |
| Conditional experts | [Findings](../reports/baf-conditional-lightgbm-experts-v1/findings.md) |
| Shallow CatBoost | [Findings](../reports/baf-shallow-catboost-v1/findings.md) |
