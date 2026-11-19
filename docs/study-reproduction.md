# Later-study reproduction recipes

Use the [main code guide](submission-code-guide.md) for report-table export, environment/data setup and the original controlled study. This reference retains later-study restore lists and stage order. Full-suite recipes are historical where the test-to-training handoff below remains unverified.

## Find an experiment

| Experiment | Instructions | Results |
|---|---|---|
| Original controlled study | [Main guide](submission-code-guide.md#original-controlled-study) | [Findings](../reports/baf-study-v1/findings.md) |
| Feature and ensemble improvements | [Main guide](submission-code-guide.md#feature-and-ensemble-improvements) | [Findings](../reports/baf-improvements-v1/findings.md) |
| XGBoost score fusion | [Recipe](#fixed-raw-score-and-log-odds-fusion) | [Findings](../reports/baf-xgboost-score-fusion-v1/findings.md) |
| Bagged anchor fusion | [Recipe](#bagged-ensemble-additions-to-the-anchor-complete) | [Findings](../reports/baf-bagged-anchor-fusion-v1/findings.md) |
| Recall-weighted F2 | [Recipe](#recall-weighted-f2-study) | [Findings](../reports/baf-recall-weighted-v1/findings.md) |
| Extended F2 search | [Recipe](#adaptive-f2-extension) | [Findings](../reports/baf-recall-weighted-extension-v1/findings.md) |
| Chronological stacking | [Recipe](#chronological-score-stacking) | [Findings](../reports/baf-chronological-stacking-v1/findings.md) |
| Calibrated policies | [Recipe](#calibrated-f2-batch-policies) | [Findings](../reports/baf-calibrated-f2-policy-v1/findings.md) |
| Focal loss | [Recipe](#focal-loss-score-learning) | [Findings](../reports/baf-focal-loss-v1/findings.md) |
| Direct ranking | [Recipe](#direct-fraud-ranking-study) | [Findings](../reports/baf-direct-ranking-v1/findings.md) |
| XGBoost rank ensembles | [Recipe](#xgboost-and-rank-ensembles) | [Findings](../reports/baf-xgboost-rank-ensembles-v1/findings.md) |
| Stopped L-BFGS attempt | [Recipe](#additive-spline-logistic-attempt-stopped-at-convergence-gate) | [Findings](../reports/baf-spline-logistic-v1/findings.md) |
| Newton splines | [Recipe](#newton-spline-logistic-successor-complete) | [Findings](../reports/baf-newton-spline-logistic-v1/findings.md) |
| Spline score fusion | [Recipe](#spline-score-fusion-study) | [Findings](../reports/baf-spline-score-fusion-v1/findings.md) |
| Spline interactions | [Recipe](#fixed-spline-interactions) | [Findings](../reports/baf-interaction-spline-logistic-v1/findings.md) |
| Interaction score fusion | [Recipe](#fixed-interaction-score-fusion) | [Findings](../reports/baf-interaction-score-fusion-v1/findings.md) |
| Unweighted reference fusion | [Recipe](#unweighted-reference-fusion) | [Findings](../reports/baf-unweighted-reference-fusion-v1/findings.md) |
| Bagged LightGBM seeds | [Recipe](#bagged-lightgbm-seed-study) | [Findings](../reports/baf-bagged-lightgbm-seeds-v1/findings.md) |
| Conditional experts | [Recipe](#conditional-lightgbm-experts) | [Findings](../reports/baf-conditional-lightgbm-experts-v1/findings.md) |
| Shallow CatBoost | [Dedicated guide](../reports/baf-shallow-catboost-v1/reproduction.md) | [Findings](../reports/baf-shallow-catboost-v1/findings.md) |
| Supplementary notebooks | [Separate branch instructions](submission-code-guide.md#later-studies-and-supplementary-notebooks) | [Pinned branch audit](submission-branch-audit.md) |
| Context-feature ablation | [Incomplete: checks only](#incomplete-context-feature-ablation) | No completed BAF result |

## Shared setup and isolation

Run Bash blocks from the repository root on Linux or WSL, with Python 3.12.9 and verified BAF Base v2. Follow the main guide's [venv/pip setup and checksum check](submission-code-guide.md#environment-and-data). Early recipes need `requirements-improvements.txt`; XGBoost and later recipes need `requirements-xgboost.txt`. The expanded historical suite requires the latter plus its ancestor records. Tests may read Base.csv and fit small synthetic models; those are separate from full-data training.

Use a disposable checkout with original LF bytes and no target-study outputs. Reporting writes tracked `reports/` paths as well as ignored `outputs/`; retain the research checkout and original evidence outside the replay. Filesystem-copy blocks assume a provisioned source environment/data and move only the copy's current-study report export aside. They do not establish a valid full-suite receipt handoff.

Restore only a recipe's declared ancestors for training. Check existing selections against their exports; `cp -n` leaves an unequal existing file untouched and is not a verification step. Python restore blocks verify exact hashes. Required caches must be wholly absent or pass the individual identity/inference checks; reject partial or incompatible sets. Resume only with unchanged sources, configuration, data and software. Never remove a research selection, relax a gate or overwrite frozen receipts to simulate a fresh run.

Run stages sequentially and stop on failure. Full-suite commands below are reference sequences subject to the next section, not currently verified copy-and-run instructions. Month 7 is already evaluated; all follow-up comparisons are exploratory. F2 has beta=2 without a quota or precision floor. New tuning needs a separate study and cannot restore an independent holdout.

## Full-suite receipts and training replay

### What source inspection establishes

At reviewed revision `7425088`, the test modules, scientific sources and runners are identical to `7e9baab`. Checking out `7e9baab` therefore does not recover an earlier per-study test suite.

The current `tests/test_shallow_catboost.py::test_preservation_and_prospective_enumeration_without_fits` calls the shallow study's `guard`. Its configuration requires eighteen ancestor selections, including the original interaction, unweighted, bagged-seed, bagged-anchor and conditional selections. Those earlier targets cannot simply run today's full suite in a training copy with only their own ancestors restored. The bagged-anchor declaration test also checks its sixteen ancestors and the pinned scientific identity. The context-ablation configuration lists nineteen selections, but restoring all nineteen in a fresh training copy is not a solution: that also restores the target's already-frozen selection.

The research checkout retains original evidence. A separate test-only copy can hold ancestor/target records needed by broad verification, while a fresh training copy must retain only its declared ancestors and no target selection where the recorder forbids one. **The transfer of fresh accepted receipts between those copies has not been validated.** Do not interpret the copy commands below as completing that transition.

| Target recipe | Recorder input in its `report_dir` | Specific current-replay limitation |
|---|---|---|
| Spline interactions | `tests.xml` | Expanded-suite restores differ from fresh training prerequisites; recorder does not itself reject a target selection, but later validation does. |
| Interaction-score fusion, unweighted fusion, bagged seeds | `tests.xml` | `record_checks` rejects a target selection that descendant preservation checks require during broad tests. |
| Bagged anchor, conditional experts | `tests.xml`, `test_model_fits.json` | Same selection conflict; exact full-suite/ledger acceptance must also be established. |
| Shallow CatBoost (dedicated guide) | `tests.xml`, `test_model_fits.json` | Same strict ledger contract; no equivalent own-selection conflict is established here. Its current-suite receipt acceptance remains unverified. |

For bagged anchor, [the runner](../scripts/check_bagged_anchor.py) accepts `--junit` and `--ledger`, rejects an existing ledger, and runs all `tests`. [The recorder](../baf/bagged_anchor_study.py) reads the fixed filenames above, requires every current test module, matches collected node IDs to JUnit cases, rejects deselection/failures/skips, checks six actual current-study fits and verifies ledger source hashes. It writes `outputs/baf-bagged-anchor-fusion-v1/checks.json` and report fit accounting with the current fingerprint. Preflight/execution recheck those receipt hashes.

The ledger names relative source paths and test node IDs; the inspected consumer does not compare an absolute checkout root. That alone does not validate receipt portability. The receiving copy must pass its preservation guard, match source/configuration/software fingerprints, supply the exact receipt bytes at the expected paths, and retain source/report state used by subsequent gates. Conditional and shallow consumers also require the complete current execution-source key set. Arbitrarily copying two historical files or substituting a scoped test pass does not establish acceptance. The three earlier XML-only recorders and bagged-seed recorder have different checks; do not infer the latest ledger contract applies to them.

### Executed scope and remaining boundary

Source/configuration inspection establishes the dependencies and filename contracts above. This cleanup environment has Python 3.14.2, no Python 3.12.9 virtual environment, no LightGBM/CatBoost/XGBoost installation and no Base.csv. No full methodology run, fresh receipt-consumption test or scientific training was executed. The historical sequences below remain useful references to the original completed runs, whose verification records are linked in the index; they are not a verified latest-checkout replay.

A supported handoff still needs a provisioned pair of isolated copies, genuine fresh runner output, exact source/runtime matches and successful unmodified `record_checks` acceptance in the selection-free training copy before preflight. Do not change a guard, fingerprint or test to make this transition pass. Evidence-table export and the scoped original-study instructions in the main guide do not depend on this handoff.

## Recall-weighted F2 study

Install the improvements requirements. Restore the two original-study/improvement selections and improvement validation summary below. Missing eligible predictions are refitted into this study: 30 new fits with the historical caches, or 45 fits without them.

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

Models/predictions: `outputs/baf-recall-weighted-v1/`. Tables, figures and receipts: `reports/baf-recall-weighted-v1/`. Validation checkpoints allow unchanged runs to resume. Freeze requires complete validation and refuses an existing selection. The report includes workload, missed fraud and legitimate reviews alongside F2.

## Adaptive F2 extension

Install the improvements requirements and restore three ancestor selections. The `cp -n` commands preserve existing records; check any existing record against its export before continuing. Missing reference predictions are fitted only into the extension output directory.

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

Models/predictions: `outputs/baf-recall-weighted-extension-v1/`. Tables, figures and receipts: `reports/baf-recall-weighted-extension-v1/`. The recipe requires 36 new development fits with its reference caches, or 117 base candidate/month fits without them. Freeze requires complete validation; transfer reuses only the frozen selected dependencies and baseline references.

## Chronological score stacking

Install the improvements requirements and restore four ancestor selections. Four fixed members generate earlier out-of-time scores; stacking and its matched controls train only on earlier scores.

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

Models/predictions: `outputs/baf-chronological-stacking-v1/`. Tables, figures and receipts: `reports/baf-chronological-stacking-v1/`. Expect 12 early base fits with the historical reference caches, or 27 base fits without them, plus 72 combiner/control fits. Percentile variants require batch inference on the unlabelled application batch. Freeze fits the selected final combiner/control before transfer.

## Calibrated F2 batch policies

Install the improvements requirements and restore five ancestor selections. Calibrators and batch policies use earlier out-of-time predictions. With no score caches, up to 27 frozen-recipe base results and three fits of the fixed stack are needed in this study's output directory.

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

Models/predictions: `outputs/baf-calibrated-f2-policy-v1/`. Tables, figures and receipts: `reports/baf-calibrated-f2-policy-v1/`. Run diagnostics only after freeze: the label-aware development prefix comparisons are excluded from selection. Never compute a month-7 hindsight optimum. Transfer evaluates frozen selected dependencies; the batch chooser must not use current-month labels.

## Focal-loss score learning

Install the improvements requirements and restore six ancestor selections. Use only verified reference caches; otherwise reproduce their exact individual recipes in the focal output directory.

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

Models/predictions: `outputs/baf-focal-loss-v1/`. Tables, figures and receipts: `reports/baf-focal-loss-v1/`. Validation checkpoints model adapters, raw margins, sigmoid scores and numeric traces. Resume only with unchanged identities. Focal curvature floors approximate optimisation; objective calls are not fitted-tree counts. Policies learn from earlier prediction months.

## Direct fraud-ranking study

Install the improvements requirements. First restore the six ancestor selections listed in the [focal-loss recipe](#focal-loss-score-learning), then add the focal selection below. This is a restore dependency, not an instruction to train the focal study first.

```sh
mkdir -p outputs/baf-focal-loss-v1
cp -n reports/baf-focal-loss-v1/selection.json outputs/baf-focal-loss-v1/selection.json
.venv/bin/python -m pytest -q tests/test_direct_ranking.py
.venv/bin/python -m baf.direct_ranking validate
.venv/bin/python -m baf.direct_ranking freeze
.venv/bin/python -m baf.direct_ranking exploratory
.venv/bin/python -m baf.direct_ranking_report
```

Models/predictions: `outputs/baf-direct-ranking-v1/`. Tables, figures and receipts: `reports/baf-direct-ranking-v1/`. Validation records ordered training/group identities as well as model predictions. Query construction uses eligible months, IDs and seed, retaining all-negative groups and merging singleton tails without labels. Finite raw rank scores are not probabilities.

## XGBoost and rank ensembles

Install `requirements-xgboost.txt` through the shared venv/pip setup. Restore eight ancestor selections. Missing reference caches are reproduced only under their frozen recipes in this study's output directory.

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

Models/predictions: `outputs/baf-xgboost-rank-ensembles-v1/`. Tables, figures and receipts: `reports/baf-xgboost-rank-ensembles-v1/`. Batch percentiles use no labels and are not calibrated probabilities. Freeze includes the selected strategy, its pooled-policy control, both baseline references and the prescribed member controls; transfer must remain within that dependency set.

## Fixed raw-score and log-odds fusion

Install `requirements-xgboost.txt` and restore nine ancestor selections. Inspect each required prediction/model pair: reject partial or incompatible caches. Where reference predictions lack saved models, exact-recipe verification fits must reproduce the scores before supplying the missing models.

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

Models/predictions: `outputs/baf-xgboost-score-fusion-v1/`. Tables, figures and receipts: `reports/baf-xgboost-score-fusion-v1/`. Composition uses no labels or batch ranks. Freeze retains the actual development winner and prescribed controls. Do not sweep losing settings or choose a hindsight cutoff during transfer. Conditional intervals and the negative consumed-month comparison are in the findings.

## Additive spline-logistic attempt (stopped at convergence gate)

Install `requirements-xgboost.txt` and restore ten ancestor selections. This is the historical stopped recipe, not a completed experiment: the weighted spline C=0.1 full-data fit reached the declared 2,000-iteration limit.

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

Models/predictions: `outputs/baf-spline-logistic-v1/`. Tables, figures and receipts: `reports/baf-spline-logistic-v1/`. The miniature gate tests convergence, deterministic repeats and saved inference. A full-data warning or invalid inference stops execution before selection; later commands in the block must not run after that failure. Retain failure evidence. Changing solver, iterations, C or representation requires a separate study.

## Newton spline-logistic successor (complete)

Install `requirements-xgboost.txt` and restore ten ancestor selections. Use the corrected `newton_spline_audit` entry points below, retaining the separate original fit and corrected analysis identities.

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

Models/predictions: `outputs/baf-newton-spline-logistic-v1/`. Tables, figures and receipts: `reports/baf-newton-spline-logistic-v1/`. Required absent reference caches use exact frozen recipes in this destination; partial/incompatible caches fail. Stopped L-BFGS model caches are optional: verify any present against their recorded hashes, but do not rerun the failed attempt. Missing old-solver objective diagnostics do not enter selection. Stop at any numerical gate failure.

## Spline-score fusion study

Install `requirements-xgboost.txt` and restore eleven ancestor selections. Required caches are checked individually, including saved inference and Newton objective/gradient/fallback evidence. An empty-cache replay can require twelve Newton and nine tree development fits.

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

Models/predictions: `outputs/baf-spline-score-fusion-v1/`. Tables, figures and receipts: `reports/baf-spline-score-fusion-v1/`. Run the independent score audit before reporting. Optional historical bulk outputs need not exist in a fresh clone, but any present files must retain their identities. Do not invoke unrelated grid loaders or rerun the stopped L-BFGS attempt. Transfer follows only the prescribed winner-type roles.

## Fixed spline interactions

Historical full-suite recipe: read the [receipt limitation](#full-suite-receipts-and-training-replay) before executing this section. Install `requirements-xgboost.txt`; restore twelve ancestor selections with the checked Python block. Required missing references use exact individual recipes in this destination.

Restore the declared ancestor selections:

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

Historical stage order (subject to the receipt limitation above):

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

Models/predictions: `outputs/baf-interaction-spline-logistic-v1/`. Tables, figures and receipts: `reports/baf-interaction-spline-logistic-v1/`. Freeze requires the independent development audit and refuses an existing selection. The recorder consumes JUnit and records the study test-source hash; it does not consume a model-fit ledger. Numerical auditing uses the host's actual `longdouble` width, which may offer no extra precision over float64.

## Fixed interaction-score fusion

Historical full-suite recipe: read the [receipt limitation](#full-suite-receipts-and-training-replay). Install `requirements-xgboost.txt`; restore thirteen ancestor selections. The exported eight-fit miniature ancestry receipt is required; historical miniature models are optional and checked if present.

Restore the declared ancestor selections:

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

Historical stage order (subject to the receipt limitation above):

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

Models/predictions: `outputs/baf-interaction-score-fusion-v1/`. Tables, figures and receipts: `reports/baf-interaction-score-fusion-v1/`. Validation includes the independent development audit; `audit_development` is also available before selection. New fits require the actual-host predictor-only preflight and unchanged numerical gates. Freeze limits transfer to the prescribed roles and cache-specific bounds.

## Unweighted-reference fusion

Historical full-suite recipe: read the [receipt limitation](#full-suite-receipts-and-training-replay). Install `requirements-xgboost.txt`. The copy block assumes the source checkout already has the pinned environment and verified read-only data. Move only the disposable copy's target report export aside, then restore fourteen ancestors.

Prepare the disposable copy:

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

Restore the declared ancestor selections:

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

Historical stage order (subject to the receipt limitation above):

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

Models/predictions: `outputs/baf-unweighted-reference-fusion-v1/`. Tables, figures and receipts: `reports/baf-unweighted-reference-fusion-v1/`. Run the full test/recording gate before preflight and validation. Validation includes the independent development audit. Cache checks and transfer stay within this grid's frozen source recipes; do not overwrite the original report or selection.

## Bagged LightGBM seed study

Historical full-suite recipe: read the [receipt limitation](#full-suite-receipts-and-training-replay). Install `requirements-xgboost.txt`. The copy block links an existing pinned environment and verified read-only dataset; it does not install them. Restore fifteen checked ancestor selections in the disposable copy.

Prepare the disposable copy:

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

Restore the declared ancestor selections:

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

Historical stage order (subject to the receipt limitation above):

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

Models/predictions: `outputs/baf-bagged-lightgbm-seeds-v1/`. Tables, figures and receipts: `reports/baf-bagged-lightgbm-seeds-v1/`. With no required references, expect 15 development reproductions in addition to 18 new bagged fits. Preflight and miniature gates must pass before production fitting. Validation performs the grid-specific audit. Transfer fits only the bases needed by its frozen winner/control roles.

## Bagged-ensemble additions to the anchor (complete)

Historical full-suite recipe: the [receipt limitation](#full-suite-receipts-and-training-replay) applies directly to this target. Install `requirements-xgboost.txt`. The copy block assumes a pinned environment and verified read-only data in the source checkout. Restore sixteen ancestors, checking all existing/exported records before copying missing ones.

Prepare the disposable copy:

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

Restore the declared ancestor selections:

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

Historical stage order (subject to the receipt limitation above):

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

Models/predictions: `outputs/baf-bagged-anchor-fusion-v1/`. Tables, figures and receipts: `reports/baf-bagged-anchor-fusion-v1/`. The runner produces JUnit and an actual-fit ledger for `record_checks`; a scoped pass cannot replace that gate. Preflight checks preserved miniature ancestry without refitting old gates. Missing development sources can require eighteen bagged, six additive and nine tree fits. `validate` includes the independent audit; transfer remains within the declared dependency bounds.

## Conditional LightGBM experts

Historical full-suite recipe: read the [receipt limitation](#full-suite-receipts-and-training-replay). Install `requirements-xgboost.txt` and restore seventeen checked ancestor selections before declaration verification. The default runner paths must be unused in the replay copy.

Restore the declared ancestor selections:

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

Verify the declaration:

```sh
.venv/bin/python -m baf.conditional_expert_study verify-declaration
```

Historical stage order (subject to the receipt limitation above):

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

Models/predictions: `outputs/baf-conditional-lightgbm-experts-v1/`. Tables, figures and receipts: `reports/baf-conditional-lightgbm-experts-v1/`. Preflight checks preserved sampling/miniature ancestry before the separate sixteen-fit production miniature. Required missing references use exact individual recipes; partial/incompatible sets stop execution. Freeze limits transfer to the declared roles. Retain the [miniature metadata qualification](../reports/baf-conditional-lightgbm-experts-v1/miniature-metadata-note.md) when interpreting the original gate evidence.

## Shallow CatBoost

The [dedicated reproduction guide](../reports/baf-shallow-catboost-v1/reproduction.md) retains its eighteen-selection restore, preflight and ordered stages. Read the [current receipt limitation](#full-suite-receipts-and-training-replay) before treating its full-suite block as a latest-checkout replay. Preserve the original verification receipts; this cleanup does not replace them.

## Incomplete context-feature ablation

This study remains at 3/14 verified tasks and has no completed BAF performance result. Its [active tasks](../openspec/changes/test-context-feature-ablation-f2/tasks.md) define remaining work. The commands below inspect implemented methodology only; they require the pinned environment, verified Base and nineteen original selections. Use fresh receipt paths.

```sh
.venv/bin/python -m baf.context_ablation_study verify_declaration
.venv/bin/python scripts/check_context_ablation.py --scope current --junit reports/baf-context-feature-ablation-v1/replay-current.xml --ledger reports/baf-context-feature-ablation-v1/replay-current-fits.json
```

`--scope current` runs synthetic methodology checks and cannot satisfy the full-suite gate. Keep the stopped L-BFGS attempt and separate Newton fit/analysis identities intact; neither is made complete by these checks.
