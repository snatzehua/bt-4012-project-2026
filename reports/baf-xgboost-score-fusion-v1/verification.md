# Score-fusion study verification

Verified on 11 October 2026. This is a completed exploratory study with a qualified gain against balanced LightGBM, not achievement of the broader performance goal. The frozen blend improves development and consumed-period F2 over balanced, but still trails the stronger unweighted consumed-period reference. Month 7 was already consumed before this study and supplies no independent confirmation.

## Identity and preservation

- Config SHA-256: `69346bf2c7554f557e4f52470c7a8de92b47b5a57399a260606c009f903b6ebc`.
- Selection SHA-256: `04b704a72ee9350b5444ced3cd167a68c78e9c6d4e4af521d1cb06838b81f222`.
- The 30-file source/dependency fingerprint and pinned software/Python identity agree across the config, validation record, frozen selection and audit. No package was added or changed. `uv pip check --python .venv/bin/python` checked 27 compatible packages.
- Recursive guards and a before/after snapshot verified 49 protected earlier source/config/selection/dependency files. All 22 explicit preservation entries, including nine original/exported selection pairs, agree. Earlier findings remain preserved.
- All 399 frozen development artifacts pass their stored hashes. The exported config/selection/access/completion records and copied tables agree with bulk artifacts. Subsequent model/score/policy changes require a new study identity.

## Requirement-to-evidence audit

| Contract | Checked evidence |
| --- | --- |
| Separate finite preserved study | Config, protocol, exact 61-strategy grid, guards, preservation snapshot and rejection tests |
| Label-free finite row-wise composition | Hand arithmetic, endpoint/tie/weight/alignment/label/batch tests; independent reconstruction of every score; dependency and clipping exports |
| Comparable chronological actual-F2 selection | Independent earlier-only rule reconstruction and sklearn confusion/F1/F2/F3/precision/recall checks for all 366 rows; macro ordering and unchanged old benchmarks |
| Complete matched comparisons | All 866 numerical contrasts and their named model/representation/weight/policy factors checked; all failures, ties and self comparisons retained |
| Frozen limited exploratory transfer | Immutable selection, eight predetermined roles, three base dependencies, six final score outputs, freeze/access chronology and independently rebuilt final decisions |
| Verified reproducible evidence | Full exports, saved-model inference, independent paired draws/quantiles, inspected figure, README commands, assistance attribution, tests, compileall and strict OpenSpec validation |

## Fits, reuse and saved-model inference

Development reused 12 frozen XGBoost models and nine old reference predictions. The old development reference caches contained no saved models. Nine separately counted exact-recipe model-verification fits reproduced those frozen predictions, with no warnings, and supplied saved models in the new destination. Their recorded total verification time is approximately 294.86 seconds. No new XGBoost development fit or new learning recipe was used.

Every reused XGBoost cache passed source and destination model/adapter/prediction audits, exact parameters, fitted rounds, training/evaluation IDs and feature identity checks. Every reference saved model reloaded, matched its factory recipe and reproduced the frozen scores. Training-only adapters, missing values, class orientation and unseen-category semantics remain those of the frozen recipes. Unit fixtures verify complete/absent/partial/incompatible caches, changed origins, saved-score tampering and the one-fit fresh-clone reference path. Test fixtures are outside study fit counts.

Transfer needed one new exact-recipe XGBoost fit on months 0-6, approximately 15.8 seconds, plus two reference prediction/model reuses. There were no final reference refits or additional reference-model verification fits. Only the frozen dependencies reached consumed month 7.

## Complete development evidence

| Artifact | Verified rows/results |
| --- | ---: |
| Raw base predictions | 21 |
| Derived development score records | 162 |
| Total development strategy/month predictions | 183 |
| Forward operating rows | 366 |
| Macro summaries | 183 |
| Pooled confusion summaries | 183 |
| Matched contrasts | 866 |
| Score diagnostics | 183 |
| Source/month clipping diagnostics | 18 |

The independent policy audit rebuilds thresholds and learned fractions directly from earlier labels/scores, with explicit score/ID ordering, tie boundaries and rational ceiling. It then constructs decisions and checks confusion counts, F1/F2/F3, precision, recall, workload, AP and ROC AUC. Macro month-5/6 means, deterministic winner ordering and pooled confusion metrics were rebuilt separately. No current-fold labels enter policy fitting.

All 183 scores reconstruct from original member arrays and independently expressed raw/logit formulas. Every derived record's member record and prediction hashes agree. All 866 contrasts have verified factors and arithmetic: 366 against old pooled references, 122 policy-minus-own-pooled, 18 logit-minus-raw singles, 288 fusion-minus-matched-member and 72 logit-minus-raw fusions. Diagnostics reconstruct actual ranges, ties and clipping boundaries. Zero source scores were changed by clipping in development; this is measured, not assumed.

The winner is `logit_mean_xgb_d4_w1_baseline_balanced_a50 / pooled`, macro F2 **0.3289065702**. Differences are **+0.0011049921** against balanced and **+0.0062528853** against unweighted. The paired conditional development interval against balanced is **[-0.002882, 0.005802]**, so this is not a clear or confirmed development gain.

## Frozen transfer and interpretation

The selection contains exactly eight roles: selected, balanced, unweighted, same-model pooled, transformed XGBoost/anchor members and their raw members. These resolve to six unique score outputs and three base dependencies. Final rules reconstruct from months 4/5/6. Freeze preceded further access, every recorded hash agrees, and the freeze command rejects an overwrite. No losing-setting sweep or hindsight month-7 threshold was computed.

All eight final metric rows were independently rebuilt. Selected F2 is **0.3726941227**, precision **0.1924141750**, recall **0.4866946779**, with **695 TP, 2,917 FP and 733 FN**. Balanced F2 is **0.3644085570** and unweighted F2 **0.3768844221**. The selected-versus-balanced difference is **+0.0082855656**, conditional 95% interval **[0.001241, 0.014945]**; selected-versus-unweighted is **-0.0041902994**, interval **[-0.013532, 0.003542]**. Named consumed-period improvement over balanced does not establish improvement over unweighted or independent generalization.

Balanced and selected each detect 695 frauds, with 671 detections shared. Each catches 24 frauds missed by the other. The blend flags 212 fewer legitimate applications in total; equal fraud counts do not mean identical detected cases.

## Paired uncertainty and export checks

Development has 1,000 draw rows, 500 for each of months 5 and 6. Consumed transfer has 500 draw rows. An independent direct-row resampling implementation rebuilt all **1,500 paired draws**, using sampled Boolean decisions rather than the reporting implementation's weighted count formulation. All seven comparator differences match on every draw. All **28 interval summaries** were independently rebuilt from quantiles, including the mean of separately resampled monthly F2 differences for macro development.

Models, selection and realized decisions remain fixed. Intervals exclude adaptive search/selection, retraining, temporal/generative uncertainty and multiplicity correction. They do not account for the many prior development experiments. Neither a zero-overlapping interval nor an unsuccessful comparison is hidden, and significance is not substituted for a positive measured performance difference.

There are 165 checked derived provenance records after the three permitted final compositions are included. Run/transfer provenance contains 21/3 base results. Report table counts, hashes, config and selection copies agree with bulk outputs. `fusion_operating.png` was opened and visually inspected: both panels, axes, legends and labels are readable without clipping; overlapping points reflect repeated fixed-weight comparisons.

## Reproduction and final checks

README includes Python and dependency installation, verified Base access/checksum, all nine selection-restoration commands, exact absent-cache reproduction, execution order, model/bootstrapping seeds and separate destinations. Every restoration source/target and module/protocol/config path was checked. Assistance attribution credits Codex and the official SciPy transform definition, distinguishes analyst-selected settings and states that no public notebook was copied.

- Full methodological suite: **152 passed**, latest full run 20.79 seconds.
- `python -m compileall -q baf tests`: passed.
- `openspec validate --all --strict --no-interactive`: **16 passed, zero failed** before main-spec synchronization.
- Main-spec synchronization: six requirements and eight scenarios match the implemented delta; all 12 main specs passed validation. Strict validation passed all 17 items before archive and all 16 remaining items after archive.
- Report and independent output/preservation audits: passed.

The implemented requirements are synchronized and the completed change is archived at `openspec/changes/archive/2026-10-11-test-xgboost-score-fusion/`, with all 14 tasks complete and its metadata preserved. These records preserve a useful selected-policy improvement against balanced and the failed comparison against the stronger unweighted reference. The broader performance goal remains active and unachieved.
