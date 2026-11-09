# Focal-loss study verification

Verified on 11 October 2026. All 14 implementation tasks are supported by the evidence below. Search completion does not establish the performance goal; this study found no improvement.

## Execution and preserved identity

The ordered validate, freeze, exploratory and report commands each returned exit code 0. Validation completed 24 full new development fits and nine verified reference reuses, plus four separate miniature integration fits. Full new fit time totals 772.08 seconds, excluding audit/report overhead. Every full new model fitted 800 trees; custom objectives recorded 800 calls per fit. No captured fit warnings occurred.

Config SHA-256: `4b2d742c49cd3c395afdc4dde65365d5f6ba949568273f1f12aa2c4f0c642390`. Selection SHA-256: `a8817aa2f2398c87dc8f55e4cf71b34ed49891c3dc69c1b67846b285b8a0653a`. Selection time: `2026-10-11T04:48:01.195453+00:00`. The exported fingerprint includes Python 3.12.9, the pinned software and all 17 source hashes. The preservation guard verifies all six previous source/config/software identities, twelve original/exported selection-record hashes and the calibrated-study config hash. All remain unchanged after execution. Frozen artifacts and the six development-table hashes match current files; exported config/selection/access/transfer records match ignored originals byte for byte.

## Loss mathematics, integration and chronology

The complete methodological suite passes: **83 passed in 2.43s**. `python -m compileall -q baf tests` returned 0. Tests cover finite-difference gradients/Hessians for both labels and gamma/weight combinations, extreme margins, negative curvature and its declared floor, exact gamma=0 derivatives, explicit one-sigmoid inference, training-only feature levels, saved-model reconstruction and incompatible/tampered caches. Both miniature gamma=0 integration gates have maximum probability discrepancy 0.0 against the 1e-6 tolerance; four miniature fits are not full-data performance fits.

The full audit reconstructs 33 prediction results from saved models/reference artifacts and checks source/config/software, row/label identity, earlier training boundaries and training-only preprocessing. All new models use months before 4, 5 or 6 for the corresponding prediction month. All 66 forward operating rows reconstruct from earlier-only policies, with independent sklearn/confusion/F2 checks. Original pooled balanced/unweighted macro F2 reproduce 0.32780157809618937 and 0.3226536849212962. Summary sorting selects balanced/pooled, and only the two old baseline recipes are in the final dependency set.

Full ordinary-loss implementations have probability discrepancies after 800 rounds despite exact miniature equivalence: mean differences range approximately 0.000925-0.002858 and rank correlations 0.991796-0.993715. All six discrepancy rows and both controls remain visible; this is not evidence of focal improvement. Custom gamma=0 curvature was never floored. The 14,400 trace rows cover 18 custom fits; the minimum recorded true focal Hessian is negative, and all supplied focal Hessians satisfy the declared 1e-6 lower bound. The floor is explicitly approximate optimization.

## Tables, decisions and intervals

Checked table counts: crossfit 66, macro summary 33, contrasts 106, pooled-confusion summaries 33, ordinary-score diagnostics 6, final role rows 4, numeric traces 14,400 and numeric summaries 18. Every contrast's arithmetic was recomputed from its named candidate/reference and policy; matched loss comparisons retain the same positive weight and policy. Pooled F2 is reported separately from macro F2.

Final access follows selection freeze and retains the same selection hash. Exactly two baseline prediction reuses and zero new month-7 fits occurred; there is no losing focal sweep or hindsight month-7 optimum. All four final role confusion/F2 metrics reconstruct independently from saved decisions. Selected F2 is 0.3644085570469799 versus unweighted 0.3768844221105528, difference -0.0124758650635729. This is exploratory evidence on an already-consumed month.

The new unequal-month fixture independently reproduces shared row resampling for each paired comparison and averages the two monthly F2 differences. Exported development draws total 1,000 rows (500 per month); final draws total 500. Every exported macro/final interval quantile was recomputed from those draws. The final selected-minus-unweighted conditional 95% interval is [-0.022196928356154423, -0.004001763688668752]. Same-policy comparisons yield exact zero differences. Draws use seed 1027 and fixed decisions, excluding adaptive model selection, retraining, temporal/generative uncertainty and multiplicity correction.

## Findings and completion

The best focal policy is custom_g2_w1/rank_fraction, macro F2 0.323007, below balanced 0.327802. The best added ordinary control is builtin_w3/pooled, 0.323783. All unsuccessful comparisons remain exported. Actual F2, rather than training loss or AP, determines the conclusion. The performance goal remains active and unachieved.

The loss_operating.png figure was opened and visually inspected: both panels, legends, axes and labels are readable with no clipping. README commands, Base checksum/access, pinned environment, six-selection restoration, new-destination missing-cache behavior, seeds, ordered stages and output paths were checked. AGENTS and project context state the measured result and seven frozen studies. The findings and README acknowledge Codex's design, code, numerical/methodological tests, execution, figures and interpretation; no public notebook was copied.

All five implemented requirements were synchronized into baf-focal-loss-study. Strict spec validation passed 9 entries; strict all-artifact validation passed 14 entries before archive. The standard archive step and final strict validation are recorded after the completed checklist is checked.

Archive completed at `openspec/changes/archive/2026-10-11-test-focal-loss-score-learning/`, preserving .openspec.yaml and all 14 checked tasks. Main/delta requirement blocks and Purpose match exactly. Post-archive strict validation passed 13 entries with zero failures. No source/config/model/prediction changes occurred after freeze.
