# Spline-score fusion verification

The finite experiment is negative against the broader goal criteria. The development-selected 25% unweighted spline / fusion blend with latest policy scored F2 0.3290099447918403, versus the frozen fusion/pooled reference 0.3289065701775199. Its consumed F2 is 0.3702475685234306 (670 TP, 2666 FP, 758 FN); the existing fusion scores 0.3726941226941227 and unweighted LightGBM 0.3768844221105528. The selected role is preserved despite transfer failure.

## Actual execution and identities

- 21 development model/prediction reuses (12 Newton, nine tree), zero new development fits; 51 derived scores, 72 total score artifacts, 69 selectable results.
- 138 earlier-only forward rows; 69 macro/pooled/score summaries each; 433 factor-matched contrasts; 12 clipping rows, 12 score-diversity rows and 24 joint-error rows.
- 40 production source files in the complete transitive identity. Newton fit and corrected analysis ancestors are separately retained. The frozen config SHA-256 is 0b8277763cfacc0f03ffb85a721f8b2e77f2dd671ee3f39bb9bdc0535fd70b34; selection SHA-256 is 3d6b693e78be6dee30b0911a0776cce6580f4edbaa1cb9c389d0a02a9e0c8dae.
- 325 mandatory hashes and 2,287 available historical files are preserved. The earlier 341-hash completion snapshot was checked unchanged. No original study was edited or rerun.
- Ten frozen roles, six unique required base models: three new Newton fits, three reference reuses, 13 final score artifacts. Every new Newton fit took eight iterations, with zero warnings/fallback and maximum independent absolute gradient 9.951854287721934e-7, below 1.1e-6.

## Independent evidence

The independent development entry point rechecks saved inference for every required model, reconstructs earlier-only thresholds/rank fractions separately, and checks every operating metric using sklearn confusion/metric APIs. Complete score/table key sets, summary/pooled arithmetic, matched factors, clipping, correlations and joint-error outcomes are checked before freeze. Source models retain their original numerical/representation gates; partial/incompatible caches are rejected and wholly absent required recipes reproduce only in the new destination.

The supplementary independent_score_audit.py verifies all 85 development/transfer score artifacts and exact immediate-parent hashes, reconstructs 15 Newton scores using expit(X*coefficient+intercept), independently recomputes all final/development compositions from original members, and reconstructs all 433 contrast factor tuples through a separate declaration. Its result and auditor hash are exported.

All 1,500 paired fixed-decision draws were reconstructed using directly sampled Boolean decisions; 27 development and nine consumed interval summaries and every quantile were checked. Seed 1027, 500 draws per month. Development-versus-fusion macro difference 0.000103 has interval [-0.004381, 0.004735]; consumed-versus-fusion difference -0.002447 has interval [-0.007202, 0.003564]. These conditional intervals exclude adaptive source/C choice, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity correction. Month 7 is already consumed.

## Interpretation and visual checks

Every pooled-policy blend trails fusion/pooled in development. The selected latest-policy blend exceeds its own pooled rule by about 0.000805 and fusion/latest by 0.001428; its lead over fusion/pooled is only 0.000103. Thus the reported lead combines composition and policy changes. At transfer, the selected latest-policy rule trails both its own pooled rule and fusion/latest. All raw/logit, representation, weighting and alpha controls remain visible; no losing setting replaces the selected model.

Newton/fusion development correlations range approximately 0.923-0.970 (Pearson) and 0.916-0.968 (Spearman). This is consistent with limited complementarity but does not establish the cause of transfer failure. The explicit joint-error tables provide the error-overlap counts without changing inference.

fusion_operating.png was visually inspected: all axes, units, legend entries and both panels are legible, with no clipping. The first panel shows all 69 macro results; the second holds pooled policy fixed and shows all twelve blend differences against fusion/pooled. Source smooth-effect figures remain in the frozen Newton report.

## Checks and reproduction

The initial suite passed 286 tests in 10.78s; after adding absent-reference-cache coverage, the final suite passed 288 tests in 10.93s, including 52 new methodological checks (domains, extreme margins, endpoints, weight meaning, label/row invariance, misalignment, complete factor families, all winner types, earlier-only policy ties and partial-cache/preservation rejection). Composition fixtures fit zero models. Two additional tiny chronological cache-reproduction recipe fixtures made two reference fits per targeted/full-suite run, four test-only fits in total; these are separate from zero full development fits and three full transfer fits. They verify exact saved inference, no second fit on repeated preparation, unchanged source destination and rejection after deleting an own model file. compileall and the supplemental auditor's py_compile passed. uv pip check verified 27 compatible installed packages. Exact README commands restore eleven original selection pairs and order initialize, tests, validate, audit, freeze, exploratory transfer, supplementary audit and report. Data/Python/packages/source/configuration are pinned.

Strict OpenSpec validation and the final requirement audit are recorded separately in completion_audit.json. Finite-study completion does not achieve the broader goal: the selected blend fails the consumed fusion and unweighted comparisons. The goal remains active.
