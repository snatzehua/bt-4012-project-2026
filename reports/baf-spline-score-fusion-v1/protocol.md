# Fixed spline-score fusion protocol

This is a separate prospective exploratory composition study, declared before its evaluation. C=0.1 and the source members were chosen adaptively after eleven completed studies. Settings are analyst-selected and are not approved group decisions. Month 7 is already consumed and cannot provide independent confirmation. All older completed identities, the separate Newton fit/analysis identities and the stopped L-BFGS attempt remain intact.

## Frozen inputs and inference

Use Base v2 at `data/raw/baf/Base.csv`, SHA-256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`, Python 3.12.9 and unchanged pins in the three requirements files. Configuration is `configs/spline-score-fusion.json`. Full source identity comprises the preserved 37-source analysis closure plus the three new modules. Seed 2026, six threads.

Required Newton members are linear/spline crossed with unweighted/balanced, each at C=0.1. Required tree bases are XGBoost d4/w1 and balanced/unweighted LightGBM under their frozen recipes. For each development month 4/5/6, every model uses the same expanding earlier population. Expected local work is 21 verified model/prediction reuses, zero fits. Saved positive-class inference, original populations, numerical gates, training-only feature metadata and provenance must check. Newton gates retain finite objective/gradient, absolute gradient <=1.1e-6, fewer than 100 iterations, no warnings or alternative-solver fallback.

Required caches are audited individually. Only a wholly absent required cache may be reproduced, under the exact frozen recipe in the new output directory; disclose actual fits. Reject partial/incompatible caches. Do not require unrelated old model grids or rerun the stopped L-BFGS attempt. Mandatory source/config/report hashes and all eleven restored selection pairs establish ancestry. Capture all available historical output files before execution and reject deletion/change thereafter. Old raw data and bulk artifacts remain ignored.

No predictor or preprocessing changes occur: Newton models retain 29 decoded original predictors, 17 smooth/7 discrete/5 categorical roles, train medians, all 24 numeric missing indicators, one-hot vocabularies and variance scaling without centering. Spline models retain observed-only cubic quantile knots and constant tails. Tree adapters retain their frozen training schemas. Current labels or unrelated inference rows cannot alter a row's features or composed score.

## Finite composition

For finite Newton score p in [0,1], epsilon=1e-6, define u=log(clip(p))-log1p(-clip(p)). The existing selected tree fusion g is already a finite log-odds margin; never transform it as a probability. Blend a*u+(1-a)*g at a=0.1/0.25/0.5, where a always means Newton weight. No learned weights, label use or batch statistics. Scores are not asserted calibrated probabilities. Check alignment, finite domains and every immediate parent/model/prediction hash. Deterministic endpoint/extreme-margin/invalid-domain/row-label-invariance fixtures require zero model fits.

There are 23 selectable strategies: four original Newton scores, four logit singles, three old references and twelve blends. Across three folds, export 21 bases and 51 derived scores (three reference fusions, twelve logits, 36 blends): 72 artifacts, including three nonselectable XGBoost dependencies. Export twelve clipping rows even if no value changes.

## Chronological policies and contrasts

Unchanged pooled/latest/rank-fraction policies optimize earlier positive-class F2=5TP/(5TP+4FN+FP), beta=2. Month 5 learns on month 4; month 6 on months 4/5 or latest month 5. No imposed review quota or precision floor. Rank fractions use exact rational ceilings and stable score/ID ties. Select by actual macro month-5/6 F2, then recall/precision and deterministic IDs. AP and pooled F2 are descriptive.

Retain 138 forward rows, 69 macro summaries, 69 pooled summaries and 69 score diagnostics. Report F1/F2/F3, precision/recall, TP/FP/FN/TN, workload, AP and ROC AUC. Export all 433 contrasts: 207 against three pooled references, 46 policy-versus-own-pooled, 12 logit-versus-raw, 72 blend-versus-common-margin-member, 18 matched representation blends, 18 matched class-weight blends, 36 alpha-pair comparisons, 12 matched representation singles and 12 matched weight singles. Retain negative differences and ties. Twelve score-diversity and 24 joint-error rows are diagnostics, never new inference or selection criteria. Constant-vector correlations have explicit undefined flags.

## Freeze and bounded transfer

Freeze actual winner, fusion/pooled, balanced/pooled, unweighted/pooled and same-model/pooled. Raw Newton winners add matched representation/weight originals (seven roles); transformed winners add matched transformed representation/weight and raw member (eight); blend winners add matched representation/weight blends at identical alpha/policy, transformed Newton member, fusion member and raw Newton member (ten). Reference winners retain five. Preserve aliases; deduplicate fitting. Freeze all development artifacts/source/config/software and final rules from months 4/5/6 (latest=6) before recording further consumed-month access.

Only frozen roles transfer. With local caches expect three reference base reuses and at most three new Newton fits, no new tree fit. Reproduce wholly absent required references only under exact recipes and count them. No losing-setting sweep or hindsight threshold. The broader goal requires positive measured F2 differences against all three references on both development and consumed transfer. Clear-gain language also requires positive conditional interval bounds for the named scope.

## Verification and reporting

Before freeze, independently reconstruct every required saved-model score, parent identity/composition, earlier-only policy, metric, contrast factor/arithmetic, clipping, correlation, joint-error row and actual-F2 winner. Verify complete expected sets. Use 500 paired fixed-decision row-bootstrap draws per month 5/6/7, seed 1027: 1,000 development and 500 consumed rows. For N roles, retain 3*(N-1) development interval summaries and N-1 consumed summaries. Independently reconstruct every draw and quantile using sampled Boolean rows. Conditional intervals exclude adaptive source/C choice, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity correction.

New modules: `baf/spline_fusion_scores.py`, `baf/spline_fusion_study.py`, `baf/spline_fusion_report.py`; destinations: ignored `outputs/baf-spline-score-fusion-v1/` and small exported `reports/baf-spline-score-fusion-v1/`. Inspect operating/composition figures. Findings distinguish standalone, transformation, composition and policy effects; preserve negative results. Update README reproduction commands, AGENTS/OpenSpec context and assistance attribution. Tests/compileall/strict OpenSpec validation and requirement-by-requirement evidence are required before tasks, sync and archive. Finite-search completion is not performance-goal achievement.

Codex assisted with this design and implementation using preserved repository APIs and pinned public libraries. The group must verify and explain submitted results.
