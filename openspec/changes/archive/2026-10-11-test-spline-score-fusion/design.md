## Context

See proposal.md and specs/baf-spline-score-fusion-study/spec.md. Eleven completed studies and one stopped L-BFGS attempt are preserved. The Newton selection SHA-256 is `25d54fb91852008f5669ff90f11d49fbf43b961615ce0c4e6fa97d3d97e0820e`; its configuration SHA-256 is `55cf8de250c82e19ca1c00476f1e69d1e9ee21870f7ddcabd8bd58b4cb8d1ec9`. Original 35-source fit and corrected 37-source analysis identities are distinct and both frozen. This design is needed for mixed score domains, cross-study saved-model provenance and winner-dependent transfer controls.

The three pooled references remain selected fusion `logit_mean_xgb_d4_w1_baseline_balanced_a50`, balanced LightGBM and unweighted LightGBM. Macro F2 is 0.3289065701775199 / 0.32780157809618937 / 0.3226536849212962; consumed F2 is 0.3726941226941227 / 0.3644085570469799 / 0.3768844221105528. These are adaptively reused observations. C=0.1 is fixed because the completed study's best added strategy used it, with a full representation/weight control square retained. No independent test period remains. All settings are analyst-selected.

## Goals / Non-Goals

**Goals:** Test fixed common-margin complementarity, isolate composition from transformation and matched representation/weight effects, and preserve explainable model identities without unnecessary refitting.

**Non-Goals:** Search C/knots/features/solvers again, learn weights or calibration, use batch ranks in composition, treat margins as probabilities, impose quotas/floors or restore independent confirmation. Diagnostic score/error overlap cannot become a new selection rule after outcomes.

## Decisions

### 1. Isolated identity and exact required models

Use Base v2 SHA-256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`, Python 3.12.9 and unchanged package pins. New configuration/protocol/modules/destinations are those in proposal.md. Declare a full transitive source identity (preserved 37-source closure plus three new modules), configuration/software/Python/data identities and all eleven original/exported selections. Preserve frozen sources, configurations, tracked evidence and every available original artifact, including the stopped run. Capture optional ignored-artifact presence before execution and reject subsequent deletion/change. A fresh clone can lack those bulk caches; do not reproduce unrelated prior grids merely to recreate history.

Four Newton members are `linear_c0p1_unweighted`, `linear_c0p1_balanced`, `spline_c0p1_unweighted`, `spline_c0p1_balanced`. Reuse their four models for each development month 4/5/6: 12 checked model/prediction pairs. Verify source fit/analysis identities, original recipe/populations, serialized positive-class inference, training representation, independent gradient and absence-of-fallback evidence. Reuse three XGBoost d4/w1, three balanced and three unweighted reference models/predictions: nine additional pairs. Derive the fixed existing fusion on each fold. Local execution expects 21 base reuses and zero development fits.

Wholly absent required caches may be reproduced under these exact recipes in the new destination (up to 12 Newton plus nine reference development fits), counted separately. Do not call a prior full-grid loader that demands unrelated absent caches or rerun the stopped L-BFGS attempt. Mandatory tracked selection/config/source hashes establish ancestry; required individual caches then establish recipe/population/inference compatibility. Partial/incompatible required caches fail rather than being overwritten. Use explicit helper interfaces, not patched globals. Preserved Newton numerical gates apply to every reproduced fit: declared Newton–Cholesky/L2 recipe, independent gradient <=1.1e-6, no warnings/fallback, iterations below 100 and checked saved inference.

No predictor or preprocessing changes occur. Newton members retain the same 29 decoded original predictors, 17 smooth/7 discrete/5 categorical roles, training medians, 24 missing indicators, one-hot vocabularies and train-only scaling; spline members retain observed-only cubic knots and constant tails. Tree members retain their frozen model-specific adapters. All compared models use the same expanding earlier population for each future month. Reproduced models retain seed 2026 and six-thread limits; no package or numerical preset changes occur.

### 2. Domain-correct composition and finite strategy set

Let `p` be the Newton member's finite score in [0,1] and `g` the existing fusion's finite log-odds margin, which is not restricted to [0,1]. For epsilon=1e-6:

```
u = log(clip(p, epsilon, 1-epsilon)) - log1p(-clip(p, epsilon, 1-epsilon))
blend = a*u + (1-a)*g,  a in {0.1, 0.25, 0.5}
```

Thus `a` is always the Newton-member weight. Never pass `g` through a probability-domain logit or the old probability-only fusion helper. Check aligned IDs/labels, source provenance, finite domains, declared weights and exact row-wise arithmetic. Current labels and unrelated rows cannot change inference. Weighted model scores and composed margins are not asserted calibrated probabilities. The fixed common-margin rule avoids fitting another combiner; raw-probability blending and learned calibration are excluded from this finite question.

Retain four original Newton singles, four clipped-logit singles, three old references and twelve blends: **23 selectable strategies**. Save 69 selectable strategy/month predictions plus three nonselectable XGBoost dependencies: **72 total score artifacts**. These comprise 21 base artifacts and 51 derived results (three reference fusions, twelve Newton logits, 36 new blends). Every derived record names its immediate parents, transforms, weight/epsilon and source/model/prediction hashes. Export twelve Newton clipping rows (four sources * three folds), including zero-change cases. No composition miniature needs model fitting; use deterministic synthetic score fixtures for endpoints, extreme finite margins, invalid domains, misalignment and row/label invariance.

### 3. Chronological policies and complete comparisons

Base models train on months 0..m-1 and predict m in {4,5,6}. Apply unchanged pooled/latest/rank-fraction policies: month 5 uses month 4, month 6 uses months 4/5 or latest month 5. Policies optimize positive-class F2 `5TP/(5TP+4FN+FP)` on eligible earlier predictions; there is no imposed review quota or precision floor. The rank-fraction policy does not make composition batch-dependent. Preserve score/ID ties and exact rational ceilings. Select by actual macro month-5/6 F2, then recall, precision and deterministic strategy/policy IDs.

Export **138 forward rows**, **69 macro summaries**, **69 pooled-confusion rows**, **69 score diagnostics** and **433 matched contrasts**:

- 207 against three original pooled references: 23*3*3.
- 46 non-pooled versus own pooled: 23*2.
- 12 logit versus original Newton singles: 4*3.
- 72 blend versus each corresponding common-margin member: 12*2*3.
- 18 spline-minus-linear blends at equal training weight/alpha/policy: 2*3*3.
- 18 balanced-minus-unweighted blends at equal representation/alpha/policy: 2*3*3.
- 36 Newton-weight-pair contrasts within source/policy: 4*3 pairs*3.
- 12 spline-minus-linear singles within raw/logit domain and training weight/policy: 2*2*3.
- 12 balanced-minus-unweighted singles within raw/logit domain and representation/policy: 2*2*3.

Retain named factors, zeros, ties and negative differences. Report F1/F2/F3, precision, recall, confusion counts, flagged count/fraction, AP and ROC AUC. AP or pooled F2 cannot replace the macro objective. Export twelve score-diversity rows (four common-margin Newton sources versus fusion on three folds, Pearson/Spearman and score ranges) and 24 joint-error rows (four transformed members versus fusion at the same three earlier-only policy types on months 5/6). Undefined correlations for constant vectors retain an explicit undefined flag rather than being invented as zero. Correlations/joint outcomes are descriptive, never inference inputs or adaptive weight selection.

### 4. Winner-type roles and limited transfer

Always freeze five roles: selected, fusion/pooled, balanced/pooled, unweighted/pooled and same_model (winner/pooled).

- Original Newton winner: add matched_representation and matched_weight originals using the winning policy (seven roles).
- Transformed Newton winner: add corresponding transformed matched_representation/matched_weight and raw_member at winning policy (eight roles).
- Blend winner: add matched_representation/matched_weight blends at the same alpha/policy, transformed_newton_member, fusion_member and raw_newton_member at winning policy (ten roles).
- Old reference winner: no extra roles (five roles).

The full 2*2 Newton source square ensures every matched role exists. Keep aliases for comparisons and deduplicate score/model work. Freeze selected identity, every dependency, complete development outputs, code/config/software and final rules from months 4/5/6 (latest uses 6) before recorded further consumed-month access. Only these roles transfer. With cached references available, final work uses three reference base reuses and at most three new Newton fits trained on months 0-6; no new tree fit or losing Newton/alpha sweep occurs. Record actual counts, including any exact-recipe absent-reference reproduction.

The broader goal requires positive measured F2 differences against fusion, balanced and unweighted on both development and consumed transfer. Clear-gain language additionally needs positive conditional interval bounds for its named scope. Failed comparisons remain explicit; neither positive estimates nor intervals restore an independent holdout.

### 5. Reproducible evidence and completion

Use 500 paired fixed-decision row-bootstrap draws per month 5/6/7, seed 1027. Share sampled rows across frozen roles; macro differences average separately resampled monthly F2 differences. Retain 1,000 development and 500 consumed draw rows, aliases included. For N roles there are 3*(N-1) development interval summaries and N-1 consumed summaries: 12/18/21/27 and 4/6/7/9 for N=5/7/8/10. Independently reconstruct all draws/quantiles through direct sampled Boolean decisions. State fixed-model/policy conditioning, adaptive C/source choice and repeated search, selection, excluded retraining/temporal/generative uncertainty and lack of multiplicity adjustment.

Independently reconstruct every base inference, composition, clipping count, operating rule, metric, contrast factor/arithmetic, diversity/joint-error row, selected roles and freeze/access record. Inspect an operating-results/composition-effect figure; source smooth-effect figures remain those of the frozen Newton study. Findings must distinguish a standalone model gain, transformation effect, composition effect and policy change. Verify copied report hashes and all preserved prior identities.

README must include pinned dependencies/data checksum, eleven selection-restoration commands, optional historical-cache handling, exact required absent-cache reproduction, ordered verification/development/freeze/transfer/report stages, seeds and destinations. Update measured context, acknowledge substantive Codex/API assistance and run relevant tests, compileall and strict OpenSpec validation. Sync/archive only after real outputs and tasks verify; leave the performance goal active if measured acceptance fails.

## Risks / Trade-offs

- Similar predictive errors can eliminate complementarity → retain both members, complete matched controls and joint-error evidence; do not presume a blend gain.
- Mixed score domains can create invalid transforms → accept the fusion as a margin, transform only checked Newton scores and verify numerical reconstruction/endpoints.
- Equal common-margin weights do not imply equal calibration → retain fixed alpha controls and describe margins as scores.
- Adaptive C/source choice and consumed-month reuse increase optimism → expose that history and conditional limits rather than claim fresh confirmation.
- Cache provenance can be incomplete on fresh clones → reproduce only wholly absent required recipes in the new destination and reject incompatible/partial caches.

## Migration Plan

Introduce only new files/destinations. Verify preservation and composition fixtures, prepare exact required models, run complete development and independent audits, freeze type-determined roles, perform bounded transfer and finish reporting. Any numerical/provenance failure retains evidence and incomplete tasks without changing recipes. Synchronize the new capability and archive only after verified completion.
