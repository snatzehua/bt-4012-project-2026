## Context

See `proposal.md` for the question. The preservation parent is the completed shallow-CatBoost study, `configs/shallow-catboost.json`; individual reference adapters still originate from `configs/bagged-anchor-fusion.json`. These ancestry roles are distinct. Nineteen studies and repeated month-7 access make this an adaptive exploratory experiment. All settings below are analyst-selected, open to group review.

The development-only first-study EDA reports month-0 versus month-6 KS distances of 0.903312/0.619099/0.572748 for velocity_4w/24h/6h and 0.356031/0.360701/0.110224 for zip_count_4w/date_of_birth_distinct_emails_4w/bank_branch_count_8w. The last group is a coherent local-count intervention, not a claim that its members are the three most shifted or that any shift causes fraud-model failure. Existing producers/configs contain no controlled removal of these groups. Earlier native-versus-engineered and capacity searches do not answer this feature-removal question.

`planning-enumeration.py` enumerates exact strategy, factor, role and dependency identities without new fitting, inference or policy evaluation. Its evidence freezes 2020 mandatory hashes, nineteen original/exported selection pairs, all currently present historical ignored files, 64 planned scientific source paths, Python/package/data identity and required-original cache presence. Cache presence is not saved-inference verification. Three project-context snapshots preserve the parent's execution-time README/AGENTS/OpenSpec context before future mutable documentation edits. Planning leaves project code/config/protocol and all parent artifacts untouched.

## Goals / Non-Goals

**Goals:** isolate two fixed feature-group removals under the original model/representation/population/cost recipe; test whether reduced-feature native margins improve actual future F2 directly or complement the fixed incumbent; retain matched full-feature margin controls and all negative effects.

**Non-Goals:** data-driven feature selection, new group definitions, engineered replacements, resampling, recency interventions, hyperparameter tuning, learned blends, calibration, current-label batch fitting, random-split evidence or restored holdout independence. Feature removal changes available information and encoded width, not training size. No optional extension is bundled.

## Decisions

### 1. Isolated ancestry and versioned preservation

Apply adds only the isolated config/protocol, three scientific modules named in the proposal, `tests/test_context_ablation.py`, a separate `scripts/check_context_ablation.py` and supporting exports. New destinations must not overlap any historical output/report root. Inherit the parent's 61 scientific sources and add three, giving 64. Freeze exact configuration, protocol, source closure, software and prospective planning hash before scientific work. Preserve stopped-solver and separate Newton fit/analysis identities.

Use the completed shallow study as preservation parent and its explicit original bagged-anchor ancestry as the reference adapter. Never supply an adapter view to a whole-study predecessor guard or execute any old grid. Original individual source access is complete own cache, original bagged-anchor origin, then its declared earlier origins. Inspect only the eight bases required by the six-reference DAG. A wholly absent individual source may reproduce its exact original recipe in the new destination after unchanged gates; an inspected partial/corrupt/incompatible set fails in place without overwrite/fallback. Derived reference scores retain exact arithmetic, original domains and aligned IDs/labels.

Historical report/run manifests are execution-time records. Before updating mutable root documentation, retain the exact three prospective snapshots and map only those original context entries to their recorded immutable versions for historical verification. Never rewrite a parent manifest or use this mapping for scientific sources, configs, protocols, selections, models, predictions or closed logs. Before documentation changes, verify all parent indices directly; afterward verify those three historical context entries against snapshots and every other historical member against its original unchanged path. This prevents normal project-context maintenance from concealing scientific drift. Export the mapping and check it during clean-copy reproduction and final closure.

### 2. Fixed two-factor representations and original model controls

Start from decoded BAF Base-v2 predictors via the original sentinel rules. Target, month and constant device_fraud_count remain excluded. The feature interventions are:

| Representation | Removed original numerical fields | Retained raw width |
| --- | --- | ---: |
| full | none | 29 |
| drop_velocity | velocity_6h, velocity_24h, velocity_4w | 26 |
| drop_counts | zip_count_4w, bank_branch_count_8w, date_of_birth_distinct_emails_4w | 26 |
| drop_both | union of the two lists | 23 |

Retain original column order after removal and all five original categoricals. Reject target/month/excluded/engineered inputs or altered group lists. Removed fields must not reappear as transformations, missing indicators, cross-features or training-derived substitutes. Changing their values cannot alter fitted preprocessing or predictions when the retained population/values are fixed.

Use the original sparse one-hot baseline pipeline: numerical median imputation with missing indicators and keep-empty features, no scaling; training-fitted categorical vocabulary, unknown categories ignored. Numerical medians, indicator locations, categorical levels, transformed feature names/order and width use eligible training rows only. Retain original handling of missing categorical values and all-missing numericals; do not silently substitute native-categorical inputs. Logistic/spline references preserve their original representations.

For every representation compare unweighted and original `class_weight='balanced'` LightGBM, without separate sample/scale/automatic class weights. Balanced per-class costs are n/(2*n_class) from the complete eligible training population; both representations see identical ordered training IDs/classes. The exact original recipe is 200 requested rounds, learning_rate 0.05, num_leaves 15, min_child_samples 100, deterministic True, force_col_wise True, verbosity -1, seed 2026 and six threads. All remaining effective settings must match the frozen original full-feature source, including ordinary defaults. Full-feature controls reuse those saved baselines, not newly tuned equivalents. Six reduced-feature recipes make eighteen full development fits. No eval_set, early-stopping callbacks or current-label best iteration is allowed. Full-size models must fit 200 rounds; failure retains evidence. Miniature models may naturally terminate earlier if no valid splits remain; record actual booster iteration/tree count, require 1..200 and agreement with persisted metadata, and distinguish this from evaluation-driven early stopping.

Primary documentation: [LightGBM classifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) documents requested/actual iterations, raw scores and class-weight semantics, including multiplication by sample weights. The experiment preserves the actual original wrapper cost mechanism and checks it independently. [Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) documents the fixed deterministic/column-wise settings. Neither source implies feature-removal gains. Codex assistance and these sources must be acknowledged; no public notebook is copied.

### 3. Native margins, full-feature controls and fixed blends

Pure IDs are `context_<representation>_<balanced|unweighted>`. For reduced models store finite native LightGBM raw margins from the saved fitted pipeline; for the two full controls reconstruct the raw margins from the verified original full-feature pipeline. Full controls are six newly calculated margin scores over three months, not new fits and not existing probability-score aliases. Independently require expit(raw_margin) to equal the original/model positive-class probability at absolute 1e-12, relative zero. Probability rounding or saturation can merge ties: do not assume probability/native-margin policies have identical decisions, or rename such differences as feature-removal gains.

For each of eight pure sources add alpha 0.10/0.25/0.50 blends named `context_anchor_<representation>_<weighting>_a<10|25|50>`. Evaluate float64 `(1-alpha)*M + alpha*raw_margin` in that association. M is `bagged_anchor_balanced_logit_a25`, the current unbounded margin. No new sigmoid/logit/clipping/normalization/calibration/learned transform enters selectable scores. Alpha zero aliases M; alpha one aliases the exact pure native margin. All scores require aligned ordered IDs/labels and finite declared domains.

Retain six selectable original reference scores and their fixed comparator policies: bagged_anchor M/pooled; latest_fusion interaction_blend_spline_c1p0_balanced_a10/latest; recent_fusion margin_blend_spline_c0p1_unweighted_a25/latest; fusion logit_mean_xgb_d4_w1_baseline_balanced_a50/pooled; balanced baseline_balanced/pooled; unweighted baseline_lgbm/pooled. All selectable scores get all three policies. These six reference DAGs share eight old bases and five old derived scores; original probability/margin domains stay unchanged. The union is 6+8+24=38 strategies, with no alpha-zero duplicates.

### 4. Chronology, policies and actual F2 objective

Models for months 4/5/6 train on all strictly earlier applications; each comparator predicts the same full future population. Month-5 policies learn from month-4 out-of-time scores/labels. Month-6 pooled/rank policies learn from months 4/5; latest learns from 5. Final models train on 0-6, with rules learned from score months 4/5/6 (latest6), frozen before additional month-7 scoring.

Use unchanged pooled/latest maximum-F2 thresholds and learned rank fraction, equal-score groups, rational ceiling and stable row-ID tie handling. Select by actual mean month-5/6 positive-class F2 beta=2, then macro recall, macro precision, candidate/policy IDs. F2=5TP/(5TP+4FN+FP), rather than support-weighted F1. No precision floor or review quota, including the earlier 1%, is imposed. AP, pooled F2 or consumed-month outcomes cannot select/replace the winner. Report original precision/recall/F1/F2/F3, confusion counts, missed fraud, legitimate flags, flagged fraction, AP, ROC-AUC, MCC, balanced accuracy/FPR/accuracy; keep source score domains explicit.

### 5. Complete effects, diagnostics and measured work

With source caches available, development has 18 new reduced-feature fits, 24 old base reuses, fifteen original derived reuses, six new full-native-margin calculations and 72 new blends. Forty-two model objects use45 model files (27 reference files, 18 new pipelines); 45 physical scores per month give 135. Export 228 forward operating rows, 114 macro and 114 pooled summaries.

The exact1192 contrast factor identities are prospectively enumerated:

| Contrast | Rows | Held fixed / definition |
| --- | ---: | --- |
| fixed_reference | 684 | 38 strategies*3 policies*6 fixed-policy references |
| own_policy | 76 | latest/rank minus own pooled |
| remove_velocity | 48 | both count states*2 weights*4 pure/interior alpha states*3 policies |
| remove_counts | 48 | both velocity states, other factors matched |
| remove_both | 24 | both removed minus full at same weight/alpha/policy |
| removal_interaction | 24 | F2(both)-F2(velocity)-F2(counts)+F2(full), same weight/alpha/policy |
| balanced_minus_unweighted | 48 | 4 representations*4 alpha states*3 policies |
| pure_vs_anchor | 24 | 8 pure sources*3 policies |
| blend_vs_pure | 72 | 24 blends*3 policies |
| blend_vs_anchor | 72 | same |
| higher_alpha | 72 | 8 sources*3 interior-alpha pairs*3 policies |

The four-term interaction is an interaction between retrained model-performance interventions, not evidence of causal relationships between variables. It must be reconstructed from four explicit keyed terms, not an incorrectly reused two-row subtraction. All corresponding metric differences, including negative effects/ties, remain visible. The same groups/weights/populations/200-round settings hold across deletion controls; removal does not isolate encoded width, numerical defaults affected by data dimension, runtime, histogram/binning effort or exact model capacity. Export complete requested/effective parameters and report those limitations.

Export 135 physical and 114 selectable score diagnostic rows with min/max, finite domains, unique count/tie fraction, out-of-unit counts and quantiles 0/.01/.05/.25/.5/.75/.95/.99/1. Export 42 diversity rows: all 8 pure sources versus M and 6 reduced sources versus their same-weight full margin, across 3 months; constant members retain null correlation with explicit reason. Export 516 decision-overlap rows:8 pure versus M, 6 reduced versus full, 24 blends versus M/own pure/same-weight full blend, across 3 policies/2 months. Retain self/endpoint aliases visibly and verify fraud/legitimate both/only/neither counts.

Export 42 model/work rows, 18 new preprocessing/parameter records and six full-margin-control provenance records, with original model reuse, actual full-size fits, model bytes, training rows/classes, requested/fitted rounds/trees, logical row-round exposure, wall time, high-water RSS and inference calls. Source spline iterations and unavailable historical resource measurements remain separate. Do not treat nominal rounds as equal runtime/work or count score reconstruction as training.

### 6. Methodological, source, numerical and resource gates

A separate complete test runner observes12 planned current tiny fits: six reduced-feature formats and six original formats (balanced/unweighted ordinary/bagged LightGBM, spline, XGBoost). Reuse the two ordinary tiny full sources for full-native-margin checks without extra fits. The existing full suite implies103 historical calls, to be actually observed and reported rather than asserted as passed. Preserve failed/focused/full/clean ledgers and original sources; do not change predecessor tests/runners. Cover all 114 winner/policy cases, exact factor identities/counts, corruption/partial-cache/whole-absence behaviour, removed-column perturbation invariance, actual OHE/imputer/cost/parameter/inference persistence, source ancestry and no unrelated execution.

Audit the preserved nine-fit sampling/eight-fit original-source gates using the explicit adapter view; preserve all completed shallow-CatBoost evidence without rerunning its unrelated fourteen-fit gate. New production miniature work is eight fits: all six reduced recipes plus deterministic repeats of drop_both/balanced and drop_velocity/unweighted. Use the original 29-column fixture before removal, both labels, missing/all-missing numerical and unseen/missing categorical values. Verify exact removal sets/schema/order, earlier costs, raw-score/probability consistency, actual rounds/warnings and saved/reordered/batched/single/unrelated-row/current-label/removed-column inference at absolute 1e-12/relative 0. Preserve failed started/completed work without relaxed recipes or retries that erase evidence.

For actual-host feasibility use predictor-only development rows. Let n=903157; d_source be the largest verified required original encoded-tree/additive-spline width and d_new the maximum training-fitted original/full/reduced encoded width. Define CSR(n, d)=12*n*d+8*(n+1), source_workspace=4*CSR(n, d_source)+8*(d_source+1)^2+4*1024^3, and total=source_workspace+4*CSR(n, d_new)+64*n. Verify every allocation and reject total>70% actual RAM. Export actual fitted dimensions, deep input memory, timing and high-water RSS. This conservative feasibility proxy is not a guaranteed native allocation ceiling; retain actual failures. Gate preprocessing may fit training-only representations, never models or policies.

### 7. Frozen winner-dependent transfer

Reference winners retain eight visible roles: selected, six references and same-strategy pooled; only 13 old scores/eight bases/nine files transfer and no new reduced model is needed. New pure/blended winners retain 19 visible roles: those8, four same-weight representation controls at winner alpha, two same-representation weighting controls at winner alpha, and five alpha roles0/10/25/50/100 at winner representation/weight. Every control uses winner policy; pure winners have alpha 100. Aliases remain visible and share work.

| Winner | Roles | New reduced recipes | Full-native controls | Full scores | Base objects/files |
| --- | ---: | ---: | ---: | ---: | ---: |
| original reference | 8 | 0 | 0 | 13 | 8/9 |
| pure full control | 19 | 3 | 2 | 21 | 11/12 |
| pure reduced | 19 | 4 | 1 | 21 | 12/13 |
| blend full control | 19 | 3 | 2 | 25 | 11/12 |
| blend reduced | 19 | 4 | 1 | 25 | 12/13 |

Pure winners need three interior blends; blend winners need seven, after deduplication. Required new reduced recipes are the three reduced representations at winner weight, plus alternate weighting of winner representation when that is reduced. Original full baselines already exist in the reference DAG. Check all 114 prospective cases and actual unique operating pairs. Tighten new-fit bounds for complete compatible new caches and original-reference bounds for actual inspected availability; at most3 bagged/2 spline/3 other-tree reproductions are possible if wholly absent, zero when present/audited. Freeze roles, exact DAG, original-cache hashes, rules and per-type fit/file/score bounds after independent development audit and before additional consumed scoring. No losing-grid sweep, winner replacement, hindsight threshold or fresh-confirmation claim.

### 8. Independent reporting and acceptance

Independently reconstruct actual source/new training populations, weights, removal sets, preprocessing/feature names, full effective settings, underlying persisted inference and raw/probability consistency. Check all 42 saved development models/135 scores and every table/factor/rule/mask/metric/work record before freeze; afterward audit only the frozen transfer DAG. Full-native controls must reference actual original serialized models, not logit-converted stored probabilities or metadata-only reuse.

Reconstruct500 paired fixed-decision row-bootstrap draws for each of months 5/6/7, seed1027; keep original policies/masks fixed, use paired rows across roles, and macro-average month 5/6 draws. Export 1500 draws with 28 intervals for reference winners or72 for new winners: every other role against selected in month 5, month 6, macro, consumed scopes. Verify every draw/direction/quantile and exact-zero aliases. These conditional intervals exclude retraining, adaptive group/source/model/policy/selection search, temporal/generative variation and multiplicity.

Produce six panels and exact value exports:114 macro precision/recall operating points; eight canonical five-endpoint alpha curves for each policy (24 curves/120rows); all 144 deletion/interaction effects; six-reference intervals in macro and consumed scopes. Check plotted artist arrays/exported values at absolute 1e-12/relative 0, regenerate and visually inspect. Floating CSV serialization need not be byte-identical; preserve any failed verifier assertions without changing scientific tolerances.

Broader achievement still requires strictly positive selected-minus-all-six-reference F2 in macro development and consumed exploratory transfer; incumbent aliases/ties fail. Claim a feature-removal gain only if the reduced winner additionally beats its same-weight full margin/control blend at identical alpha/policy in the named scopes; a full-feature winner cannot support that claim. Clear-gain wording additionally needs positive conditional interval lower bounds for named comparisons. Finishing this finite search alone cannot complete the broader goal, and even pointwise gains remain exploratory after nineteen studies/consumed access.

### 9. Reproduction and closure

Document the actual implementation interfaces in checked order: pinned environment/Base; nineteen exact selection restores/changed-record rejections; context-version mapping and prospective declaration; complete checks/ledgers; resource/source/miniature gates; development; producer and independent audits; freeze; bounded exploratory transfer; independent report/bootstrap/figure; preservation/closed-log/run/report indices; strict OpenSpec validation and verified sync/archive. Proposed names are verify_declaration/record_checks/preflight/miniature/prepare_development/validate/audit_development/freeze/exploratory/report, but README must use actual implemented interfaces, not assert this plan was executed.

Run the exact documented setup and full methodological suite in a disposable copy without previous models/predictions. Verify all 27 pins/Python/Base, 19 exact restored selections/rejections and archived-planning resolution. Shared existing runtime/data are permissible if disclosed; claim fresh installation/full-size clean retraining only if performed. Preserve execution-time snapshots of mutable context in both setup and final historical-manifest auditing. Count current/historical tests, eight production miniature fits, eighteen full new fits, exact missing-source reproduction, references and score-only calculations separately. Hash only closed logs, exclude indices from self-inclusion, verify all members, and audit every requirement/scenario/task before sync/archive. Do not rewrite frozen parent reports to make them current.

## Risks / Trade-offs

- Removing informative signals can hurt F2 → preserve complete factorial controls, pure/blended outcomes and all losses.
- The groups were chosen after previous results → declare adaptive origin and fixed membership now; no later feature search within this identity.
- Marginal shift need not imply harmful concept drift → interpret matched performance evidence narrowly and avoid causal claims.
- Raw/full probability policy differences can be numerical → keep original probability references, native-margin controls and direct mask checks separate.
- Smaller schema changes binning/width/work despite matched training settings → report full effective parameters, encoded names and measured work without equal-capacity claims.
- Historical mutable docs can invalidate naive run-index checks → preserve exact pre-edit context snapshots and explicit version-only mapping while all scientific paths remain unchanged.
- Conditional bootstrap omits adaptive-search/generalization uncertainty → retain exclusions and consumed status even for positive effects.

## Migration Plan

This is additive. Planning creates no scientific configuration, implementation, fit or score. Apply freezes a new isolated identity; prior scientific/model/result files and original indices remain unchanged. A failed gate stops execution in place with evidence retained; a protocol remedy needs another study. Sync/archive only verified finite work. Update current project context during apply using the recorded historical context versions; do not alter archived historical content.
