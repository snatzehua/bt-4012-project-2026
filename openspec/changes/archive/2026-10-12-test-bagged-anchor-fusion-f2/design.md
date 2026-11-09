## Context

See proposal.md for motivation. The previous finite bagging study is complete and archived with 14/14 verified tasks: 18 new development fits, 15 reference reuses, 54 score results, 90 forward rows and 342 contrasts. Best new macro F2 was 0.3278848931; the existing anchor remains selected at 0.3307932537. Consumed F2 remains 0.3715337955 versus unweighted 0.3768844221. Its 52-source execution identity, 94-file final report manifest, original 72-file generated manifest and 210 ignored run artifacts are frozen. Preserve all sixteen completed histories, separate Newton fit/analysis identities and the stopped L-BFGS attempt.

Base v2 is at `data/raw/baf/Base.csv`, SHA256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. Python is 3.12.9; the same 27 pins across the three requirements files suffice. Month 7 is already consumed. Proposal, feedback and source-model limitations remain unchanged.

The development-only planning diagnostics in planning-evidence.json use existing source scores and earlier-only policies, not new blends. Under the latest policy, balanced probability means catch 15 frauds missed by the anchor in each of months 5/6, while missing 33/30 caught by the anchor. Unweighted mean-logit controls catch 8/71 additional frauds but also flag 157/1651 additional legitimate applications. Rank correlations are approximately 0.988-0.993. These show some different errors alongside high similarity and false-alarm costs; they do not establish an ensemble benefit. All four ensemble forms remain included symmetrically. Adaptive use of these prior development outcomes is disclosed.

## Goals / Non-Goals

**Goals:** Test whether fixed additions of existing randomized sources improve actual chronological F2; separate addition effects from the unchanged anchor, individual seeds, aggregation, score transform, class weighting and threshold policy.

**Non-Goals:** New model, feature, seed, row fraction, tree capacity, anchor, learned-weight or constraint search; optimizing AP; causal drift identification; replacing a losing development winner using consumed outcomes; claiming an independent holdout. No improvement is assumed and no material design question is deferred.

## Decisions

### 1. Separate identity and exact source preparation

Use the isolated configuration/protocol/modules/destinations listed in proposal.md. `planning-enumeration.py` mechanically verifies the declaration without fitting or evaluating a new blend. The previous 52-source closure gains three planned modules, for **55** distinct paths. The exact mandatory union extends the previous 634 paths by the previous three modules, configuration, protocol, original selection, final report manifest and all 94 indexed final report files: **735 paths**. Verify **sixteen original/exported selection pairs** and retain the exact available ignored-output snapshot of **3732 files**. Do not modify previous source, config, score, model, selection, test, report or stopped-attempt evidence. Source-miniature model files remain optional on a fresh clone and mandatory if present in the retained snapshot.

The five original references remain:

- latest_fusion: `interaction_blend_spline_c1p0_balanced_a10 / latest`, macro F2 0.3307932536724773;
- recent_fusion: `margin_blend_spline_c0p1_unweighted_a25 / latest`, 0.3290099447918403;
- fusion: `logit_mean_xgb_d4_w1_baseline_balanced_a50 / pooled`, 0.3289065701775199;
- balanced: `baseline_balanced / pooled`, 0.32780157809618937;
- unweighted: `baseline_lgbm / pooled`, 0.3226536849212962.

Their five base dependencies are XGBoost depth4/unweighted, balanced/unweighted original LightGBM and Newton spline C0.1/unweighted plus C1/balanced. Add the six original bagged LightGBM bases (two weightings, seeds17/29/43). This is **eleven bases per month, 33 development results** for months4/5/6. Current-parent caches contain all 33, so available-cache execution should make zero new full fits. Their 21 existing derived results (nine old reference margins and twelve ensemble aggregates) are also checked/reused or reconstructed exactly in the new destination, recording actual work. Do not rename arithmetic or count a cached audit as a new fit.

Inspect the new destination first, then the current bagged-study parent. For old references only, inspect their declared individual older origins in the preserved order after whole absence. For bagged sources the single historical origin is the current parent. A partial or incompatible inspected set fails immediately; do not skip it, overwrite it, rerun an ancestor grid or change recipes. Wholly absent sources reproduce individually in the new destination. Fresh complete-cache absence can require **18 exact bagged plus 15 exact old-reference development fits**. Record actual per-type fits, reuse origins, source/config/software/training/serialized hashes and elapsed/RSS evidence.

All source settings stay fixed. Bagged training retains original numeric median/missing-indicator and one-hot categorical preprocessing, whole eligible rows and balanced class weights when declared; effective fraction0.8/frequency1, seeds17/29/43, model seed2026, six threads, deterministic/column-wise CPU fitting and200 rounds/15 leaves remain unchanged. Old XGBoost/native and Newton additive inputs retain their original preprocessing/numerical identities, including separate fit/analysis histories. Independently reconstruct actual feature roles, training medians/categories/knots, whole-training class weights, requested/effective parameters and eligible train/evaluation IDs; verify saved inference within absolute1e-12, relative zero. Reproductions retain all warning/convergence/allocation gates with no fallback or recipe relaxation.

### 2. Fixed arithmetic and canonical controls

Let H be the preserved score of `interaction_blend_spline_c1p0_balanced_a10`, independently of its operating policy. Its original association is retained: 0.1 times the balanced C1 spline logit plus0.9 times the existing equal XGBoost/balanced-LightGBM logit margin. Never apply a probability logit to H.

For each weighting w, use the three original ordered float64 seed probabilities p17,p29,p43. Preserve the original probability mean P=(p17+p29+p43)/3 and mean-logit margin L=(logit(clip(p17))+logit(clip(p29))+logit(clip(p43)))/3. Epsilon is1e-6; original source mean arithmetic and seed order remain byte-reconstructible.

Two additions are Gprob=logit(clip(P)) and Glogit=L. They are distinct: logit(mean(p)) is not mean(logit(p)). For lambda in **0.1,0.25,0.5**, evaluate exactly **S=(1-lambda)*H+lambda*G** using that association in float64, without flattening it into a differently associated leaf sum. The three weights test small/moderate additions to the stronger anchor; they are fixed analyst choices, not optimal values. Larger weights or another anchor are separate experiments.

Declare IDs `bagged_anchor_<balanced|unweighted>_<prob|logit>_a10/a25/a50`: **twelve new blends**. Keep all **25 controls**: five original references; six native seeds; four original aggregates; ten `logit_<source>` controls covering six seeds, two probability means and two unbagged baselines. The canonical `logit_baseline_balanced` and `logit_baseline_lgbm` retain their existing mathematical identities/arithmetic from the preserved helper. Do not logit the already-logit mean. Displayed lambda0 uses H's existing ID; lambda1 uses the existing G control's ID. Neither is a duplicate new strategy.

Validate exact member completeness/uniqueness and row/label alignment. Reject invalid probabilities, nonfinite margins, wrong dimensions, undeclared epsilon/weights and arithmetic changes. Current labels, unrelated rows and batch statistics do not enter composition; no learned weights or normalization. Preserve clipping/tie effects and equivalent decisions honestly. Probability-like values are not asserted calibrated, and margins are not probabilities or costs.

Export each blend's canonical expression. For Glogit use three original anchor leaf logits plus three seed logits; for Gprob treat the logit-probability-mean as a nonlinear expression atom alongside the three anchor leaves. This yields **60 expression-leaf rows** (six Glogit blends times six atoms, plus six Gprob blends times four). The diagnostic expansion must agree within tolerance but must not replace the declared evaluation association.

### 3. Complete chronological grid and matched comparisons

There are **37 selectable strategies**. Each future month has40 physical scores: eleven bases, seven preserved parent-derived results, ten new logits and twelve new blends. Development therefore has **120 physical /111 selectable scores**, with **54 inherited results and 66 new calculations**. Three policies pooled/latest/rank_fraction yield **222 operating rows** for months5/6 and **111 each macro, pooled-confusion and monthly-score summaries**. Source models use months0..m-1; month5 rules use month4; month6 rules use months4/5 or latest5. Final rules use4/5/6 or latest6. No random-split result establishes drift.

Optimize actual positive-class F2=5TP/(5TP+4FN+FP), macro-averaged over months5/6, then recall, precision and deterministic IDs. Preserve largest-threshold, smallest-count, stable-row-ID and rational review-fraction ceiling ties. Rank fraction is learned earlier, not a 1% quota. Report F1/F3, AP/ROC, workload, precision/recall and confusion counts; AP, pooled F2 and diagnostics cannot select an alternative. No quota or precision floor is added.

Independently verify **947** unique factor identities:

| Factor | Rows |
| --- | ---: |
| Each strategy/policy versus five fixed operating references | 555 |
| Nonpooled minus own pooled | 74 |
| Blend minus unchanged anchor at the same policy | 36 |
| Blend minus added margin at the same policy | 36 |
| Higher minus lower lambda, all three unordered pairs | 36 |
| Blend balanced minus unweighted | 18 |
| Blend Glogit minus Gprob | 18 |
| Each logit control minus its native source | 30 |
| Added Glogit minus Gprob | 6 |
| Native matched balanced minus unweighted | 15 |
| Logit matched balanced minus unweighted, including baselines | 15 |
| Probability mean minus each native seed | 18 |
| Mean-logit aggregate minus each logit seed | 18 |
| Logit-probability-mean minus each logit seed | 18 |
| Randomized controls versus same-representation unbagged control | 54 |

Keep ties, negative effects and repeated numeric outcomes under distinct valid factors. The randomized-versus-unbagged controls match raw/margin representation: raw seed/probability means use native baseline, while logit controls and mean-logit margins use logit baseline. Native/logit equivalence is not a new gain.

Export **18 diversity rows** for six declared pairs across months4/5/6: H versus each of four G margins, plus Gprob versus Glogit within each weighting. Export **36 decision-overlap rows** for those six pairs, three policies and months5/6, including joint caught/missed fraud and legitimate applications. Correlations use explicit null/reason for constant scores. Export **39 clipping rows** for eleven probability-like bases plus two probability means per month. Preserve all 33 source audit records,18 bagged preprocessing/parameter blocks and15 old-reference numerical/preprocessing blocks, plus111 monthly score summaries. Diagnostic values cannot change source, seed, lambda or selection choices.

The independent grid-specific auditor reconstructs all 120 physical results, exact key/factor sets, original source preprocessing/inference/numerical evidence, aggregates, logits/blends, expression atoms, earlier-only rules, masks/metrics/summaries, diagnostics and actual macro winner before freeze. Analytical full-grid fixtures plus corruption checks must verify missing/duplicated/replaced keys and actual values, not just row counts.

### 4. Gates and explicit work accounting

Before any full reproduction, pass the complete methodological suite and actual-host predictor-only allocation gate for the largest eligible month0..6 train population, unchanged 70%-RAM allowance and 8192-row reference-verification chunk bound. Preserve the previous exported nine-fit sampling miniature gate and its earlier source ancestry, verifying frozen source identity, active/inactive sampling, repeatability, diversity and persistence. Present optional old miniature files must match; a wholly absent ignored miniature set must not require historical fitting. **No new production miniature fits** are declared because source recipes do not change.

Each complete current-study methodological invocation includes **six separately counted genuine tiny source-cache fits**: two bagged weightings and four reference formats (balanced/unweighted Newton spline, XGBoost and original LightGBM). They exercise the new destination's absent/cache-reuse interface, independent preprocessing/inference/numerical validity, partial/corrupt rejection and no refit on an audited cache. Include all 111 winner/policy cases, label/row/endpoint/tie invariance, complete analytical-grid exports and meaningful corruptions, failure retention and conditional-role aliases. Historical test-module fits and all retries/failures remain separate. Retain exact test sources, receipts and fit ledgers; never overwrite selection-era evidence after freeze.

### 5. Predetermined bounded exploratory transfer

Always keep selected, the five fixed operating references and same_strategy/pooled: seven roles. Original reference identities have precedence, even if their score also belongs to another descriptive category. For each nonreference class, retain aliases and deduplicate only work. The exact mappings and DAGs for **111 cases** are in planning-evidence.json:

| Winner class | Cases | Roles | Maximum bases | Maximum physical scores | Maximum bagged fits if absent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original reference | 15 | 7 | 5 | 8 | 0 |
| Canonical unbagged logit | 6 | 9 | 5 | 10 | 0 |
| Native/logit single bagged seed | 36 | 21 | 9 | 20 | 4 |
| Native/logit bagged aggregate | 18 | 21 | 11 | 23 | 6 |
| New anchor blend | 36 | 26 | 11 | 28 | 6 |

A baseline-logit winner adds native_member and matched_weight. A single/aggregate winner adds all three native and logit seed roles, probability_mean, probability_mean_logit, logit_mean, unbagged_member, unbagged_logit, native_member, transformed_member and matched_weight. Matched weighting retains seed/aggregate/representation; all constituent and aggregation roles use the winner policy.

A blend winner keeps the same native/logit seed, aggregate and unbagged roles, and adds anchor_at_policy, added_member_native, added_member_margin, matched_weight at the same form/lambda, matched_form at the same weighting/lambda, and alpha_a10/a25/a50 at the winner weighting/form. These are exactly26 visible roles, including aliases. Opposite-weight constituent scores are dependencies of its aggregate; no extra losing-grid operating rule is evaluated outside the frozen roles.

Freeze actual winner/rules, all role mappings, exact source/score DAG, immutable development hashes and per-type existing-cache bounds before any further month7 access. Current parent contains all five old-reference final bases and no bagged final bases; bounds therefore allow zero old-reference reproductions in this checkout and only the unique required bagged sources. Fresh absence can additionally require two additive and three old-tree reproductions, counted separately. Every case has an exact tighter bound; the maxima are not a blanket allowance to fit extra models. Transfer only those dependencies and roles, without a losing grid, hindsight threshold or winner replacement.

Broader acceptance requires positive measured differences against all five references on both macro development and consumed transfer. Clear-gain language additionally needs positive conditional lower bounds on the named scope. A finite negative result leaves the broader goal active. Consumed month7 remains descriptive; no archive or new directory restores independent confirmation.

### 6. Conditional uncertainty, findings and reproduction

Use seed1027 and500 paired fixed-decision row draws each for months5/6/7: **1000 development and500 consumed draws**. For7/9/21/26 roles, export18/24/60/75 development intervals and6/8/20/25 consumed intervals. Retain aliases and independently reconstruct every draw and quantile. Intervals condition on fixed models/rules and exclude training-seed/retraining, adaptive source/weight/selection/search, temporal/generative uncertainty and multiplicity correction. They are not uncertainty over all training seeds.

Findings must cover all five fixed-reference comparisons and every matched effect, actual reuse/reproduction/test/gate/failure work, precision/recall/workload and clipping/equivalence limitations. Figures show all 111 strategy/policy operating points and matched lambda curves for both weightings/forms, including canonical H/G endpoints without evaluating duplicate strategies. Verify plotted values/legends and regeneration, then visually inspect.

README supplies Python/data SHA and27 pins, sixteen exact selection restores that reject changed records, individual wholly absent source reproduction, and ordered clean-copy tests/receipt -> resource/source-miniature ancestry -> preparation/validation/independent audit -> freeze -> bounded transfer -> report commands. Preserve all original receipts, config/protocol/software/source/selection/access/provenance, closed logs and final run/report hashes. Verify instructions with no prior ignored model/prediction caches in a disposable copy; do not claim full-size retraining there unless it actually happens. Update current context and Codex/API attribution. Synchronize/archive only after every requirement/scenario/task has actual verified output, relevant tests/compile checks/strict OpenSpec validation and preservation pass.

## Risks / Trade-offs

- High source correlation and worse standalone F2 may leave no gain -> preserve symmetric source/member/reference controls and all negative results.
- Mean logits and logit probability means have different scaling -> domain-correct arithmetic, explicit representation controls and earlier-only policy fitting; no calibration claim.
- Many adaptive comparisons can overfit development -> finite prospective grid, complete matched effects, conditional uncertainty and consumed-holdout caveats.
- Legacy source formats can hide changed recipes or caches -> individual full source audits, whole-absence-only reproduction and no ancestor-grid fallback.
- Numerical reassociation can move tied decisions -> preserve original composition association, stable IDs, canonical endpoints and exact score provenance.

## Migration Plan

Complete declaration/preservation/role enumeration first, then implement the three isolated modules and config/protocol. Pass methodological/resource/source-miniature gates, prepare only required sources, execute the full declared chronological grid and independently audit before freeze. Transfer only frozen dependencies, finish checked uncertainty/figures/reproduction and completion evidence, then sync/archive. Retain any exact-source failure without relaxing gates. No implementation or new blend evaluation occurred during planning.
