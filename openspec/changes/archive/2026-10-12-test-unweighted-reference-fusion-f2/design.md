## Context

See proposal.md for motivation and specs/baf-unweighted-reference-fusion-study/spec.md for the contract. The last finite study is complete and archived. Its frozen winner, `interaction_blend_spline_c1p0_balanced_a10 / latest`, scores macro development F2 0.3307932536724773 and consumed F2 0.3715337954939341. It fails the tree-fusion and unweighted consumed comparisons. Its recent-reference development interval and every original-reference consumed interval include zero. New choices are adaptive to those known results; neither further transfer nor conditional intervals create independent confirmation.

The parent is configs/interaction-score-fusion.json, SHA256 `d21f6f5b3c9e2eb4f0426cbfb85d80ae0e5da247b2c348294401a6bcc8f6dd7b`. Its selection SHA256 is `b804b3bf8ff5f24d60ef896435456c99539eca7f32f8cc32eb9c8e293f63d407`; its final report-manifest SHA256 is `56711164b335b83e58bcf7d4047ea86315276f01d0e40869b5adb1661811643d`. All 71 report hashes, 421 parent run-artifact hashes and required local development cache records/predictions were checked during planning. No new addition scores have been evaluated.

## Goals / Non-Goals

**Goals:** Test incremental unweighted-score complementarity with matched balanced additions, fixed weights, preserved anchor identities and chronological actual-F2 selection; verify every required source and result without historical-grid dependencies.

**Non-Goals:** Tune base models/features/pairs, learn addition weights/calibration, alter beta, introduce quotas/floors, select using AP or consumed results, expand the losing-grid transfer, or relabel previous scores as novel. The comparison asks about fixed additions and their operating policies; it does not establish a universal causal effect of training class weights.

## Decisions

### 1. Separate identity and individual source preparation

Use Base v2 SHA256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`, Python 3.12.9, the unchanged three pinned requirements files, model seed 2026, six numerical threads and bootstrap seed 1027. Planned paths are those in proposal.md. The parent's 46-source transitive closure gains only the three new modules, exactly **49** paths. Freeze source/config/software/protocol identity before execution.

The mandatory preservation union is the parent's 451 paths plus its three new modules, config, protocol, original selection, 71 final report files and final report manifest: **529 distinct paths**, including **fourteen original/exported selection pairs**. Snapshot every available historical ignored output before implementation/execution and reject mutation/deletion. Preserve separate Newton fit/analysis histories and the stopped attempt. Old completed planning documents remain historical. Do not invoke a predecessor full-grid initializer/selection loader that requires unrelated ignored artifacts.

For months 4/5/6 require exactly six sources: xgb_d4_w1, baseline_balanced, baseline_lgbm, spline_c0p1_unweighted, spline_c1p0_balanced and interaction_c1p0_balanced. This gives **18 development base results**, locally all available; new local full development fits are zero. Inspect the new destination first, then the current parent cache, then only the parent's declared individual source origins in their existing order. Partial/incompatible inspected sets fail; only wholly absent sets advance. XGBoost includes its native model and training-fitted adapter. Missing exact recipes fit only in the new destination, with actual new/reused counts recorded. Do not edit or monkey-patch frozen modules.

Source preprocessing stays unchanged. Logistic sources retain 29 predictors with 17 smooth, seven discrete and five categorical roles; medians, missing indicators, knots, scales, vocabularies and constant tails use training rows only. The interaction source retains the same three centered/scaled pair blocks and inactive-column rules. Independent verification reconstructs marginal/pair statistics, actual saved feature names, coefficients/sigmoid inference and the full weighted objective/gradient after persistence. Keep Newton-Cholesky L2/l1_ratio=0, tol=1e-6, max_iter=100, intercept, no warm start, zero warnings/fallback, iterations 1..99 and independent gradient maximum <=1.1e-6. Exact tree parameters, native adapters and saved inference remain frozen. Compatible individual source-format audits may be reused; no predecessor hardcoded full-grid table auditor may be applied to this grid.

Matrix construction/audits use chunks at most 8,192. Before any required reproduction, run an actual-host predictor-only preflight on eligible months 0-6 (903,157 rows locally), retaining dimensions/storage/Hessian/time/RSS and the unchanged 70%-of-host-RAM bound. Maximum logistic dimension is 259 locally. Preserve the source eight-fit miniature gate through its exported receipt and ancestry; optional ignored old miniature models are verified if present, without new production miniature fitting. Four genuine tiny absent-cache tests fit interaction, additive, LightGBM and XGBoost/native-adapter formats once each per full invocation; count those four separately from historical test-module fits. Allocation/numerical failure retains evidence and prevents retries/completion without a separate changed study.

### 2. Fixed anchors, additions and canonical arithmetic

Let `L(p)=log(clip(p,1e-6,1-1e-6))-log(1-clip(p,1e-6,1-1e-6))`, validating finite probability-like inputs in [0,1] before clipping. Preserve four finite-margin anchors:

| Tag | Canonical ID | Exact source recipe |
|---|---|---|
| tree | logit_mean_xgb_d4_w1_baseline_balanced_a50 | 0.5 L(XGBoost) + 0.5 L(balanced LightGBM) |
| recent | margin_blend_spline_c0p1_unweighted_a25 | 0.25 L(additive C0.1/unweighted) + 0.75 tree |
| latest | interaction_blend_spline_c1p0_balanced_a10 | 0.10 L(additive C1/balanced) + 0.90 tree |
| interaction | interaction_blend_interaction_c1p0_balanced_a10 | 0.10 L(interaction C1/balanced) + 0.90 tree |

For each anchor g, addition b in {baseline_balanced, baseline_lgbm} and lambda in {0.10,0.25,0.50,0.75}, compose `lambda*L(p_b)+(1-lambda)*g`. Lambda is the added-LightGBM weight; it is not confidence, fraud cost or review capacity. Anchors are already margins: never logit them or constrain them to [0,1]. Scores use no current labels, learned weights, batch ranks or statistics. Reject non-finite arithmetic, bad dimensions/IDs/labels, domains and undeclared weights. Changing aligned labels or unrelated applications cannot change a row's score; a separately learned rank-fraction operating policy can depend on its unlabeled batch.

Retain all four anchors, both native LightGBM scores, both transformed LightGBM singles and the 32 additions: **40 selectable strategies**. Use `lgbm_blend_<anchor-tag>_<balanced|unweighted>_l10/l25/l50/l75`, except tree/balanced/lambda=0.50 retains `logit_mean_xgb_d4_w1_baseline_balanced_a25`. This exact alias is 0.25 L(XGBoost)+0.75 L(balanced), where historical alpha names the XGBoost weight and new lambda names the added balanced weight. It has one selectable identity.

Floating-point association can differ between the nested and flattened formulas. Evaluate this alias using the frozen direct historical 0.25/0.75 arithmetic, verify it against its original score exactly when reused, and independently verify algebraic nested equivalence within the declared inference tolerance. Do not silently replace an old score with a numerically regrouped version under the same identity. All other additions keep the declared multiply-then-add order. This exception and its provenance must be tested with ties and finite extreme margins.

Locally reuse twelve anchor results from the current parent, checking original identity/parents and own arithmetic. Also inspect only the three canonical asymmetric-pair development caches from the original tree-fusion study; all exist locally. Wholly absent derived origins are rebuilt exactly from own verified sources; partial derived origins fail. Probability logit singles can be derived directly under their existing IDs. Record own parent hashes, domains, weights, canonical leaf coefficients and any original score/record hashes. With all local caches, there are **15 derived reuses and 99 new derived computations**, not new model fits. New source/score outcomes must not be evaluated during planning.

An unconstrained learned fusion would ask a different adaptive question, while another pairwise tree search repeats tested work. The fixed additions test complementary information beyond those pairs and distinguish an unweighted member from merely changing the original balanced member's total weight. The four weights and anchors are analyst-selected and open to group review.

### 3. Complete chronological comparisons and independent audit

Models/preprocessing for month m use months 0..m-1. Keep pooled/latest/rank_fraction policies maximizing actual positive-class F2 `5TP/(5TP+4FN+FP)`. Month-5 policies use month 4; month-6 policies use months 4/5 or latest month 5. Preserve largest-threshold ties, smallest-flag-count rank ties, stable row-ID ordering and exact rational capacity ceilings. The learned rank fraction is not a 1% quota. Select macro month-5/6 F2, then recall, precision and deterministic candidate/policy IDs. AP and pooled F2 are supplementary.

Five fixed reference policies are latest/ latest, recent/latest, tree/pooled, baseline_balanced/pooled and baseline_lgbm/pooled, with canonical IDs from the table above. Their verified macro F2 values are respectively 0.3307932536724773, 0.3290099447918403, 0.3289065701775199, 0.32780157809618937 and 0.3226536849212962. Retain exact reference identities/policies and verify reference values before selection; adding the latest winner prevents claiming progress against an outdated development benchmark.

Three logistic native sources and XGBoost are nonselectable dependencies, so development has **132 physical scores = 18 bases + 114 derived**, of which **120** are selectable. Export **240 forward operating rows**, **120 macro summaries**, **120 pooled-confusion summaries** and **120 monthly score summaries** on identical IDs/labels. Include F1/F2/F3, precision/recall, confusion counts, workload, AP and ROC AUC.

Independently enumerate and reconstruct **1,199 contrast rows**, retaining duplicate numerical values with distinct scientific factors, ties and negative effects:

- 600 against five fixed operating references: 40*3*5.
- 80 non-pooled minus own pooled: 40*2.
- 96 addition minus anchor, 96 minus native added member and 96 minus transformed added member: 32*3 each.
- 54 unweighted-added minus balanced-added: 48 matched additions (4 anchors*4 lambdas*3 policies) plus six native/logit single controls (2 forms*3 policies).
- 27 interaction-anchor minus additive-latest-anchor: 24 matched additions (2 added members*4 lambdas*3 policies) plus three unchanged-anchor controls.
- 144 larger-lambda minus smaller-lambda: 4 anchors*2 additions*6 unordered lambda pairs*3 policies.
- Six transformed minus native LightGBM: 2*3.

The recent-versus-latest anchors differ in C, training weighting and spline fraction; their reference contrasts cannot isolate any one of those factors. Equal C across additive/interaction representations does not imply equal effective complexity.

Export **18 clipping rows** for every probability-like base/month, including unchanged source members, and **48 diversity rows** (4 anchors*2 additions*3 policies*months5/6). Diversity compares transformed added-member and anchor margins plus their own earlier-only decisions at the same policy: common-margin Pearson/Spearman and fraud/legitimate both/one/neither counts. Undefined constant-vector correlations remain null with explicit reasons, never invented zero evidence. Diagnostics do not alter selection.

Retain **nine** logistic solver/numerical/resource/main-feature blocks, **three** interaction blocks, **522** original-feature diagnostics and **18** pair diagnostics. Derive coefficient/intercept keys from actual saved names; the checked local dimensions imply **1,728** rows, but names/dimensions, not a magic count, govern acceptance. Distinguish historical fit measurements from current verification/composition resources.

The independent audit must reconstruct all required source statistics/inference/objectives, all 132 physical scores and origin/parent identities, eligible earlier-only rules, exact masks/metrics/summary order, every diagnostic and the complete contrast factor set before freeze. Use analytical full-grid fixtures without fitting estimators, corruption fixtures across exports/provenance, canonical arithmetic/tie fixtures and all **120 possible winner/policy cases**. Existing hardcoded predecessor table counts are incompatible and must not be reused.

### 4. Predetermined roles, dependencies and bounded transfer

Always freeze seven roles: selected; latest_fusion/latest; recent_fusion/latest; fusion/pooled; balanced/pooled; unweighted/pooled; same_strategy/winner-pooled. Original-reference identity takes precedence over score recipe, so those five winner IDs always use seven roles, even if an anchor represents a blend.

For the only nonreference unchanged-anchor winner (interaction), add matched_anchor=latest anchor at the winning policy: **eight** roles. For a transformed LightGBM single winner, add native_added_member and the other weighting's transformed single at the winning policy: **nine** roles.

For a new addition winner, add native_added_member, transformed_added_member, unchanged anchor_member, matched_added_weight (same anchor/lambda, other LightGBM weighting) and all three other-lambda roles at the winning policy: **fourteen** roles. If its anchor is latest or interaction, also add the otherwise-identical addition with the other of those two anchors: **fifteen** roles. Canonical asymmetric-tree additions use this addition-winner branch, since that historical ID is not one of the five operating reference IDs. Keep aliases visible while deduplicating work.

Mechanical enumeration yields 15 seven-role, three eight-role, six nine-role, 48 fourteen-role and 48 fifteen-role winner/policy cases, totaling 120. Exact DAGs use at most **six base models and seventeen physical final scores**. All reference dependencies always require two additive sources and three trees; an interaction/matched-anchor role adds only the one interaction source. Alternate lambdas introduce no extra base fitting. All six possible local month-7 bases already exist and must be individually verified; the local bound is **zero new final fits**. A fresh clone can require at most two additive, one interaction and three tree fits. Freeze the tighter actual per-type/cache bounds and every existing origin hash before access, rather than rely on the maximum six.

Final rules learn only months 4/5/6 (latest=6). Freeze actual winner, code/config/software/development/audit hashes, role mappings, exact unique source/score dependencies, original cache availability and final rules before further month-7 access. Only frozen DAG nodes may be evaluated. Verify persistence, numerical validity, composition and identical evaluation populations; no losing-grid sweep, cutoff hindsight or winner replacement.

Broader acceptance requires positive measured F2 differences against **all five** fixed references on **both** macro development and consumed transfer. Clear-gain claims additionally need positive conditional interval lower bounds on the named scope. A gain only against balanced, AP improvements or an alias do not satisfy the goal. Consumed month 7 remains descriptive/exploratory and cannot restore independent confirmation.

### 5. Checked uncertainty, figures and reproduction

Use seed 1027 with 500 paired fixed-decision row draws for each month 5/6/7: **1,000 development + 500 consumed rows**. For role counts 7/8/9/14/15, produce **18/21/24/39/42 development intervals** (month5/month6/macro) and **6/7/8/13/14 consumed intervals**, retaining aliases. Independently reconstruct every sampled-row draw and quantile. Intervals condition on fixed models/policies and exclude adaptive source/anchor/lambda choice, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity correction.

Findings cover five reference comparisons and every matched effect, actual fit/reuse/derived/test counts and retained failures. Inspect a figure showing all 120 strategy/policy operating points and pooled F2 versus lambda for balanced/unweighted additions to all four anchors, with a clearly matched latest/interaction comparison. Verify plotted values/legends. Any lambda=0/1 endpoint displayed must be its existing anchor/transformed-member control, not a new evaluated strategy. Record canonical arithmetic and score semantics.

README must give pinned environment/data access and checksum, exactly fourteen original selection restores rejecting changed records, individual absent-cache reproduction and ordered tests/receipt -> predictor-only preflight -> prepare -> validate/independent audit -> freeze -> bounded exploratory transfer -> report commands. Existing frozen audit/selection/report evidence must not be overwritten. Update current context and assistance/API attribution, keeping proposed settings distinct from approved group choices. Recheck preservation, relevant tests/compileall/strict OpenSpec, exact copied/exported identities and every final report hash after logs close. Synchronize/archive only after requirements and task outputs verify; finite completion does not change failed goal acceptance.

## Risks / Trade-offs

- Unweighted additions may dilute useful rankings -> matched balanced/member/anchor/lambda controls and retain every negative result.
- Prior consumed results motivate the anchors -> disclose adaptive choices and conditional limitations; prohibit independence claims and hindsight selection.
- Nested arithmetic can rename or perturb an old score -> one canonical historical ID and exact old direct arithmetic, independently checked against the nested identity.
- Reuse spans current and older source formats -> individual verified origins, four actual tiny reproduction formats and no old-source edits or unrelated-grid loaders.
- Fresh exact source fitting can fail resource/numerical gates -> actual-host preflight, bounded chunks and retained failure evidence; no post-outcome relaxation.

## Migration Plan

After planning, implement only the isolated declared files and current documentation. Snapshot preservation and verify methodological/source-format fixtures before execution. Prepare required bases, compose all finite scores, evaluate complete chronological controls and independently audit before freeze. Transfer only frozen roles, finish checked uncertainty/figures/findings/reproduction, then audit, synchronize and archive actual completion. Preserve failures and all completed histories; further tuning requires another study. No material design question is deferred, and all new settings remain analyst-selected.

## Implementation revision record

The first preparation attempt verified/reused all six month-4 bases, then stopped before persisting derived scores, operating metrics, selection or month-7 access. Its adapter incorrectly interpreted historical tree-fusion symbolic `record`/`prediction` dependency keys as paths. All 49 original sources, configuration/protocol/tests, gates/logs and partial artifacts are preserved under outputs/baf-unweighted-reference-fusion-v1/implementation_attempts/symbolic-provenance-v1/, with 87 checked hashes in implementation_revisions.json. The source-format adapter now reconstructs exact origin paths and requires the historical schema/hashes, without changing recipes, scores, objective or numerical/resource gates. Fifteen real historical record-schema regression cases supplement the synthetic fixtures. A new full-source test receipt and resource gate are required before fresh preparation; the prior 907-test receipt is historical evidence, not the revised execution gate.

The copied source snapshot is stored in `.source_snapshot/` so pytest does not collect historical duplicate test modules. The original 86 artifact/source files remain byte-identical; the 87th checked file retains the original preservation index and the current index records the path relocation. The failed collection log is retained separately; collection performed zero estimator fits.
