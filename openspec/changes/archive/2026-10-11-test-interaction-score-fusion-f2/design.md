## Context

See proposal.md for motivation and specs/baf-interaction-score-fusion-study/spec.md for the contract. Thirteen completed studies and a stopped L-BFGS attempt are frozen. The interaction study's best new macro F2 is 0.3229761865325379, below the selected recent fusion/latest at 0.3290099447918403. Seven of twelve matched interaction effects are positive. The selected existing policy's consumed F2 is 0.3702475685234306 versus tree fusion 0.3726941226941227 and unweighted 0.3768844221105528. These results motivate a complementarity hypothesis, not an assumed improvement.

The current parent is configs/interaction-spline-logistic.json, config SHA 4178dbb6b2bae9b9c121dfc81ba6157ae4b8e5948ce82f7ee0feaaccf9624cd7. Its original/exported selection SHA is ec8623c26a777dbd9966b5b4f3b837ac7257b779a418336e2b923efa8b2aadac and final report-manifest SHA is b8b8abf1ba6a79a2af3c7b6067e55f0041b8bd06b4b059ecaf48657efaeb55f1. All 78 report file hashes and required current development caches were checked during planning. No new composition outcomes have been evaluated.

## Goals / Non-Goals

**Goals:** Test fixed interaction/tree score complementarity under matched additive, C, training-weight, alpha and policy controls; independently verify actual chronological F2 and preserve all outcomes.

**Non-Goals:** Tune source models/features/pairs, learn fusion weights or calibration, introduce a quota/floor, change beta, select on AP/diversity, duplicate an existing strategy, or sweep losing settings on consumed month 7. This remains an adaptive exploratory study; no independent evaluation month remains.

## Decisions

### 1. Separate finite identity and individual cache preparation

Use Base v2 SHA 7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809, Python 3.12.9 and unchanged requirements.txt, requirements-improvements.txt and requirements-xgboost.txt. Seeds remain model 2026 and bootstrap 1027, with six numerical threads. Introduce only the new config/protocol/three modules/tests and output/report roots named in the proposal. The transitive source closure is the parent's 43 paths plus the three new modules, exactly 46. Freeze source/config/software/protocol identity before execution.

The mandatory preservation set is the parent's 366 identities plus its three new sources, config, protocol, original selection, all 78 final report files and final report manifest: **451 distinct paths**, including thirteen original/exported selection pairs. Snapshot every available historical ignored output file before implementation/execution and reject mutation/deletion. Preserve both original Newton fit and corrected analysis histories and the stopped failure evidence. Optional ignored bulk caches can be absent on a fresh clone; their absence does not authorize recreating unrelated history. Do not call a predecessor full-grid initializer/loader that requires unrelated ignored artifacts.

For each future month 4/5/6 require eight logistic members: representation spline/interaction, C=0.1/1.0, training class_weight None/balanced. Also require XGBoost d4/w1 and balanced/unweighted LightGBM. This is **33 individual base results**. Locally all exist and are reused: zero new development fits. Origin precedence is current interaction-study caches; for absent additive caches, original Newton then completed spline-fusion caches; for absent tree caches, the original tree-fusion caches. Interaction caches have only the current parent origin. Inspect complete required sets individually, including XGBoost's native model and adapter. A partial/incompatible inspected required set fails; only a wholly absent set may advance to another declared origin or reproduce in the new destination. Source model/recipe/population/software/preprocessing/inference proof is required; a selection alone does not validate a model.

Exact missing reproductions retain original preprocessing and recipes. Logistic sources retain the unchanged 29 predictors, 17 smooth/seven discrete/five categorical roles, train-only medians/knots/scales/vocabularies, missing indicators and constant tails. Interaction sources retain the same three centered/scaled pairs and inactive-column rules. Keep Newton-Cholesky L2/l1_ratio=0, tol=1e-6, max_iter=100, intercept, no warm start, zero observed fallback/warnings, finite independent objective/full gradient <=1.1e-6 and iterations 1..99. Verify coefficient/sigmoid reconstruction after serialization. Use compatible frozen matrix/inference interfaces only; do not modify or monkey-patch old modules or apply an additive-only/full-grid auditor to incompatible artifacts. Tree source recipes/adapters remain exactly frozen.

Source feature audits and any required fitting use bounded chunks of at most 8,192. Before a required new fit, perform an actual-host predictor-only resource estimate using eligible months 0-6 (903,157 rows locally), which covers the largest possible final training population. Retain actual dimensions/storage/Hessian/time/RSS and the prior 70%-of-host-RAM bound. Do not change representation or numerical gates after allocation/convergence failure. Count new/reused development and final fits separately. No new production miniature estimator fits are declared: the source representation/solver is unchanged and its eight-fit gate is preserved. New absent-cache tests must make one actual tiny fit for each of four formats—interaction, additive, LightGBM reference and XGBoost/native-adapter—and record four new test-only fits per complete invocation. Feature/composition/analytical fixtures fit no estimators; historical test-module fits remain separately disclosed.

### 2. Fixed common-margin composition and canonical identity

Let epsilon=1e-6 and L(p)=log(clip(p,epsilon,1-epsilon))-log(1-clip(p,epsilon,1-epsilon)). Validate probability-like inputs are finite and in [0,1] before clipping. The existing tree margin is g=0.5*L(p_xgb_d4_w1)+0.5*L(p_baseline_balanced). It is already a finite margin, not a probability.

For every one of the eight logistic members p, retain its native score, its transformed single L(p), and fixed blends alpha*L(p)+(1-alpha)*g for alpha {0.10,0.25,0.50,0.75}. Alpha is the logistic-member weight. Do not logit g, learn weights, center/standardize by an inference batch or use current labels. Weighted model scores and composed margins are not asserted calibrated fraud probabilities. Reject non-finite results, invalid weights/domains/dimensions and misaligned IDs/labels. Changing aligned labels or unrelated rows cannot change a row's score; a rank-based operating decision can depend on the batch through its separately declared policy.

Raw member IDs remain unchanged. Transformed singles use logit_<member>. Blend IDs use interaction_blend_<member>_a10/a25/a50/a75, **except** spline_c0p1_unweighted at alpha=0.25 uses the existing ID margin_blend_spline_c0p1_unweighted_a25. That exception has one recipe, score and selectable identity; it remains a blend in factor enumeration and an original reference for transfer precedence. Add only three extra strategies: tree fusion, balanced LightGBM and unweighted LightGBM. The union is **51 strategies**: eight native + eight transformed + 32 blends + three extras, not 52.

Reuse the three canonical recent-fusion development scores from the current parent when complete, verifying original identity/parents and own checked arithmetic. If wholly absent, derive exactly from own audited members; partial derived caches fail. Derive the tree margin from own checked XGBoost/balanced parents. Native/logit/blend provenance records exact domains, epsilon, weights, parent IDs and own hashes, plus original parent/record hashes for any derived reuse. A blend's direct parents are its native logistic member and tree margin; transforming that member internally need not create an additional final-only logit artifact unless a frozen role requires it.

The alternatives are another standalone interaction grid (already completed), unrestricted learned stacking (a different adaptive question) or a large unconstrained weight search. Four fixed weights extend the established composition controls while keeping arithmetic and matched effects explainable. They are analyst-selected research settings, not bank confidence or cost estimates.

### 3. Complete chronological development, diagnostics and contrasts

Models/preprocessing predict m from months 0..m-1 for m=4/5/6. Reuse unchanged pooled/latest/rank_fraction policies maximizing actual positive-class F2=5TP/(5TP+4FN+FP), beta=2. Month-5 rules use month 4; month-6 rules use months 4/5 or latest month 5. Preserve largest-threshold ties, smallest flag-count rank-optimum ties, stable row-ID ordering and exact rational ceilings. Rank fraction is learned from F2, not a 1% quota. Select actual macro month-5/6 F2, then recall, precision and deterministic candidate/policy IDs. Pooled F2 and AP supplement that objective.

Original references remain recent fusion/latest, tree fusion/pooled, balanced/pooled and unweighted/pooled. The three nonselectable XGBoost dependencies bring physical development score artifacts to **156**: 33 bases plus 123 derived results (24 transformed singles, 96 blend scores including three canonical recent-reference reuses, and three tree margins). **153** scores are selectable. Export **306 forward operating rows**, **153 macro summaries**, **153 pooled-confusion summaries** and **153 monthly score summaries**; these equal counts describe different key sets.

Enumerate **1,386 contrast rows**, preserving labels/factors even when numeric differences duplicate:

- 612 against all four original operating references: 51 strategies*three policies*four references.
- 102 non-pooled minus own pooled: 51*two.
- 96 blend minus native member; 96 blend minus transformed member; 96 blend minus tree member at the same policy: 32*three each.
- 72 interaction minus additive; 72 balanced minus unweighted; 72 C1 minus C0.1: four matched source pairs*six forms*three policies each, where forms are native, logit and the four alphas.
- 144 larger-alpha minus smaller-alpha: eight members*six unordered alpha pairs*three policies.
- 24 transformed minus native: eight*three.

Keep F1/F2/F3, precision/recall, confusion counts, workload, AP and ROC AUC, all negative effects and ties. Export **33 clipping rows**, one for each probability-like base/month, including the unweighted baseline even though it is not transformed. Export **48 diversity rows**, eight members*three policies*months5/6. Diversity compares transformed-member and tree-margin scores and their own earlier-only decisions at the same policy: common-margin Pearson, Spearman, overlaps and fraud/legitimate one-member-only/both counts. Undefined constant-vector correlations are explicitly null with a reason, not invented zero evidence. Diagnostics never alter the grid or selection.

Retain all 24 logistic solver/numerical/main-feature blocks, twelve interaction metadata blocks, 1,392 original-feature diagnostics, 72 pair diagnostics, all coefficient/intercept keys and resource values. Locally there are 5,016 coefficient rows, with dimensions derived from saved names rather than a hardcoded acceptance count. Distinguish historical fit resource measurements from current verification/composition timing and RSS.

Independent verification must enumerate every expected strategy/output/contrast key and independently reconstruct source marginal/pair statistics and saved inference/objectives, every composition and parent identity, earlier-only policy/mask/metric, diagnostics, contrast arithmetic and actual macro winner before freeze. Do not use the predecessor's hardcoded 12-strategy table auditor for this grid. Test independent reconstruction corruption cases, exact canonical aliasing and all 153 possible winner/policy role cases.

### 4. Frozen roles and exact dependency-specific transfer

Always freeze six roles: selected, recent_fusion/latest, fusion/pooled, balanced/pooled, unweighted/pooled and same_strategy (winner/pooled). **Original-reference identity takes precedence over recipe kind**: any of the four reference IDs keeps exactly six roles, including the canonical recent blend.

For a new native/transformed member winner, add original_member/native and transformed_member/logit plus matched_representation, matched_weight and matched_C at the winning policy, giving **eleven** roles. Matched controls keep the winning source form and otherwise change exactly the named factor. For a new blend winner add those five roles and all three other-alpha roles, giving **fourteen** roles. Alpha roles use the same sources; no extra base fits are needed. If a matched/alpha control maps to the canonical reference ID, retain that ID at the winning policy while the original reference role retains latest. Keep aliases visible and deduplicate actual work.

Before further month-7 access, freeze final rules learned on months 4/5/6 (latest=6), full code/config/software/development/audit identities and exact unique source/score dependencies. Check frozen type/cache-specific fit bounds before access. Every final role must predict identical IDs and labels, with source model/feature/numerical validity audited after persistence. No losing-grid sweep, hindsight cutoff or replacement of the development winner.

An interaction-source winner requires at most three interaction members, its matched additive member and the fixed recent-reference additive C0.1/unweighted member; an additive-source winner requires one matched interaction and up to four additive members including that reference member. All cases also require the three tree bases. Reference winners need only the fixed additive reference member and three trees. Thus a fresh clone needs at most eight unique base fits: 3 interaction+2 additive+3 tree, or 1 interaction+4 additive+3 tree. Locally both C0.1 additive finals and three tree finals are available through declared origins, while C1 additives and interaction finals are absent. Interaction winners need at most three new interaction fits plus one C1 additive reproduction; additive winners at most one interaction fit plus two C1 additive reproductions. Reference winners need zero new fits. Freeze the tighter actual set and original-cache availability; do not rely only on conservative family maxima.

Broader goal acceptance remains strictly positive F2 differences against all four original references on both macro development and consumed transfer. Clear-gain language additionally requires positive conditional lower bounds on the named scope. A renamed alias, AP gain or isolated matched contrast cannot satisfy it. All settings are analyst-selected and adaptive after thirteen studies; consumed month 7 cannot become fresh confirmation.

### 5. Verified reporting and clean reproduction

Use bootstrap seed 1027 with 500 paired fixed-decision sampled-row draws each for months 5/6/7: 1,000 development plus 500 consumed draw rows. Independently reconstruct every draw and quantile and retain aliases. Six/eleven/fourteen roles require **15/30/39 development intervals** (month5, month6, macro) and **5/10/13 consumed intervals**. Condition on fixed models/policies; exclude adaptive source/C/alpha/pair choice, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity correction.

Findings expose every original-reference and matched effect, fit/reuse/test counts and failures. Inspect an operating figure for all 153 strategy/policy results and pooled-alpha panels for matched additive/interaction sources, with readable labels/legends and correct actual values. Any alpha=0/1 endpoints shown are the declared tree/transformed controls, not new evaluated models or tuned settings. Do not label score correlation causal or a substitute for F2.

README must document the pinned environment/data checksum, exactly thirteen selection restores that reject changed records, individual absent-cache reproduction and ordered tests/receipt -> predictor-only preflight -> prepare -> validate/independent audit -> freeze -> exploratory transfer -> report. Existing frozen runs must not be overwritten by rerun audit-writing/freeze/report stages. Record assistance and external APIs/code; no public notebook is assumed. Recheck all preserved identities, relevant tests/compileall, copied artifacts and every final report hash after logs close. Synchronize/archive only after all requirements and actual outputs verify; leave the broader goal active if any required comparison fails.

## Risks / Trade-offs

- A weaker source may add no useful complementarity -> fixed matched controls and retain all negative results.
- Common margins can differ in scale and training-weight offsets -> retain native/transformed/member/alpha controls; do not assert calibration or alpha as confidence.
- An existing blend appears twice algebraically -> one canonical ID plus reference precedence, tested across every winner/policy case.
- Cache provenance and absent-cache paths span source formats -> individual compatibility checks, four actual tiny reproduction formats and no old-module edits/full-grid loaders.
- Reused development and repeated consumed access increase optimism -> prospective finite declaration, conditional limitations, frozen bounded roles and no independent-confirmation claim.
- Fresh exact fitting can fail numerical/resources gates -> bounded construction, actual-host preflight, retained failure evidence and no post-outcome recipe/gate relaxation.

## Migration Plan

After planning, implement only the isolated config/protocol/three modules/tests and current documentation. Verify preservation, canonical arithmetic, cache-format and full analytical export fixtures before execution. Prepare only required compatible bases, evaluate the complete finite grid and independent audit, freeze exact roles, transfer, then finish checked uncertainty/figures/findings/reproduction. A gate failure leaves the study incomplete and preserved. Archive only after actual completion; no historical study is rewritten.
