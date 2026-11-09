## Context

See proposal.md and specs/baf-spline-interaction-study/spec.md. Twelve completed studies and one stopped L-BFGS attempt are preserved. The latest selected spline fusion scores macro F2 0.3290099447918403 at its latest policy but consumed F2 0.3702475685234306; it did not beat the older tree fusion or unweighted reference on transfer. Its config/selection hashes are `0b8277763cfacc0f03ffb85a721f8b2e77f2dd671ee3f39bb9bdc0535fd70b34` / `3d6b693e78be6dee30b0911a0776cce6580f4edbaa1cb9c389d0a02a9e0c8dae`. Original Newton fit and corrected analysis identities remain separate. The existing engineering has scalar credit/income and age/credit products, while AdditiveInputs has only separate feature bases. Pair choices here are adaptive hypotheses, not discoveries or approved group choices.

## Goals / Non-Goals

**Goals:** Isolate three nonlinear pair surfaces under otherwise unchanged additive features, populations, optimizer, C and weighting; expose whether added feature capacity improves actual F2 and whether the selected result transfers.

**Non-Goals:** Choose pairs using outcomes, search knots/degrees/solvers, add new interaction-score blends, change the loss/metric, replace actual macro F2 with diagnostics, introduce quotas/floors or claim independent confirmation. Scalar-product engineering and tree results are historical context, not additional selectable grids.

## Decisions

### 1. Separate identity, data and exact cache preparation

Use Base v2 SHA-256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`, Python 3.12.9 and the unchanged three requirements files. Introduce only the config/protocol/three modules/destinations listed in proposal.md. The complete source identity is the latest study's 40-source transitive closure plus those three new modules: 43 sources. Preserve mandatory source/config/report identities, all twelve restored original/exported selection pairs, Newton fit/analysis histories and the stopped failure evidence. Snapshot every available historical output file before execution and reject deletion/change. Optional ignored bulk history can be absent on fresh clones; do not recreate it to satisfy history.

For each development month 4/5/6 reuse four original additive models: `spline_c0p1_unweighted`, `spline_c0p1_balanced`, `spline_c1p0_unweighted`, `spline_c1p0_balanced`. Reuse XGBoost d4/w1 and balanced/unweighted LightGBM: twelve additive plus nine tree pairs, **21 base reuses** locally. Verify exact original recipes, train populations, source/software identities, saved positive-class inference and additive numerical/representation evidence. Required individual cache origins may be the original Newton run or the completed spline-fusion run, but ancestry must match and selection records alone do not validate models. Reject partial/incompatible caches; never use a full-grid loader that demands unrelated models. Wholly absent required individual recipes reproduce only in the new destination, counted explicitly (at most twelve additive and nine tree development reproduction fits).

Retain the latest selected fusion score as a fourth reference. Reuse its three checked development score records when available and verify all original parents and original 40-source identity. Otherwise derive exactly `0.25*logit_clip(p_spline_c0p1_unweighted)+0.75*g_tree_fusion`, epsilon 1e-6, from required already-checked members. The tree fusion is already a finite margin; never logit it. Own reference provenance names both new required-member hashes and any original record/parent hashes used for reuse. Partial derived caches fail. This reference is fixed row-wise, not a new blend search; its probability-to-logit intermediate need not be an additional own score artifact. Model predictors/preprocessing remain unchanged for every reused reference.

### 2. Three fixed training-only tensor blocks

The four new candidates are `interaction_c0p1_unweighted`, `interaction_c0p1_balanced`, `interaction_c1p0_unweighted`, `interaction_c1p0_balanced`. They use the same additive main-effect matrix as the matched original spline controls, with exactly these ordered pairs:

1. `credit_risk_score`, `income`: ask whether the risk-score association varies nonlinearly across income.
2. `customer_age`, `proposed_credit_limit`: ask whether a nonlinear age/credit surface adds to existing separate effects and previously tried scalar products.
3. `name_email_similarity`, `current_address_months_count`: ask whether identity-name consistency relates differently to address tenure.

All six variables are existing smooth predictors. No target/month, extra identifiers, rolling features or causal interpretation is introduced. The unchanged 29 original predictors retain 17 smooth/7 discrete/5 categorical roles, observed-training cubic quantile knots with duplicate removal and constant tails, train medians, 24 numeric missing indicators, variance scaling without centering and one-hot vocabularies. Fit an unchanged AdditiveInputs('spline') for the main matrix. Pair blocks use its **unscaled marginal basis columns**, before the original numeric scaler. They introduce no new knots, imputers or categories.

For pair (a,b), let U and V be its training-fitted marginal bases. Fit training column means mu_U and mu_V, then form row-wise products T_ij=(U_i-mu_U_i)*(V_j-mu_V_j), ordered i-major then j. Fit each tensor column's training population mean mu_T and standard deviation sigma_T. Append Z_ij=(T_ij-mu_T_ij)/sigma_T_ij to the unchanged main matrix. Preserve all columns and their names. Exactly constant tensor columns use scale=1 and are forced zero during both training and inference; a training-degenerate marginal block makes its entire pair block inactive. Save inactive flags and dimensions rather than dropping columns. Future missing values use saved medians; tails use saved marginal rules. All pair statistics are predictor-only and training-only, using unweighted population moments irrespective of the model's class-weight setting. Centering reduces overlap with main effects but does not claim orthogonality or identical effective regularization.

Use stable population-moment computation and bounded row chunks of at most 8,192 during tensor construction; never allocate an n*d_a*d_b temporary for all pairs together. There are at most six marginal columns per smooth predictor and at most **108 added tensor columns** across the three pairs; actual duplicate/degenerate knot dimensions remain recorded. Keep CSR output and record dimensions/nnz/actual storage, Hessian estimate, fit time and peak process RSS. Before full fits, estimate matrix/workspace needs on the actual host and prove chunked construction on miniature data; an allocation failure retains evidence and stops completion without silently reducing features. Independent moment/scale reconstruction uses atol=1e-12, rtol=1e-10; saved inference uses atol=rtol=1e-12. Inactive decisions use actual constant values, not outcome-chosen near-zero cutoffs.

The alternatives are repeating additive C/weight searches (already done) or introducing many unrestricted pairs (a different capacity/resource question). This finite three-pair design tests richer local effects while retaining interpretable matched controls. It does not presume these pairs improve performance.

### 3. Numerical gates and finite execution

Reuse the preserved matrix-level Newton recipe explicitly: L2/l1_ratio=0, Newton–Cholesky, tol=1e-6, max_iter=100, intercept, warm_start=False, seed 2026, six threads. Every new fit requires finite objective/full gradient, independent absolute gradient <=1.1e-6, iterations 1..99, no warnings and no actual entry to alternative-solver fallback. Reconstruct expit(X*coefficient+intercept), class orientation and serialized inference. The normalized weighted objective and intercept/penalty convention remain exactly those of the original Newton study. Do not call its additive-only audit on an interaction artifact or edit/patch old modules; the new driver verifies the new representation and delegates only compatible numerical interfaces.

The eight-fit miniature gate crosses all four new recipes with two deterministic fixtures: an ordinary chronological synthetic fixture and one with degenerate/duplicate knots, numeric missingness, unseen categories and future tail values. Feature-only invariance/outer-product fixtures fit no models. Count any separate tiny cache-reproduction test fits independently. Full development makes **12 new interaction fits**, four per future month. Numerical/allocation failure stops selection and transfer without changing pairs, C or gates.

Twelve selectable strategies comprise eight model scores (four additive, four interaction) and four references (latest spline fusion, older tree fusion, balanced, unweighted). Export **39 score artifacts**: 33 base results (24 logistic plus nine tree) and six derived references (three tree fusions and three latest spline-fusion scores), including three nonselectable XGBoost dependencies. Thus 36 are selectable. Record the three latest-fusion derived reuses separately from the 21 base reuses; reproduced derived arithmetic is not a model fit.

### 4. Chronological policies and complete contrasts

Every model uses months 0..m-1 to predict m in {4,5,6}. Month-5 policies learn on month 4; month-6 policies learn on months 4/5 or latest month 5. Apply the unchanged pooled/latest/rank-fraction policies optimizing F2=5TP/(5TP+4FN+FP), beta=2, with stable ID ties and exact rational ceilings. Rank fraction is learned from F2, not a 1% quota. No imposed precision floor. Select actual macro month-5/6 F2, then recall, precision and deterministic IDs.

Export **72 forward rows**, **36 macro**, **36 pooled-confusion**, **36 score summaries** and **204 contrasts**:

- 144 against the four original operating references: 12 strategies*3 policies*4. The latest spline fusion reference retains **latest**; tree fusion, balanced and unweighted retain **pooled**.
- 24 non-pooled versus own pooled: 12*2.
- 12 interaction-minus-additive: 2 C*2 weighting*3 policies.
- 12 balanced-minus-unweighted: 2 representations*2 C*3 policies.
- 12 C=1-minus-C=0.1: 2 representations*2 weighting*3 policies.

Retain all factors, ties and negative results, F1/F2/F3, precision, recall, confusion counts, workload, AP and ROC AUC. No diagnostic or pooled metric replaces macro F2. Export 24 logistic solver/numerical rows and main-feature metadata blocks (new/reused tags), 1,392 original-feature diagnostic rows (24 model/folds*29 variables*2 populations), twelve interaction metadata blocks, 72 pair diagnostic rows (12 fits*3 pairs*2 populations), all coefficients/intercepts and resource measurements. Audit exact expected sets and actual dimensions rather than inventing a fixed coefficient-row count.

### 5. Frozen bounded transfer and acceptance

Always freeze six roles: selected, recent_fusion/latest, fusion/pooled, balanced/pooled, unweighted/pooled, same_model (winner/pooled). If a model wins, add matched_representation, matched_weight and matched_C at the winning policy, producing nine roles. A reference winner keeps six. Keep aliases and deduplicate work. Freeze every required model/score dependency, full development outputs and code/config/software plus final rules learned on months 4/5/6 (latest=6) before recorded further consumed-month access.

Only frozen roles transfer. An interaction winner needs at most three distinct new interaction models, its matched additive control and the fixed recent-fusion additive member. An additive winner needs one matched interaction model and up to three additive controls, plus the fixed recent-fusion member. A reference winner needs no interaction fit. Current cached C=0.1 additive models and the three tree bases should be reused only after checking exact provenance; C=1 additive models may need exact-recipe reproduction. Locally the overall conservative bound is at most three interaction plus two additive fits and three tree base reuses; actual winner dependencies give the tighter bound and must be disclosed. Without caches, reproduce only the exact required unique members, at most four additive and three tree bases, counted separately. Never transfer losing grids, new pair sets or hindsight thresholds.

The broader goal requires positive measured F2 differences against **all four** original reference policies on both development and consumed transfer. This includes the latest development-selected fusion despite its failed previous transfer. Clear-gain language additionally requires positive conditional interval bounds on the named scope. Positive AP, an isolated matched contrast or merely finishing the grid cannot satisfy that acceptance. Month 7 remains consumed and no result restores independent confirmation.

### 6. Checked evidence and reporting

Independently reconstruct every required saved model inference, marginal/pair feature statistic and coefficient contribution, every derived reference/parent identity, earlier-only policy, metric, contrast factor/arithmetic, diagnostic and actual-F2 winner before freeze. Verify all frozen role types in methodological tests, including partial/missing cache handling and no old-module changes.

Use seed 1027 and 500 paired fixed-decision row draws per month 5/6/7: 1,000 development and 500 consumed draw rows. For N=6/9 roles, export 3*(N-1)=15/24 development interval summaries and N-1=5/8 consumed summaries. Reconstruct every draw and quantile independently through sampled Boolean rows; preserve aliases. State conditioning on fixed models/policies and exclusions of adaptive pair/C/source choice, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity correction.

Inspect an operating/matched-effect figure and three pair-contribution heatmaps. For the selected interaction model, or best development interaction if a reference/additive wins, use its month-6 coefficients and train reference values. Export **1,323 pair-surface rows** (3 pairs*21*21): each panel isolates the corresponding centered/scaled pair-block coefficient contribution relative to the reference pair values, over train 1–99% marginal ranges. Independently reconstruct block contributions; do not label them total model effects or causal relationships. A diagnostic losing model is not transferred because it is plotted.

Findings expose all reference and matched comparisons, failures, actual full/miniature/test-only fits and preservation checks. README includes pinned dependencies/data checksum, twelve selection restores, exact required absent-cache reproduction and ordered miniature/development/audit/freeze/transfer/report stages. Update AGENTS/context/assistance attribution to measured state, run relevant tests/compileall/strict OpenSpec validation, verify copied report hashes and audit all requirements before sync/archive. Leave the goal active if acceptance fails.

## Risks / Trade-offs

- Tensor columns increase memory and regularization geometry → retain only three fixed pairs, bounded construction, resource evidence and matched additive controls; equal C does not imply equal complexity.
- Degenerate knots or columns can make moments unstable → retain frozen inactive flags/dimensions, stable population moments, independent feature/numerical checks and failure evidence.
- Pair/source/C choices and reused development months increase optimism → label them adaptive and report complete contrasts and conditional limitations.
- Old cache provenance spans original and copied fits → audit only declared compatible individual artifacts with both original fit/analysis identities; reject partial/incompatible caches.
- The latest-policy winner may fail transfer again → freeze before access, retain its failed comparison and do not switch the selected role afterward.

## Migration Plan

Introduce only new source/config/protocol/destinations. Verify ancestry, train-only tensor fixtures, resource estimates and miniature fits first; prepare only required caches; execute the complete finite development and independent audit; freeze bounded roles; transfer; finish uncertainty, checked figures and reporting. Numerical/allocation failure preserves incomplete work. Synchronize/archive only after actual requirements and outputs verify; no historical study is rewritten.
