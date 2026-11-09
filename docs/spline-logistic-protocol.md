# BAF additive spline-logistic protocol v1

Declared before fitting on 11 October 2026. Analyst-selected exploratory settings, not group-approved decisions. Configuration: `configs/spline-logistic.json`.

## Context

The active OpenSpec change `test-spline-logistic-f2` records motivation and the requirements contract. Ten frozen studies exist. Their selected fusion is `logit_mean_xgb_d4_w1_baseline_balanced_a50 / pooled`, macro month-5/6 F2 0.3289065701775199 and consumed-month F2 0.3726941226941227. Original balanced/unweighted pooled macro F2 is 0.32780157809618937/0.3226536849212962; consumed F2 is 0.3644085570469799/0.3768844221105528. These are adaptively reused observations. No independent test month remains. Settings below are analyst-selected and open to group review.

This protocol introduces a fitted representation with solver gates, feature diagnostics and cross-study model provenance. No frozen source, configuration, dependency or selection is edited.

## Goals / Non-Goals

**Goals:** Isolate smooth additive representation from linear modelling with paired recipes; provide stable saved inference for sparse, missing and degenerate inputs; compare actual chronological F2 against all three existing references; retain enough feature/solver evidence to explain the models.

**Non-Goals:** Estimate causal effects, infer a performance ceiling, change beta, impose review quotas, restore independent month-7 validation, introduce target encoding or interactions, tune knots after outcomes, or optimize AP instead of F2. No new score ensemble is part of this study.

## Decisions

### 1. Identity and isolated execution

Use unchanged Base v2, SHA-256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. New files: `configs/spline-logistic.json`, `docs/spline-logistic-protocol.md`, `baf/spline_inputs.py`, `baf/spline_study.py`, `baf/spline_report.py` and `tests/test_spline_logistic.py`. Destination roots are `outputs/baf-spline-logistic-v1` and `reports/baf-spline-logistic-v1`. Keep existing Python and `requirements-xgboost.txt` pins unchanged. Reuse frozen loading, metric and composition interfaces without editing them.

Declare config, full transitive source/software/Python fingerprint, ten original/exported selection hashes, all old config hashes and actual data identity before fitting. Verify preservation before and after each stage. Persist fitted preprocessing and logistic model together, stable ordered training/evaluation IDs, prediction hashes, warnings/time, actual iterations, feature metadata and fitted-recipe parameters. Caches require complete checked records; genuinely absent caches can be reproduced only under their original recipes in this destination. A partial or incompatible cache is an error, not permission to refit over it.

### 2. Fixed predictor roles and training-only preprocessing

Use the 29 original predictors returned by the frozen `features` decoder: exclude fraud_bool, month and device_fraud_count; retain the five documented -1 sentinel decodings and negative intended_balcon_amount as missing. All models use expanding earlier months and identical eligible rows, without sampling or recency weights.

Categorical columns are payment_type, employment_status, housing_status, source and device_os. Fit one-hot vocabularies on training only, use an explicit missing category, ignore unseen categories and retain that behavior after serialization.

Seven discrete numeric columns stay linear in both representations: email_is_free, phone_home_valid, phone_mobile_valid, has_other_cards, foreign_request, keep_alive_session and device_distinct_emails_8w. The other 17 numeric columns are income, name_email_similarity, prev_address_months_count, current_address_months_count, customer_age, days_since_request, intended_balcon_amount, zip_count_4w, velocity_6h, velocity_24h, velocity_4w, bank_branch_count_8w, date_of_birth_distinct_emails_4w, credit_risk_score, bank_months_count, proposed_credit_limit and session_length_in_minutes.

Fit unweighted numeric medians on eligible training predictors; fill an entirely missing column with zero. Append separate missing indicators for all 24 numeric columns, even if a column has no training missingness, so new missingness has a fixed inference schema. The imputed original numeric columns form the linear control. No signed logs, ratios, crosses, existing engineered fields or feature selection are added.

For each of the 17 designated smooth columns, compute proposed knots from its observed (not median-imputed) eligible training values at quantiles [0, .25, .5, .75, 1], using the linear quantile method; remove exact duplicates and retain sorted unique knots. Using observed values prevents high missingness from collapsing most knots to the median. With at least two distinct knots, fit a separate cubic univariate basis using these explicit knots, omit its implicit bias basis element and use constant extrapolation beyond the fitted boundary. With fewer than two distinct observed values, use the same imputed linear column in the smooth pipeline and record `constant` or `all_missing` fallback. Every column contributes a declared block with explicit names and ranges.

Scale each numeric/basis/missing-indicator output using training-only variance and no centering; leave categorical one-hot outputs unscaled. Apply the same scaling convention in both representations. Assemble finite float64 CSR features and audit actual dimension, density, transformed names and scale. Preprocessing is identical across C and class-weight choices within a representation and fold; no labels or class weights enter knot/scaler fitting. Permit checked in-destination reuse of the exact fitted transform across those recipes, without changing its identity. Inference checks ordered input schema, finite non-missing numeric values and output dimensions. Report per-column knot/fallback/missingness and future-boundary exceedance diagnostics; exceeding a training range does not cause refitting.

This uses the [official spline basis API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html). Sparse additive bases replace a much larger interaction expansion; duplicate-knot handling and fixed fallback prevent discrete or missing-heavy fields from causing invalid fits. Constant extrapolation limits tail instability during drift; linear extrapolation is excluded from this finite study.

### 3. Fixed paired logistic recipes and numerical gates

Six spline recipes and six linear recipes cross C {0.01, 0.1, 1} with training class_weight {None, balanced}. Use L2 regularization (`l1_ratio=0` under installed scikit-learn 1.9.1), intercept, solver lbfgs, tolerance 1e-6, max_iter 2000, warm_start=False and seed 2026. Control native math thread pools to six. Balanced weights use only that fold's class counts; retain every training row. C is an experiment setting, not an approved operating policy or a claim of equivalent effective complexity across representations.

The [official logistic API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) describes the solver and regularization controls. L-BFGS avoids forming a full Hessian; feature scaling and a declared convergence gate address conditioning. Do not pass deprecated n_jobs or penalty arguments. Before full-data fits, perform two deterministic miniature fits per recipe, 24 fits counted separately, including missingness, low-cardinality/duplicate knots, constants, new categories and out-of-range inference. Check finite probabilities, class order [0,1], fitted dimensions, deterministic repeats, saved/reloaded scores at absolute/relative tolerance 1e-12, and reconstructed decision_function from transformed matrix, coefficients and intercept. Test that inference is unchanged when unrelated rows/labels change and no fitted statistic sees future data.

Record all warnings, fitted iterations, model and feature parameters, input/output sizes, time and peak memory where measurable. Convergence warnings, nonfinite coefficients/scores, invalid dimensions or failed persisted-inference checks stop the affected execution before selection. Do not silently increase iterations, change solver/C or remove a failing recipe; any remedy changing the declared protocol requires a new study identity. Weighted outputs are scores, not asserted calibrated fraud probabilities. Coefficient tables and additive contributions describe predictive associations, not causal effects.

### 4. Comparable folds and exact finite outputs

Train each recipe on months 0 through m-1 and predict m in {4,5,6}. This needs 36 new full fits. Three selectable old references are the selected fusion, baseline_balanced and baseline_lgbm, with unchanged scores. Reuse the saved base models and adapters from the completed fusion study after verifying their original identities and saved inference: three XGBoost d4/w1 models, three balanced models and three unweighted models. They supply nine base dependency results; compose the fixed selected logit fusion for three additional score results. The XGBoost member is a dependency, not another selectable candidate. If all these caches exist, expected reference execution is nine verified base-model/prediction reuses and zero reference refits or model-verification fits. Fresh-clone absence triggers exact original recipe fits in the new destination, counted separately.

Thus 12 new selectable recipes plus three reference strategies give 15 strategies and 45 selectable strategy/month results. Retain three additional nonselectable XGBoost dependency predictions: 48 total score artifacts, without double-counting the six LightGBM reference artifacts. Save feature and solver diagnostics for all 36 new fits; metadata for all nine reused bases and all three derived reference fusions stays explicit. A new recipe fit failure is retained as a failure record and stops completion; do not shrink the grid and call it complete.

Apply unchanged pooled/latest/rank-fraction policies. Month 5 uses month-4 predictions; month 6 uses months 4/5, or only month 5 for latest. Policies optimize F2 beta=2 on eligible earlier predictions, without a fixed review fraction. Preserve existing score/ID ties and exact rational ceilings. All 15*3*2=90 forward operating rows, 45 macro rows and 45 pooled-confusion rows remain visible. Selection order is actual macro F2, then recall, precision and deterministic strategy/policy ID. Retain the three original pooled benchmarks exactly.

Export **237 contrasts** in five complete families: 135 against the three pooled references (15*3*3), 30 non-pooled-minus-own-pooled (15*2), 18 spline-minus-matched-linear (6*3), 18 balanced-minus-unweighted within representation/C (2*3*3), and 36 C-pair differences within representation/weight/policy (2*2*3 C pairs*3 policies).

Every contrast retains named factors, values and differences, including self comparisons, zeros and failures. Report F1/F2/F3, precision/recall, TP/FP/FN/TN, flagged count/fraction, AP and ROC AUC. Probability calibration and ranking metrics do not substitute for actual F2. Fit/data/feature choices remain frozen after development outcomes.

### 5. Predetermined transfer roles and bounded access

Always freeze five roles: selected (actual winning strategy/policy), fusion (old selected fusion/pooled), balanced (old balanced/pooled), unweighted (old unweighted/pooled), and same_model (winning strategy/pooled). If a new linear or spline recipe wins, add matched_representation (other representation, same C/weight/winning policy) and matched_weight (same representation/C, opposite training weight/winning policy). If an old reference wins, add no unmatched control role. Keep aliases in comparisons but deduplicate unique score/model work.

Freeze all eligible earlier final rules (months 4/5/6, latest uses 6), selected identity, dependencies, development tables/diagnostics and code/config/software hashes. Selection is immutable and further consumed-month access must record its SHA/timestamp. Transfer evaluates only these five or seven roles. The prior fusion always requires one XGBoost and one balanced dependency; unweighted supplies one more base. Reuse their checked saved month-7 models/predictions with no new reference fits when available. A new winner can require at most three new logistic final fits: selected, matched representation and matched weight. All train on months 0-6. Do not sweep the nine other new recipes, choose settings from month 7 or compute a hindsight optimum.

Report positive measured F2 gain against each named comparator on development and consumed scopes separately. The broader goal remains unachieved unless the selected strategy improves all three references on both scopes. A development-selected improvement below unweighted consumed F2 0.376884 is still a failed transfer comparison. Conditional zero-overlapping intervals preclude clear/confirmed gain language but do not change the actual-F2 selection rule. No outcome is independent confirmation.

### 6. Reporting, uncertainty and implementation completion

Use 500 paired fixed-decision row-bootstrap draws per development month 5/6 and consumed month, seed 1027, against every frozen comparator role. Share sampled rows across roles; development estimand is the mean of separately resampled monthly F2 differences. Retain 1000 development and 500 consumed draw rows, including comparator aliases. With five frozen roles there are 12 development interval rows (two monthly plus macro for four comparators) and four consumed rows; with seven roles there are 18 development rows and six consumed rows. Record fixed models/policies/decisions and all exclusions: adaptive multiple-study search, selection, retraining, temporal drift, generative uncertainty and multiplicity adjustment. A separate direct-row bootstrap implementation must reconstruct all draws and quantiles.

Report every new recipe's feature blocks, unique knots and fallback, scaler identities, convergence/iteration/warning evidence, coefficient associations and training/inference out-of-range rates. Inspect an operating-results figure and a selected-new-model smooth-effect figure when a new recipe wins; if a reference wins, inspect the best new spline's smooth effects as a clearly labelled diagnostic, without transferring that losing model to month 7. Smooth curves vary one original variable while fixing declared training-derived reference values, use a training-quantile range and show log-odds contributions, not causal effects or claimed calibrated probabilities. Evaluate only the selected frozen roles on consumed data.

Independently reconstruct all scores, policies, metrics, contrast factors/arithmetic, feature/solver identities, role selection, preservation and copied report hashes; visually inspect figures. README must include pinned environment/data access, all ten selection-restoration commands, absent-cache reproduction, stage commands, seeds and separate destinations. Update AGENTS/context to verified results only, acknowledge Codex and cited official APIs, and state no public notebook was copied. Run methodological tests, compileall and strict OpenSpec validation; synchronize/archive only after actual tasks and outputs are checked.

## Risks / Trade-offs

- Additivity may miss important interactions that trees already capture → retain full linear/tree/fusion controls and report negative findings without adding interactions after outcomes.
- Sparse/degenerate values and median-heavy fields can cause invalid knots or ill-conditioning → observed-only knots, deterministic deduplication/fallback, scaling, miniature fixtures and convergence gates.
- Constant extrapolation can conceal numeric drift → export per-column boundary exceedances and preserve the explicit fixed behavior.
- Different basis sizes change L2 geometry → report transformed dimensions and matched C controls; do not call this a complexity-matched comparison.
- Many adaptive studies increase selection optimism → expose all results/conditional limitations and never claim independent confirmation from consumed month 7.
- Full-data solver work can be expensive → cache only checked in-destination transforms/models, limit threads and record resources; do not replace full chronological evaluation with a small random sample.

## Execution and completion

Implement only in new files/destinations, with preservation guards before execution. Existing findings remain available throughout. If implementation fails, keep its failure evidence and tasks incomplete; do not mutate older studies. Main specs gain the new capability only after real outputs verify its requirements. Archive the change after synchronized specs and strict validation pass, leaving the performance goal active if measured acceptance fails.
