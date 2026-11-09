# Newton additive spline-logistic protocol

Prospective analyst-selected numerical remedy; beta=2 with no quota or precision floor. Read this declaration before fitting.

## Context

See proposal.md for motivation and specs/baf-newton-spline-logistic-study/spec.md for acceptance. The stopped L-BFGS study has nine successful month-4 fits, one declared convergence failure, no selection and no further month-7 access. All ten earlier completed studies remain frozen. The three pooled references are selected fusion `logit_mean_xgb_d4_w1_baseline_balanced_a50`, balanced LightGBM and unweighted LightGBM. Their macro development F2 is 0.3289065701775199, 0.32780157809618937 and 0.3226536849212962 respectively; consumed-month F2 is 0.3726941226941227, 0.3644085570469799 and 0.3768844221105528. These adaptively reused observations provide no independent confirmation.

This design is needed for numerical objective verification, fallback observation and preserved cross-study dependencies across new execution/report modules. All settings are analyst-selected and open to group review.

## Goals / Non-Goals

**Goals:** Evaluate the same fixed representation objectives using a checked numerical remedy, with complete chronological comparisons and explainable saved inference. Keep numerical validity separate from detection performance.

**Non-Goals:** Change predictor roles, add interactions or ensembles, expand the C/knot grid, introduce resampling, optimize AP, impose a review quota/precision floor, adapt numerical gates after outcomes or recover a fresh holdout. Reusing partial old predictions as selectable Newton results is excluded.

## Decisions

### 1. New identity and preserved dependencies

Use unchanged Base v2 SHA-256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`, existing Python/package pins and new files `configs/newton-spline-logistic.json`, `docs/newton-spline-logistic-protocol.md`, `baf/newton_spline_study.py`, `baf/newton_spline_report.py` and `tests/test_newton_spline_logistic.py`. Destination roots are `outputs/baf-newton-spline-logistic-v1` and `reports/baf-newton-spline-logistic-v1`.

Fingerprint configuration, Python/software, the preserved 33-source dependency closure plus the two new modules, original/exported selections from all ten completed studies and data identity before execution. Verify the stopped source/config/report identities against its tracked records. At initialization, check every locally available stopped bulk artifact against its recorded hash and snapshot its presence. Require originally present artifacts to remain unchanged at each stage. A fresh clone can lack ignored stopped caches: disclose unavailable optional diagnostics instead of rerunning the failed attempt. Tracked stopped failure/evidence records and mandatory frozen dependency identities remain required.

Reuse the preserved additive adapter and generic score/policy/report helpers through explicit configuration arguments. Add new solver, fit/audit, validation, freeze, transfer and report entry points because the stopped study's corresponding entry points enforce L-BFGS. Do not edit or monkey-patch stopped modules or their globals. Keep the stopped change incomplete at its actual 5/14 tasks; this successor does not complete its missing execution.

Required reference or new-run caches need complete compatible model/prediction/provenance records. Wholly absent required old caches can be reproduced only under their frozen recipes in this new destination and counted as fits. Partial/incompatible required caches are errors. Replacing or overwriting stopped artifacts is not an alternative.

### 2. Unchanged paired predictor representation

Reuse `AdditiveInputs` from the preserved `baf/spline_inputs.py`. All 29 decoded original predictors remain; exclude fraud_bool, month and device_fraud_count, retaining existing sentinel decoding. Training populations are expanding earlier months without sampling or recency weights.

Five categorical columns: payment_type, employment_status, housing_status, source, device_os. Training-only one-hot vocabularies use explicit missing values and ignore unseen categories. Seven discrete numeric columns stay linear: email_is_free, phone_home_valid, phone_mobile_valid, has_other_cards, foreign_request, keep_alive_session, device_distinct_emails_8w. Seventeen smooth columns: income, name_email_similarity, prev_address_months_count, current_address_months_count, customer_age, days_since_request, intended_balcon_amount, zip_count_4w, velocity_6h, velocity_24h, velocity_4w, bank_branch_count_8w, date_of_birth_distinct_emails_4w, credit_risk_score, bank_months_count, proposed_credit_limit, session_length_in_minutes.

Fit unweighted training medians (zero for entirely missing columns) and append all 24 numeric missing indicators. The linear control uses imputed original numeric values. The spline representation replaces each smooth column with a cubic basis using sorted unique observed-only training quantiles [0,.25,.5,.75,1], linear quantile interpolation, omitted implicit bias and constant extrapolation. Fewer than two unique observed knots uses the identical imputed linear column with explicit constant/all-missing metadata. Numeric/basis/indicator outputs use training-only variance scaling without centering; one-hot categories remain unscaled. Outputs are finite float64 CSR with fixed ordered feature names and dimensions. No labels or class weights enter preprocessing. Exact compatible in-destination transform reuse within a representation/fold is allowed.

Retain per-column medians, knots, categories, scale, fallback, names, training reference values and 1–99% ranges; export training/future missingness, boundary exceedance and unknown-category diagnostics. Future values do not update preprocessing. This keeps the original [spline API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) behavior and isolates the numerical remedy. Different basis dimensions change L2 geometry, so matched C is not a claim of matched effective complexity.

### 3. Fixed Newton recipes and observable numerical gates

Twelve recipes cross representation {linear,spline}, C {0.01,0.1,1} and class_weight {None,balanced}. Use `solver=newton-cholesky`, `l1_ratio=0`, fit_intercept=True, tol=1e-6, max_iter=100, warm_start=False and seed=2026; native thread pools are limited to six. Do not pass deprecated penalty/n_jobs. Random seed is recorded for recipe identity even though this solver does not shuffle. Balanced weights are n/(2*n_class) from that training fold only.

The [official logistic API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) identifies Newton–Cholesky as suitable when rows greatly exceed transformed dimensions, with quadratic Hessian memory. The stopped month-4 matrices had 74 linear and 157 spline features. Record each new matrix's actual dimensions, density and estimated dense Hessian storage `8*(d+1)^2`, plus time and peak memory where measurable. This is a numerical rationale, not a speed or F2 guarantee. Raising the old L-BFGS iteration limit would mutate its recipe; an unbounded solver search would expand scope. Both are excluded.

For each trained model independently compute, using its full training matrix, weights and unpenalized intercept:

```
z = X @ coef + intercept
S = sum(w)
objective = sum(w * (logaddexp(0,z) - y*z))/S + dot(coef,coef)/(2*C*S)
residual = w * (expit(z) - y)
gradient_coef = X.T @ residual/S + coef/(C*S)
gradient_intercept = sum(residual)/S
```

This normalization matches the installed pinned implementation's `l2_reg_strength=1/(C*S)`, checked against local source before planning. Verify the independent equations with dense/sparse finite differences, hand-computed weights and the pinned loss implementation before numerical execution. Require finite objective/coefficients/intercept/gradient and maximum absolute gradient <=1.1e-6, allowing a fixed small arithmetic margin around solver tol=1e-6. Require positive actual n_iter below 100, no fit warnings, classes [0,1], valid feature dimensions, and finite reconstructed scores. A fit at the iteration limit is rejected even if other diagnostics look acceptable. These prospective gates do not change after outcomes.

Warnings alone cannot prove the absence of fallback: the pinned implementation can enter `fallback_lbfgs_solve` after a non-descent direction without a warning. Around each fit, use a narrow, scoped Python call-profile observer to detect entry to the pinned fallback function's code identity, restoring the prior profiling state in a finally block. Record the observed fallback count and observer/solver code identity. Refuse to execute if a competing profiler cannot be safely preserved; do not alter library functions or frozen modules. Verify this observer with forced warning-free fallback entry as well as warning cases. Any fallback event rejects the fit, including an otherwise converged final model. Keep observed fallback distinct from the configured public solver string.

Perform two deterministic miniature fits per recipe, 24 fits counted separately, using the preserved mixed/degenerate fixture and independent gates. Verify batch/label invariance, coefficient-margin reconstruction and joblib round-trip scores at atol=rtol=1e-12. Record warnings, objective, full gradient maximum, iterations, identities and actual resources. On any full-data failure, retain available numerical diagnostics and failure provenance, stop before selection, leave tasks incomplete and preserve completed work. No retry with altered solver/limit/C/gate is part of this study.

### 4. Optional old-solver objective diagnostics

Where the nine successful old month-4 saved models exist, verify their stopped identity, matching ordered populations, representation statistics, C and weighting. Evaluate both old and new models' training objectives and gradients on the identical training matrix; export their differences and actual optimization residuals. Do not fit old models if these optional caches are absent. Missing diagnostics are explicit, not filled with invented values.

These rows are outside F2 selection and the 237 performance contrasts. There is no demand for an arbitrary positive loss improvement: two valid optimizers can agree within tolerance. The acceptance gate concerns the new model's own gradient, warnings, iterations and solver path. Numerical objective improvement alone is not fraud-detection improvement.

### 5. Full chronological grid and matched outputs

Fit every new recipe on months 0..m-1 and predict m in {4,5,6}: 36 full development fits. Preserve all rows and stable IDs. Three selectable references retain unchanged scores: fixed selected fusion, baseline_balanced and baseline_lgbm. Use the completed fusion study as reference source_config. Verify/reuse its three XGBoost d4/w1, three balanced and three unweighted saved model/prediction pairs, then derive the three fixed fusion scores. With caches available, reference work is nine base reuses, zero reference fits and three derived fusions. Required absent-cache reproduction is counted separately.

There are 15 selectable strategies, 45 selectable scores and three nonselectable XGBoost dependencies: 48 total score artifacts. Keep dependency provenance separate from candidate counts. Never replace any new Newton fit with one of the nine old L-BFGS results.

Use unchanged pooled/latest/rank-fraction policies and exact score/ID ties and rational ceilings. Month 5 policies learn from month 4; month 6 policies learn from months 4/5, or only month 5 for latest. The rank fraction is learned for F2 and is not a fixed 1% quota. Export 90 forward operating rows, 45 macro summaries, 45 pooled-confusion rows and exactly 237 contrasts: 135 against all three pooled references, 30 non-pooled versus own pooled, 18 spline versus paired linear, 18 balanced versus paired unweighted and 36 C-pair contrasts within representation/weight/policy. Retain all factors, self comparisons, ties and negative differences. Selection uses actual macro month-5/6 F2, then recall, precision and deterministic candidate/policy IDs.

Report F1/F2/F3, precision, recall, TP/FP/FN/TN, flagged count/fraction, AP and ROC AUC. F2 is `5TP/(5TP+4FN+FP)` with no quota or precision floor. Export 36 solver rows, 36 feature-block metadata records, all coefficient rows and 2,088 feature-diagnostic rows (36 fits*29 predictors*training/evaluation), plus optional solver-comparison rows. Reconstruct these independently before freezing; incomplete output cannot be called a complete grid.

### 6. Predetermined limited transfer

Always freeze five roles: selected winner/policy, fusion/pooled, balanced/pooled, unweighted/pooled and same_model (winning strategy/pooled). If a new recipe wins, also freeze matched_representation (same C/weight/winning policy, other representation) and matched_weight (same representation/C/winning policy, opposite weight). If a reference wins there are no extra roles. Keep role aliases for comparisons while deduplicating model work.

Freeze selected identity, full source/config/software and development output hashes, dependencies and final earlier-only rules before further month-7 access. Final policy inputs are months 4/5/6, latest uses 6. Transfer evaluates only these five or seven roles with an immutable access record. Reference work needs three distinct bases (XGBoost, balanced, unweighted), reused with checked identity when available. A new winner needs at most three new final logistic fits, all trained on months 0-6. No losing-grid sweep or hindsight threshold is allowed.

The broader goal remains unachieved unless the selected strategy exceeds all three references in measured F2 on both development and consumed transfer. Report individual passed/failed comparisons explicitly. Clear-gain claims require positive conditional interval bounds for the named scope; overlapping zero does not alter actual-F2 selection. No consumed-month result is independent confirmation.

### 7. Checked reporting and reproduction

Use seed 1027 and 500 paired fixed-decision row-bootstrap draws per month 5,6,7, sharing resampled rows across roles. Development macro differences average separately resampled monthly F2 differences. Retain 1,000 development draw rows and 500 consumed draw rows with every comparator alias. Five roles imply 12 development interval summaries (two months plus macro for four comparators) and four consumed summaries; seven roles imply 18 and six. Independently reconstruct every draw and quantile using direct row decisions. Disclose fixed-model/policy conditioning, adaptive search and selection, excluded retraining/temporal/generative uncertainty and lack of multiplicity adjustment.

The new report entry point reuses pure table, draw and additive-curve helpers via explicit configuration, adding Newton objective/gradient/path evidence and optional old-solver diagnostics. Inspect the operating figure and smooth-effect figure. Use the selected new model's effects if it wins, otherwise the best new spline's month-6 effects as a development-only diagnostic. Curves vary one original predictor over training-derived 1–99% ranges with fixed training reference values and report log-odds associations. Reconstruct matrix/coefficient contribution changes; do not imply causation or calibrated weighted probabilities.

Independently verify all saved scores, policies, metrics, contrast arithmetic/factors, solver/feature identities, role selection, freeze/access records, copied report hashes and preservation. README must document pinned dependencies, data access/checksum, all ten small selection-restoration commands, optional stopped-cache absence, exact-reference absent-cache reproduction, ordered preflight/validation/freeze/transfer/report commands, seeds and outputs. Update AGENTS/context to actual state; cite substantive Codex/API assistance without claiming copied public notebooks. Run relevant methodological tests, compileall and strict OpenSpec validation. Main capability synchronization and archive occur only after all actual tasks and outputs verify.

## Risks / Trade-offs

- Newton may still fail despite appropriate dimensions → fixed objective/gradient/path gates and retained failure records; no adaptive fallback.
- Hessian work is quadratic in dimensions and expensive over many rows → record actual matrix/Hessian sizes and resources, limit threads and persist only checked compatible results.
- A profile observer depends on pinned solver internals → record code identity, test warning-free fallback detection and restore profiling state; never infer solver path from configuration alone.
- Additivity may miss tree interactions → retain all three original references and complete matched controls; report negative results rather than silently expanding features.
- Constant tails and different L2 geometry affect interpretation → retain drift diagnostics and dimensions and explain conditional predictive associations.
- Repeated development/consumed-month reuse increases optimism → preserve the adaptive history and conditional limits; no claim of fresh confirmation.

## Migration Plan

Only new study files and destinations are introduced. First verify preservation, then numerical tests and miniature fits, full development and independent audits, freeze, bounded exploratory transfer and final reporting/audit. A failure leaves its evidence and incomplete tasks visible; it does not mutate earlier studies or their selections. Synchronize the new capability and archive only after verified completion, leaving the broader goal active if measured acceptance fails.


## Analysis-only verifier repair

The first complete Newton development run fitted all 36 declared models but stopped before selection when `verify_table` compared a sorted DataFrame's inherited indices with a CSV's reset indices. Resetting the index proves identical exported values; this was not a solver or performance failure. The original fit source/config, 36 model/score records and their 35-source fingerprints remain intact, with copies and the original terminal error under `outputs/baf-newton-spline-logistic-v1/audit-repair-original/`.

`baf/newton_spline_audit.py` and `baf/newton_spline_audit_report.py` provide the corrected analysis entry points. They delegate all fitting to the unchanged original functions and use separate complete 37-source analysis fingerprints. `analysis_declaration.json` records the retrospective audit repair, timing, preexisting fit count and hashes of every existing model/score/run artifact. It is included in frozen validation evidence; selection additionally freezes the analysis fingerprint. Both identities are checked before any further consumed-month access. Numerical recipes, policies, selection criteria, output sets and transfer roles are unchanged. Required compatible models are reused without refitting; the corrected entry points also support an initially empty destination.

Use `baf.newton_spline_audit` for miniature/validate/freeze/exploratory and `baf.newton_spline_audit_report` for reporting. Original entry points remain preserved as provenance; the corrected paths are the runnable reproduction interface. All actual outputs still require the planned independent checks before completion.
