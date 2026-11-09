## Context

See `proposal.md` for motivation. The completed conditional-expert study is the preservation parent, `configs/conditional-lightgbm-experts.json`, with frozen selection SHA256 `10d9b3929fc4e7a375fb95bcc71a5397bd02ee9973160ac081d989b27d3a4474`. The baseline to beat is the bagged-anchor/pooled development F2 of 0.33088096180021687. Month 7 has been consumed repeatedly. This is an adaptive exploratory search, not an independent validation or an approved group decision.

`planning-enumeration.py` performs read-only ancestry/cache-presence checks and mechanical enumeration. Its `planning-evidence.json` contains 1860 mandatory path hashes, eighteen original/exported selection pairs, the 4848-file currently present historical snapshot, the planned 61-source closure, all 54 strategies, 162 winner/policy cases and 1728 contrast factor identities. It verifies Python 3.12.9, 27 package pins and the Base checksum without fitting, inference, policy evaluation or reading month-7 outcomes. Presence of the 32 required reference model sets is recorded separately from future saved-inference verification. A fresh clone need not contain ignored historical model files; mandatory exported evidence and exact selection restores remain required.

## Goals / Non-Goals

**Goals:** test reduced native CatBoost capacity, the fixed positive-cost intervention and complementary margin information using controlled comparisons and unchanged chronological F2 policies. Make every searched identity, actual fit, saved model, source reuse, negative result and acceptance decision independently inspectable.

**Non-Goals:** additional grids, feature engineering, learned combinations, probability calibration, early stopping, additional review constraints, new random-split evidence or restored holdout independence. No optional extension is bundled. The earlier 1% capacity assumption is absent from this objective.

## Decisions

### 1. Isolated immutable ancestry with a separate reference-adapter origin

Add only `configs/shallow-catboost.json`, `docs/shallow-catboost-protocol.md`, three new scientific modules (`baf/shallow_catboost.py`, `baf/shallow_catboost_study.py`, `baf/shallow_catboost_report.py`), `tests/test_shallow_catboost.py` and `scripts/check_shallow_catboost.py`. Destinations are `outputs/baf-shallow-catboost-v1/` and `reports/baf-shallow-catboost-v1/`. Reject overlapping destinations. The 58 existing scientific sources plus the three new modules give 61; test/runner and verification-source hashes are recorded separately. Freeze configuration, protocol, scientific sources, software and prospective evidence before execution. Preserve the stopped spline attempt and separate Newton fit/analysis identities.

The preservation parent and reference-adapter origin serve different purposes. Inherit the frozen conditional configuration's source recipes, but set the new study's `parent_config`/fingerprint to that conditional study. Declare `reference_adapter_config=configs/bagged-anchor-fusion.json` and its exact frozen fingerprint separately, equal to the conditional parent's original parent fields. An explicit adapter view uses the new destination and execution fingerprint while replacing only these parent-ancestry fields for the existing individual reference functions. Never pass this view to a predecessor's whole-study guard, initializer, grid runner or conditional miniature gate. Guard each predecessor using its own actual configuration.

Reference access order is own complete individual cache, original bagged-anchor origin, then its already-declared older individual origins. This retains original-reference precedence and avoids depending on the conditional study's unrelated local models. Use the individual reference adapters and derived-reference arithmetic only. Preserve original provenance links when copying sources; independently audit exact population, software, model recipe, preprocessing, original fit/analysis identity and saved inference. A wholly absent required source may reproduce its original recipe in the new destination. The first inspected partial/corrupt/incompatible set fails in place. Inspect only sources required by the declared six-reference DAG, never unrelated historical grids.

For derived scores, topologically reconstruct the original arithmetic from aligned audited members; if a corresponding inspected origin pair exists, require unchanged member dependencies and exact original score equality before recording reuse. Whole-absence derived scores can be recomputed without fitting. This explicit ancestry view avoids accidentally treating the conditional parent's fingerprint as the older bagged-anchor fingerprint.

### 2. Original raw features and fixed CatBoost recipes

Use BAF Base Kaggle version 2 at the frozen checksum. Decode the five original minus-one missing sentinels and negative intended balance exactly as `baf.common.features` does. Exclude `fraud_bool`, `month` and `device_fraud_count`, leaving 29 predictors. Fit `NativeInputs(catboost=True)` only on the earlier training population: numerical training medians, zero for all-missing numerical columns, fixed column order and the original five categorical columns (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`). Categorical missing values become `__MISSING__`; unseen strings remain native CatBoost categories. No standardization, one-hot expansion or engineered features are added to the new models. Existing logistic/tree references retain their own frozen preprocessing; new representations do not rewrite them.

The twelve recipes cross depth 3/4/6, iterations 200/400 and positive cost 1/100. Use learning rate 0.05, L2 leaf regularization 5, border count 128, maximum categorical-statistic complexity 1, Logloss, CPU, Plain boosting, SymmetricTree growth, seed 2026, six threads, `use_best_model=False`, `allow_writing_files=False` and `verbose=False`. The existing native factory can supply the adapter/constructor, followed by explicit CPU/boosting/growth/best-model settings. All omitted CatBoost settings remain the pinned 1.2.10 defaults, and the complete effective parameter dictionary must be exported and independently audited; do not invent unspecified defaults.

Use `class_weights=[1,w]` exactly once. Do not also pass sample weights, `scale_pos_weight` or automatic class weights. Cost 100 is a fixed intervention, not exact balanced weighting. Unlike reusing the earlier deeper models, the matched depth/round/cost grid tests shorter and shallower recipes on identical populations. Depth 6 is the matched deeper control; no claim is made that this grid exhausts regularization choices or equalizes compute.

Before and after fitting/persistence, snapshot constructor/effective metadata into detached JSON-compatible values. Do not mutate models through shallow copies or change parameters to simplify inspection. Verify binary class order, original schema and categorical indices, the requested settings, actual `tree_count_`, finite margins, no warnings and no unintended early stopping. Save a native `.cbm` plus a `.joblib` adapter/metadata artifact; bind both to the source/recipe/population hashes. Reconstruct training medians, levels, class costs and ordered ID/label identities independently. Reload the native model and adapter and compare saved scores at absolute 1e-12, relative zero. Verify sigmoid(raw margin) against `predict_proba[:,1]` at the same tolerance as a diagnostic; probabilities are neither stored as new selectable strategies nor claimed calibrated. Check single/batched/chunked/shuffled/unrelated-row/current-label inference invariance. Requested/effective recipe and training checks apply to every full-size model as well as tiny tests.

Official primary references establish the [raw prediction API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict), [class-cost and training parameter semantics](https://catboost.ai/docs/en/references/training-parameters/common), [margin/probability relationship](https://catboost.ai/docs/en/concepts/output-data_model-value-output) and [native binary serialization](https://catboost.ai/docs/en/concepts/python-reference_catboostranker_save_model). Choices of grid, cost, rounds and blends are analyst-selected; these references do not imply performance gains. Codex assistance and these sources must appear in the protocol/findings; no public notebook is copied.

### 3. Canonical raw-margin compositions and six fixed references

Pure names are `cb_d<d>_r<r>_w<w>`. Blend names are `cb_anchor_d<d>_r<r>_w<w>_a<10|25|50>`. Let M be `bagged_anchor_balanced_logit_a25`, an existing unbounded margin. The only new arithmetic is float64 `(1-alpha)*M + alpha*CB_RawFormulaVal`, evaluated in that association with alpha 0.10/0.25/0.50. Do not apply a second logit to M, clip either margin, normalize by batches or learn coefficients. Every member must have identical ordered row IDs and labels; nonfinite values or domain/alignment corruption fail. Zero and unit endpoints alias M and the pure CatBoost score. Inference is row-local; the learned rank-fraction operating policy is the separately declared batch decision rule.

The 54-strategy union is six references, twelve pure margins and thirty-six blends. All six references are selectable with all three policies; their reference comparator policies remain fixed:

| Role | Canonical score | Fixed comparator policy |
| --- | --- | --- |
| bagged_anchor | bagged_anchor_balanced_logit_a25 | pooled |
| latest_fusion | interaction_blend_spline_c1p0_balanced_a10 | latest |
| recent_fusion | margin_blend_spline_c0p1_unweighted_a25 | latest |
| fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled |
| balanced | baseline_balanced | pooled |
| unweighted | baseline_lgbm | pooled |

Their deduplicated DAG contains eight original bases (three balanced bagged seeds, two original LightGBM baselines, two spline models and one XGBoost model) and five derived scores, totaling thirteen physical scores per month. Original score domains and clipping rules remain as frozen. No extra logit/probability aliases enter the new search.

### 4. Chronological operating policies and actual objective

For prediction months 4/5/6, train each model on all months strictly earlier than that prediction month. Fit the month-5 operating rule from month-4 out-of-time predictions; fit month-6 pooled/rank rules from months 4/5 and latest from month 5. Month-4 scores provide policy history, not a self-fitted evaluation fold. Transfer rules use months 4/5/6 (latest uses 6) and final models use months 0-6. No current evaluation labels inform fitting, composition or decisions.

Retain `pooled`, `latest` and `rank_fraction` exactly as specified by the existing F2 policies. Thresholds respect equal-score groups. Rank fractions use earlier F2 optimization and exact rational ceiling; score ties use stable ascending original row ID. They are not a 1% quota. Select the actual mean of the two month-5/6 F2 values, then macro recall, macro precision, candidate ID and policy ID. Do not select on AP, pooled F2 or a consumed-month result.

Positive-class F2 is `5*TP/(5*TP+4*FN+FP)`, a recall-weighted F-score rather than scikit-learn's support-weighted F1. No precision floor or review budget is imposed. Report precision, recall, F1/F2/F3, AP, ROC AUC, MCC, balanced accuracy, FPR, accuracy, confusion counts and flagged fraction; these supplement rather than replace the objective. Retain all negative effects and exact ties, calculating differences from the same authoritative numeric table so serialization noise cannot turn a self-alias into a gain.

### 5. Complete declared outputs, factors and work

With available references, development makes 36 new CatBoost fits, reuses 24 original base results and fifteen derived reference scores, and calculates 108 new blends. Sixty model objects use 99 serialized model files: 27 old files (XGBoost has two per model) and 72 new native/adapter files. There are 183 full physical scores and no conditional subsets; 162 are selectable strategy/month scores. Verify 324 chronological operating rows, 162 macro and 162 pooled summaries against direct reconstruction. Any exact wholly absent reference reproduction is additional separately counted work, bounded by nine bagged, six spline and nine other-tree development fits, with no old-grid replay.

The prospective contrast union has the following 1728 unique keys; the full identities/factors are in planning evidence, not merely these row totals:

| Contrast | Rows | Control |
| --- | ---: | --- |
| fixed_reference | 972 | every strategy/policy against all six fixed-policy references |
| own_policy | 108 | latest/rank minus own pooled |
| matched_depth | 144 | all three depth pairs, holding rounds/cost/alpha/policy fixed |
| matched_rounds | 72 | 400 minus 200, other factors fixed |
| matched_cost | 72 | 100 minus 1, other factors fixed |
| native_vs_anchor | 36 | native minus M at the same policy |
| blend_vs_native | 108 | blend minus its pure native source at the same policy |
| blend_vs_anchor | 108 | blend minus M at the same policy |
| higher_alpha | 108 | all interior alpha pairs, other factors fixed |

Depth/round/cost controls include pure scores and all three interior blend weights. Export 183 physical-score and 162 selectable-score diagnostic rows, 36 native-to-anchor diversity rows (one model/month), and 504 decision-overlap rows: every new native versus anchor and every blend versus anchor/pure, at three policies and two evaluation months. Include quantiles at 0/0.01/0.05/0.25/0.5/0.75/0.95/0.99/1, min/max, unique count, duplicate-score fraction `(n-unique)/n`, out-of-[0,1] counts and the declared domain (finite margins may lie outside [0,1]). Export native/anchor Pearson and Spearman correlations; degenerate constant members retain a row with null correlation and an explicit reason, not NaN or a dropped case. For overlap, record both flagged, first only, second only and neither for fraud and legitimate cases separately, so all TP/FP/FN differences can be reconstructed. Diagnostics cannot add candidates or select new transforms.

Export sixty model/work rows, 36 new preprocessing records and 36 complete new parameter records. Record actual rows/classes/rounds/trees, training row-rounds, fit wall time, process high-water RSS, model bytes and source reuse/reproduction. Account for shared bases and blends once, identifying logical per-method row-round exposure separately from physical prediction/audit calls. Nominal rounds and trees do not equalize categorical-statistic work, runtime, memory or storage. Full independent auditing must inspect all sixty available-cache saved models and every physical score, not only the winner or composition metadata.

### 6. Gates before production scoring

The apply phase first verifies the exact prospective contract and preservation snapshot, then records a complete methodological suite with a new fit-accounting runner. Plan eighteen current tiny calls per complete invocation: twelve native grid-format fits and six old required-format fits (balanced/unweighted ordinary and bagged LightGBM, spline and XGBoost). Separately observe and count historical-suite calls; the preserved full suite currently implies 85, not an asserted result of the new runner. Record actual counts for failed, focused, full and clean-copy invocations; do not edit frozen runners/tests to change history. Tests must cover all 162 winner/policy cases, exact identity/factor sets, altered hashes/settings/keys/numerics, whole-absence reproduction, partial/corrupt rejection, actual inference and no unrelated source access. Tiny fixtures are synthetic and separate from scientific scores.

Before new production fits, run a predictor-only actual-host preflight on development rows. Retain the original maximum required source representation bound: with n rows and d=max(original encoded tree dimension, additive spline dimension), `matrix=12*n*d+8*(n+1)` and `source_workspace=4*matrix+8*(d+1)^2+4*1024^3`. Additionally profile the actual native input's deep memory and reserve four times that plus four 32-bit raw-feature buffers (`16*n*29` bytes) and a `6*2^6*29*129*16` histogram allowance. Add these to the source workspace and reject estimates above 70% of actual RAM. Export dimensions, allocations, host RAM, timing and measured high-water RSS. The categorical/internal allocation allowance is a conservative feasibility proxy, not a mathematically guaranteed process peak; retain actual allocation failures and measured resource evidence. Do not silently replace the recipe to fit the host.

Audit the preserved nine-fit sampling and eight-fit older source gates using the explicit reference-adapter view, verifying all mandatory receipts and any complete optional miniature artifacts if present. Reject partial optional sets. No new old-source miniature fits or sixteen-fit conditional-expert gate are required: the new pipeline has no conditional experts, and that original evidence is preserved as parent history.

Run a separate fourteen-fit production CatBoost miniature gate: the twelve exact grid recipes plus deterministic repeats of `cb_d4_r200_w1` and `cb_d6_r400_w100`. Keep a started/completed/failure ledger and all serialized artifacts. Cover original 29-column schema, missing/all-missing numerical values, all categorical fields, unseen/missing categories, both labels, row-order/chunk/single-row/unrelated-row/current-label invariance, requested/effective settings, margins, sigmoid consistency, exact round count, warnings and persistence. Require repeated margins to agree at absolute 1e-12/relative zero. Every completed miniature fit is separate from eighteen current methodological tiny calls and from full development/transfer fits. Failed gates retain partial work and block completion; no automatic retry with changed settings or numerical tolerances.

### 7. Predetermined transfer cases and exact bounds

Freeze only after the full independent development audit passes. Record actual winner, six references, own pooled policy, original earlier rules, all role aliases, exact DAG, required individual cache states and tighter per-type work bounds. Every reference winner retains eight roles: selected, six references and same-strategy pooled. Every new native/blended winner retains twenty-two: those eight plus anchor at winner policy, pure native, three depth roles, two round roles, two class-cost roles and five alpha roles (0/10/25/50/100). All controls use winner policy; depth/round/cost controls keep the winner's other factors and alpha. Aliases remain visible but do not create work or gains.

| Winner type | Visible roles | Old bases | New CatBoost recipes | New blends | Full scores | Model objects / files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 8 | 8 | 0 | 0 | 13 | 8 / 9 |
| native | 22 | 8 | 5 | 3 | 21 | 13 / 19 |
| blend | 22 | 8 | 5 | 7 | 25 | 13 / 19 |

The five required native recipes are the winner's three depths, alternate round count and alternate class cost, deduplicated. Pure winners still retain the three same-recipe interior blends as predetermined composition controls. Blend winners additionally require the pure scores behind matched controls. Do not sweep all twelve native recipes or all thirty-six blends on month 7. If a declared individual final native artifact already exists under the same new frozen identity, audit/reuse it and reduce the actual fit bound. Otherwise only the five declared recipes may be fitted. The current reference origin sets are present; if verified reuse succeeds, original-reference new-fit bounds are zero. Wholly absent required origins permit at most three bagged, two spline and three other-tree exact reproductions, tightened by actual availability. Partial/corrupt sets fail. Freeze model-file/score/type/work bounds, not a generic maximum alone. Check all 162 prospective cases and deduplicated operating pairs against planning evidence before access.

### 8. Independent reporting and scientific acceptance

Independently reconstruct all saved scores, training preprocessing/costs, requested/effective recipes, original source provenance, score composition, chronology, policies, decision masks, metrics, factor sets, work bounds and actual selection. The auditor must derive these from underlying data/models and its own calculations, rather than merely reusing producer totals/helpers. Audit development before freeze and audit only the frozen transfer DAG afterward. Retain failures/corrections as separate dated evidence; preserve producer identities once frozen.

Use 500 paired fixed-decision row-bootstrap draws for each of months 5/6/7, seed 1027: 1500 total. Reuse each month's same resampled rows for every role; keep original decision masks fixed. Reconstruct macro-development draws as the mean of month-5/6 differences without refitting/reselecting policies. Export selected-minus-every-other-visible-role intervals for month 5, month 6, macro development and consumed transfer: 28 rows for reference winners or 84 for new winners. Role aliases remain visible with exact-zero differences. Independently reconstruct every draw and interval, including directions, seeds and deterministic ties. Intervals exclude retraining, adaptive source/recipe/policy/selection search, temporal/generative variation and multiplicity.

Produce a six-panel scientific figure and its exact plotted-value export: all 162 macro precision/recall operating points; twelve canonical alpha curves for each of three policies (36 curves, 180 rows including endpoints); matched depth/round/cost effects; and six-reference selected differences/conditional intervals in both macro-development and consumed scopes. Label model costs/rounds/depths and operating policies clearly, preserve negative effects and aliases, check arrays against authoritative tables, regenerate and visually inspect the rendered image for clipping and legibility.

Broader achievement requires strictly positive selected-minus-reference F2 against all six fixed references in both macro development and consumed transfer. The incumbent alias can never pass its own comparison. Clear-gain wording additionally requires positive conditional interval lower bounds for the claimed scope. Even a pointwise win remains exploratory because the sources and recipes were chosen after eighteen studies and month 7 is already consumed. Completing all finite tasks may yield another useful negative result while the broader goal remains active.

### 9. Reproduction, manifests and ordered execution

The apply phase must export actual checked commands corresponding to: install the existing pinned requirements; place/verify Base; restore eighteen original selections exactly from exported counterparts; verify the contract and declaration; run/record full methodological checks; run preflight/source gates and miniature; prepare development; validate and independently audit; freeze; run bounded exploratory transfer; independently report/audit; verify figure values/rendering, preservation and closed-log manifests; then validate/synchronize/archive completed requirements. Proposed CLI names are `verify_declaration`, `record_checks`, `preflight`, `miniature`, `prepare_development`, `validate`, `audit_development`, `freeze`, `exploratory` and a separate report command; the final README must use the implementation's actual command interface, not this prospective list as a tested claim.

Run the exact setup/restore block and complete methodological suite in a disposable copy with no prior model/prediction caches. Verify all pins, Base checksum, eighteen matching restored selections, rejection of altered records, archived-path planning evidence resolution and actual tiny-format inference. This is setup/methodology evidence; do not claim full-size clean-copy retraining unless it actually occurs. Report actual current/historical test fits separately from production miniature, full new fits, exact absent-source reproduction and reused models. Keep original warnings, failed attempts and portability receipts. Hash only closed logs; exclude index files from their own manifests and verify all indexed members before completion. Update README, AGENTS, current OpenSpec context and assistance attribution to actual checked status.

## Risks / Trade-offs

- Adaptive search and consumed transfer can overstate generalization → preserve all eighteen studies, distinguish conditional uncertainty from independent confirmation, and retain every failed comparison.
- Fixed cost 100 can worsen ranking or shift margins → retain cost-one matched controls and actual F2/flag/precision/recall outcomes; do not interpret scores as calibrated risks.
- Mixing raw margins depends on model scale → keep exact fixed arithmetic and native/anchor/alpha controls; do not add hindsight normalization.
- Smaller depth/round counts can underfit → retain depth-6/400-round controls and all negative effects; do not alter the grid after scoring.
- Native serialization and default parameters can drift → use the pinned environment, detached metadata, full effective-parameter export and independent persisted inference.
- Historical-cache reproduction can add work → freeze exact required origins and type bounds, count reproduction separately and fail on inspected partial/corrupt sets.
- Resource estimates can miss native internals → profile actual predictors, retain a 70% feasibility gate and measured RSS/failure evidence without claiming a guaranteed memory ceiling.

## Migration Plan

This is additive and local. Planning creates no candidate configuration, production source, model or scientific score. Apply creates only the isolated study and progress documentation after this plan is presented. Existing datasets and eighteen frozen studies remain immutable. A failed or unhelpful run remains recorded in its own destination; rollback means stopping its execution without deleting evidence or rewriting predecessors. Only fully verified finite work is synchronized/archived, independently of whether the broader performance goal is achieved.
