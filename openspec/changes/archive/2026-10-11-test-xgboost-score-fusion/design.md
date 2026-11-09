## Context

See `proposal.md` for motivation and `specs/baf-xgboost-score-fusion-study/spec.md` for the contract. Nine studies are complete and frozen. Balanced LightGBM with the pooled earlier-prediction threshold remains the macro development F2 reference at 0.3278015781; unweighted LightGBM is 0.3226536849. Consumed-month F2 is 0.3644085570 and 0.3768844221 respectively. Development data has been reused adaptively; month 7 cannot provide independent confirmation.

The ninth study's config SHA-256 is `45311327a661a429a2a851e2d6a90462fb12794bbdeca28a80ba2f7004c64225`; its original and exported selection hash is `f95dd200148b98c9dc2f77343b7a74f08f2666b7e92dfe8019f25685fd1f22f8`. Its guarded fingerprint contains 27 source/dependency files, Python 3.12.9 and the pinned existing packages including XGBoost 3.4.1. Preserve these exact identities, not merely equivalent model settings.

## Goals / Non-Goals

**Goals:** Isolate composition effects using already-fitted models; check a finite score grid and matched controls under actual forward F2; reuse exact caches safely; retain complete negative evidence and limited frozen exploratory transfer.

**Non-Goals:** Learn blend weights, change beta, add features, recalibrate scores, impose a workload quota or precision floor, add a model package, retune base models, or treat a completed experiment as a performance gain. Settings below are analyst-selected and open to group review, not approved bank operating costs.

## Decisions

### 1. Preserve the data, training recipes and software

Use BAF Base version 2 at `data/raw/baf/Base.csv`, SHA-256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. Base models for prediction months 4/5/6 train on all strictly earlier rows. Final permitted dependencies train on months 0-6. The same stable row IDs and labels must align across every compared score.

Reuse the four ninth-study XGBoost presets, depths 4/6 crossed with positive weights 1/5, and its three references: balanced baseline LightGBM, unweighted baseline LightGBM, and native31. Keep all 800-round XGBoost settings, seed 2026, six threads and native categorical inference exactly frozen. The 29 original predictors exclude `fraud_bool`, `month` and constant `device_fraud_count`; training-only category vocabularies, numerical missing values and unseen-category handling remain those of the saved adapters. LightGBM references retain their exact frozen recipes and model-specific preprocessing. There is no new feature comparison or logistic model in this study.

No dependency changes: install from unchanged `requirements-xgboost.txt`, verify the pinned versions, and reject drift before reuse. A broader base search was considered but would confound composition with training and add another adaptive grid. Existing model semantics remain covered by the saved integration identity and methodological tests; miniature test fixtures are counted separately from study fits.

The new guard delegates the old recursive identity guard, verifies its source fingerprint and adds its config plus original/exported selection to the old 19 protected entries: 22 preservation entries in total. The new fingerprint extends the old 27 files with the three additive fusion modules below, yielding 30 source/dependency keys. Both new destinations must be disjoint from each other and every preserved bulk/report directory, including ancestor/descendant overlap.

### 2. Audit and reuse complete caches, reproduce wholly absent ones

For each required base/month, distinguish a complete source cache, no source cache, and partial/incompatible cache. Reuse only after checking frozen fingerprint, exact model parameters, adapter identity, eligible training IDs, evaluation population, model/adapter/prediction hashes and saved-model inference. Reject partial or incompatible caches without silently refitting them.

Copy a verified artifact into the new destination with explicit source paths, source fingerprint and source-record/artifact hashes. The new record carries the new study identity and preserved origin provenance; do not relabel an unaudited old cache as compatible. Reuse the old audit/inference helpers unchanged. Old reference adapters may be delegated through existing new-destination reproduction helpers. A wholly absent source cache can be fitted under the exact old recipe only in the new destination. New caches receive the same identity and save/load audit; a partial new cache is also rejected.

Fresh-clone reproduction restores nine small exported selection records. Do not require every old ignored bulk artifact merely to load an old selection: validate frozen recipes and protected selection identities, audit actual caches when present, and reproduce missing dependencies in the new study. Requiring all historical manifests would make the documented fresh-clone path impossible.

With complete caches there are 21 verified development base reuses and zero new full fits: seven members times three months. With all caches absent there are at most 21 frozen-recipe development refits, 12 XGBoost and nine references. Report actual reuse/refit counts, fit times and warnings separately. Native model/adapter/prediction reconstruction is mandatory either way. Older reference development caches have complete predictions but no saved-model contract. For these, separately count exact-recipe model-verification fits, require their predictions to match the frozen scores, and save their models in the new destination. A wholly absent reference prediction/model pair is fitted together once; it does not require a second verification fit. Existing final reference models may be reused only after recipe and inference checks.

### 3. Define row-wise score transforms and matched controls

Declare exactly 61 strategies before evaluation:

| Type | Count | Definition |
| --- | ---: | --- |
| Raw singles | 7 | Four XGBoost and three references |
| Logit singles | 6 | Four XGBoost plus balanced/unweighted baseline references |
| Raw fusions | 24 | Four XGBoost × two baseline anchors × alpha 0.25/0.5/0.75 |
| Logit fusions | 24 | Same members and weights as each raw fusion |

For finite aligned base scores in [0,1], define `raw_mean = alpha*p_xgb + (1-alpha)*p_anchor`. Define `clip(p)=min(1-eps,max(eps,p))` with fixed `eps=1e-6`, `L(p)=log(clip(p)/(1-clip(p)))`, and `logit_mean = alpha*L(p_xgb)+(1-alpha)*L(p_anchor)`. Keep the latter as a finite real-valued score; do not sigmoid or clip the composed result. Single-logit controls use the same L. The [official SciPy logit definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logit.html) explains the endpoint infinities that motivate clipping; the particular epsilon and fusion grid are our choices, not claims from that source.

Reject empty/non-vector/non-finite/out-of-range inputs, invalid weights, undeclared members and population misalignment. Source labels may be checked for population integrity but must not enter score arithmetic. Transform functions accept scores alone; changing labels or unrelated batch members leaves a row's fused score unchanged. The logit transform is strictly increasing away from clipped tails; clipping can create ties, which must remain visible. Test hand-calculated endpoints, weight orientation, swapped-member symmetry, unclipped monotonicity, clipping ties and label/batch independence.

Raw averaging and logit averaging test different score geometries; neither corrects training-weight distortions nor supplies calibrated probabilities. Keep alpha fixed for each candidate, do not estimate it from the scored month, and do not introduce a prior correction. Percentile averaging is a preserved earlier experiment; omitting batch ranks here isolates a different composition mechanism. Single-logit controls distinguish numerical transformation/clipping effects from fusion effects. Native31 stays a raw reference only because it is not a fusion anchor in this grid.

### 4. Keep chronological policies and actual-F2 selection unchanged

For month 5, learn policies from month 4 predictions; for month 6 use month 4/5 predictions, or month 5 for `latest`. Keep the existing `pooled`, `latest` and learned `rank_fraction` implementations and their exact boundary/tie rules. The rank fraction is learned by maximum F2 and is not a fixed 1% quota. No current labels enter policy fitting. The raw-safe metric implementation handles finite negative/positive logit scores without treating them as probabilities.

Selection uses macro month-5/6 actual F2, then recall, precision and deterministic candidate/policy ordering. F2 remains `5TP/(5TP+4FN+FP)`. Export precision, recall, workload, TP/FP/FN/TN, F1/F3, AP and ROC AUC alongside it. Macro means and pooled-confusion metrics answer different questions; do not replace the objective with AP or pooled F2 after observing results.

Expected evidence is 21 base prediction results, 162 derived score records, 183 total strategy/month results, 366 forward operating rows, 183 macro summaries, 183 pooled summaries and 183 score-diagnostic rows. Export 18 clipping rows: six logit sources times three months, with input min/max, counts below epsilon and above 1-epsilon, actual changed-score count and epsilon. Boundary equality is not counted as changed clipping.

Export exactly 866 contrasts, retaining self/zero comparisons:

| Comparison | Rows |
| --- | ---: |
| Every strategy/policy minus both old pooled references | 366 |
| Each non-pooled policy minus its own pooled policy | 122 |
| Logit single minus corresponding raw single, matched policy | 18 |
| Each fusion minus both members in the same representation and policy | 288 |
| Logit fusion minus raw fusion, matched members/alpha/policy | 72 |

Independent checks reconstruct every decision/confusion/F2 row and each named contrast. Match factors explicitly; a logit fusion member comparison must use its logit single, not its raw single. Report all strategies, not only favorable ones.

### 5. Freeze deterministic transfer roles before month-7 access

Always freeze `selected`, `balanced` (raw balanced/pooled), `unweighted` (raw unweighted/pooled) and `same_model` (selected strategy/pooled). Add controls according to the selected strategy type, all under the selected policy:

| Winner type | Additional roles |
| --- | --- |
| Raw fusion | `xgb_member` and `anchor_member`, both raw singles |
| Logit fusion | `xgb_member` and `anchor_member`, both logit singles; `raw_xgb` and `raw_anchor`, both raw singles |
| Logit single | `raw_member` |
| Raw single with logit counterpart | `logit_member` |
| Raw native31 | None |

This yields four to eight role rows. Preserve every role even when strategies/rules coincide; deduplicate base fits. All final policies learn only from eligible month-4/5/6 predictions (latest uses 6). Freeze choices, strategy definitions, member dependencies, thresholds/fractions, all development table hashes and source/config/software identities before further month-7 access.

Only base dependencies of those roles may be scored on month 7. At most one selected XGBoost final model is needed, plus the two baseline references (or native31 as a third reference if it wins). Reuse complete verified final caches; reproduce wholly absent ones only in the new destination and count them separately. The ninth study had no final XGBoost fit, so an XGBoost-dependent winner may need one. Do not evaluate losing fusions or a hindsight month-7 threshold. Preserve the freeze/access chronology and reject late choices or changed caches.

### 6. Conditional uncertainty and improvement claims

Use 500 paired fixed-decision row-bootstrap replicates, seed 1027, against every frozen reference/control role. Development draws resample each of months 5 and 6 separately with shared indices across roles, then average monthly F2 differences. Consumed-period draws pair the same application indices across fixed role decisions. Do not refit, reselect, recompute rank-policy decisions or reoptimize thresholds within draws. Independently verify pairing, unequal-month macro aggregation and exported quantiles.

These intervals condition on fitted models, chosen strategies and realized decisions. They exclude adaptive search/selection, retraining, temporal/generative uncertainty and multiplicity correction. Repeated development reuse remains a substantial limitation. A positive point difference is a measured gain only on its named scope; an interval spanning zero precludes a clear/confirmed gain claim without becoming a different performance objective. Keep comparisons against both original pooled baselines visible. A development gain that fails transfer is not out-of-time improvement; finishing the grid alone leaves the performance goal unachieved.

### 7. Additive implementation, exports and completion evidence

Add `baf/score_fusion.py` for pure transforms/grid/role rules, `baf/score_fusion_study.py` for guards, verified reuse, staged execution and freeze/access records, and `baf/score_fusion_report.py` for checked tables/uncertainty/figures/findings. Add `configs/xgboost-score-fusion.json`, its protocol and focused tests. Do not edit any frozen source, config or requirements file. The runner provides sequential `validate`, `freeze`, `exploratory` stages, followed by reporting. Complete caches resume by exact identity; selections cannot be overwritten.

Use ignored `outputs/baf-xgboost-score-fusion-v1/` for bulk artifacts and `reports/baf-xgboost-score-fusion-v1/` for checked small exports. Export the full grid, 866 contrasts, clipping and score diagnostics, origin/base/derived provenance, fit/reuse counts, exact configuration/fingerprint/selection/access records, all paired draws/intervals and findings. Inspect a figure showing the operating trade-off and matched raw/logit F2 differences for legibility and clipping.

An independent audit must reconstruct saved scores, policy decisions, metrics, grid counts, contrast factors, selection ordering, frozen role inclusion, model save/load consistency and interval quantiles. README will include pinned installation, verified Base acquisition/checksum, nine selection-restoration commands, absent-cache reproduction, sequential execution, seeds, destinations and consumed-holdout limitations. Update AGENTS and OpenSpec context to verified state only. Acknowledge Codex's substantive planning/code/analysis and the official mathematical/package sources; no public notebook is copied. Run methodological tests, compileall and strict OpenSpec validation, then sync implemented requirements and archive only after actual tasks and outputs are verified.

## Risks / Trade-offs

- Different training weights distort member score scales → retain both anchors and matched controls, expose scale/clipping diagnostics and avoid calibrated-probability claims.
- Strongly correlated models may offer no useful complementarity → report the complete grid and negative results without expanding it after outcomes.
- Logit clipping can create ties and alter threshold transfer → fix epsilon before evaluation, test boundary behavior and expose every clipping count.
- Many prior adaptive experiments can select noise → keep conditional uncertainty honest, preserve all histories and disclose consumed-period failures.
- Reuse can conceal stale or partial artifacts → require exact source/software/population/model provenance and independently reload/reconstruct.
- Fresh clones lack ignored caches → restore small selection identities and reproduce only required exact recipes in new destinations.

## Migration Plan

All implementation is additive and local. After applying this plan, freeze its configuration and identity, pass methodological/reuse gates, execute development, freeze selected roles, run limited exploratory transfer, report and audit. Update current documentation only to reflect verified evidence, then synchronize and archive the change. An implementation failure before results can be repaired without modifying older identities; changed modelling choices after results require another study. Removing new unconsumed artifacts cannot restore month 7 as an independent holdout.
