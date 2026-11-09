# Bagged LightGBM seed study protocol v1

Declared before any new model fit or score evaluation on 12 October 2026. Configuration: configs/bagged-lightgbm-seeds.json. Analyst-selected settings are open to group review.

## Context

See proposal.md for motivation. Base v2 has SHA256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`; Python is 3.12.9 with the unchanged three pinned requirements files. Fifteen finite studies are complete; one L-BFGS attempt remains stopped. Month 7 is already consumed. The latest fixed-addition study found no new gain and preserved its 49-source identity, original 922-test gate, later 941-test clean-copy evidence and separate no-fit archive checks. New results must not overwrite these histories.

The current unweighted/balanced LightGBM references use original numeric imputation and one-hot categorical preprocessing, rather than the later native-categorical engineered presets. Both have 200 rounds, learning_rate=.05, 15 leaves and min_child_samples=100, with deterministic CPU column-wise fitting. Original training does not enable row bagging. Changing only an ineffective seed would repeat the same predictions; this experiment changes training rows per boosting iteration explicitly.

## Goals / Non-Goals

**Goals:** Test effective randomized-row training and equal-weight seed aggregation, keeping tree complexity, full eligible training population and preprocessing fixed. Separate aggregation gains from single-seed and training-weighting effects.

**Non-Goals:** Search fractions, seeds, leaves, rounds or features; tune ensemble weights; add old-anchor blends; introduce review quotas/precision floors; optimize AP; select from consumed outcomes; claim causal drift identification or an independent holdout. No new dependency is needed. These are analyst-selected experimental settings, not approved group decisions.

## Decisions

### 1. Isolated identity and individual references

Use the proposal's three new modules, configuration/protocol and separate destinations. The transitive source closure is the previous 49 paths plus the three new modules: **52** distinct paths. Before execution mechanically freeze that closure, config/protocol/software/data, the exact mandatory preservation union and available historical ignored outputs. The mandatory union extends the previous 529 paths by the previous three modules, config, protocol, original selection, final report manifest and all final report files; the checked closed-manifest union contains **634 mandatory paths**. The available historical ignored snapshot contains **3522 files**; both exact hash maps are retained in planning-evidence.json. Verify **fifteen original/exported selection pairs**. Keep separate Newton fit/analysis identities and stopped-attempt snapshots. Tests/documentation can evolve only with separately retained execution sources/receipts.

The five original references are latest_fusion/latest, recent_fusion/latest, tree fusion/pooled, balanced/pooled and unweighted/pooled, retaining canonical IDs and macro F2 values 0.3307932536724773, 0.3290099447918403, 0.3289065701775199, 0.32780157809618937 and 0.3226536849212962. Their dependency set has exactly five bases: xgb_d4_w1, baseline_balanced, baseline_lgbm, spline_c0p1_unweighted and spline_c1p0_balanced; no interaction model is required. For months 4/5/6 this is **15 old base results plus nine old derived anchor scores**. Inspect the new destination, then current-parent caches, then only declared individual older origins in their preserved order. Partial/incompatible inspected sets fail; only wholly absent sets advance. Missing exact references fit in the new destination, never by rerunning an ancestor grid. Independently verify source recipes, eligible rows, preprocessing, saved inference, original provenance and logistic numerical gates. Actual new/reused work is recorded separately.

### 2. Effective row bagging and unchanged preprocessing

Fit each of two class-weighting choices (None and balanced) with **bagging_seed 17/29/43**, bagging_fraction=.8 and bagging_freq=1. Keep random_state=2026 and all remaining original 200-round LightGBM settings unchanged. At the scikit-learn constructor boundary, map fraction/frequency once to its public `subsample=.8` and `subsample_freq=1` parameters; do not also pass conflicting primary aliases through extra kwargs. Audit normalized effective booster parameters as well as the exact requested parameters. Preserve feature_fraction=1, pos_bagging_fraction=neg_bagging_fraction=1 and data_sample_strategy=bagging; do not add class-dependent resampling. Retain original defaults for binning, regularization and other secondary RNGs as resolved by the original random_state. Record full requested and actual booster parameters to verify this, rather than assuming seed alone controls every random component.

Each model for month m uses all rows in months 0..m-1, in stable original order, with `features` sentinel handling and the original `make_model` numeric/categorical pipeline. Numeric train medians/missing indicators and categorical vocabulary/unknown handling fit on training rows only; no scaling or native-category substitution is introduced. Balanced weights use whole-training class counts before internal row sampling. Random bags are uniform subsets without replacement within an iteration and are not guaranteed to preserve exact class proportions; they are not matched-K historical controls or bootstrap samples. Row bagging changes the fitted model; deterministic=True fixes repeatability for the same data/parameters and does not disable declared sampling.

There are **2 weightings × 3 seeds × 3 months = 18 new full development fits**. Do not early-stop or adapt rounds based on current-fold outcomes. Require finite scores in [0,1], checked saved-model round trips (absolute tolerance 1e-12, relative zero), exact requested/actual seed/sampling/preprocessing identities and no unrecorded warnings/fallback. Record actual rounds/trees, fit seconds, peak process RSS and transformed dimensions. Retain failure evidence and stop before selection if a required gate fails; do not relax the recipe.

An actual-host predictor-only resource gate precedes production fitting, covers the largest eligible month-0-6 source/reference preprocessing and unchanged 70%-RAM allowance with chunk bound8192 for reference verification. Preserve source miniature ancestry without refitting old production gates. A separate **nine-fit tiny integration gate** uses nondegenerate synthetic data: all six new weighting/seed combinations, one same-recipe repeat and two unbagged controls differing only in bagging seed. Verify repeatability/persistence, effective new bagging parameters, detectable score diversity on this designed fixture and equality of the two inactive-bagging controls. This does not establish full-data gains. Two genuine tiny new-format fits (one per weighting) per full methodological invocation are counted separately from this gate and all historical module tests. Any retry/version history remains explicit.

Primary API semantics: [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#bagging_fraction), including bagging frequency/seed and deterministic CPU behavior. The .8 fraction and three seeds are fixed analyst choices, not an empirically optimal setting. Feature subsampling, many more seeds or learned weighting are different experiments and excluded.

### 3. Fixed seed aggregates and complete comparisons

IDs are `bagged_<balanced|unweighted>_s17/s29/s43`, `bagged_<weighting>_prob_mean` and `bagged_<weighting>_logit_mean`. Together with five existing reference IDs, there are **15 selectable strategies**. Native seed scores and probability means are probability-like but not asserted calibrated. Probability means sum in fixed seed order using float64, then divide by three. Logit means validate each [0,1] member, clip at epsilon1e-6, transform separately and average the three finite logits in the same order; never logit an existing margin or replace this with logit(mean(p)). All members must have identical row IDs and labels. Reject missing/duplicated seeds, invalid domains, shape/order differences and nonfinite arithmetic. Current labels, unrelated rows and batch statistics do not enter composition.

Development has **54 physical scores = 18 new bases + 15 reference bases + 9 old anchors + 12 new aggregates**, with **45 selectable month/strategy scores**. Three policies pooled/latest/rank_fraction yield **90 forward rows** for months5/6 and **45 each macro, pooled-confusion and monthly-score summaries**. Models use earlier months; month-5 policies use month4, month-6 policies use4/5 or latest5. Policies optimize actual positive-class F2=5TP/(5TP+4FN+FP), with existing largest-threshold/smallest-count/stable-ID ties and rational capacity ceilings. Rank fraction is learned earlier, not a 1% quota. Select macro month-5/6 F2 then recall, precision and deterministic IDs. F1/F3, AP/ROC, workload and confusion counts accompany it.

Independently enumerate and verify **342 contrasts**:

- 225 against five fixed operating references:15 strategies×3 policies×5.
- 30 nonpooled minus own pooled:15×2.
- 30 new randomized/aggregate strategies minus their matched unbagged native weighting at the same policy:10×3.
- 36 aggregate minus each of its three constituents at the same policy:4×3×3.
- Six logit-mean minus probability-mean:2 weightings×3 policies.
- Fifteen balanced-minus-unweighted matched forms:3 seeds+2 aggregations, each×3 policies.

Keep factor identities even when numbers tie. Export **18 seed-score diversity rows** (three unordered seed pairs×two weightings×three development months) and **36 decision-overlap rows** (three pairs×two weightings×three policies×months5/6). Include Pearson/Spearman with explicit null reasons, joint fraud/legitimate caught/missed counts and score spread; diagnostics cannot select members. Export clipping for all **33 probability-like base/month results**, plus actual new-model preprocessing/parameter/resource provenance and reference numerical evidence. Do not fabricate per-tree sampled row identities the library does not expose; the claim concerns configured/verified sampling behavior, not an audited list of every bag.

An independent grid-specific auditor reconstructs all54 score artifacts, training/evaluation IDs, saved inference, both arithmetic forms, earlier-only rules, masks/metrics, summaries,342 factor rows, diagnostics and the actual winner before freeze. Analytical no-fit full-grid and corruption fixtures must check exact key sets, not just counts. Preserve actual negative effects.

### 4. Predetermined bounded transfer

Always include selected, the five fixed references and same_strategy/winner-pooled: **seven roles**. Original reference identities take precedence and stop at seven. For any of the ten new strategies add member_s17, member_s29, member_s43, probability_mean, logit_mean, matched_weight (same seed or aggregation, opposite weighting) and unbagged_member (same weighting at winner policy): **fourteen roles**. Keep visible aliases and deduplicate work.

Enumerate **45 winner/policy cases**:15 reference cases;18 single-seed cases;12 aggregation cases. Reference cases need five old bases/eight scores and zero new bagged fits. Single-seed cases need those five references plus the winner-weighting's three new seeds and the opposite-weighting matched single: at most **nine bases/four new bagged fits/fourteen scores**. Aggregate cases require both weightings' three seeds: at most **eleven bases/six new bagged fits/seventeen scores**. Exact DAGs and existing cache availability yield tighter per-type bounds; fresh missing references can require two additive and three tree reproductions, counted separately. Freeze role mappings, exact source/score dependencies, frozen winner/rules, actual hashes and bounds before any further month-7 access. Final rules use4/5/6 or latest6; only the frozen DAG transfers, without losing-grid sweep or policy replacement.

The objective remains actual F2 beta2 without quota/floor. Broader acceptance requires positive measured differences against **all five references on both scopes**. Clear-gain language additionally needs positive conditional lower bounds on the named scope. Consumed month7 remains descriptive; no search or archive can recover independent confirmation.

### 5. Conditional uncertainty, reporting and reproduction

Use seed1027 and500 paired fixed-decision row draws each for months5/6/7:1000 development and500 consumed rows. Seven roles yield18 development/6 consumed intervals;fourteen roles yield39/13. Retain aliases; independently reconstruct every draw and quantile. Intervals exclude training-seed/retraining, selection, adaptive repeated-search, temporal/generative uncertainty and multiplicity correction. The three actual seeds describe a finite training perturbation, not a population confidence interval over seeds.

Findings cover every fixed-reference/matched effect, actual fits/reuses/gates/failures and precision/recall/workload consequences. Inspect figures with all45 strategy/policy operating points and matched single/aggregate F2 views; validate plotted values and readable legends. README must provide data/checksum/pins, fifteen exact selection restores rejecting changed records, isolated absent-reference reproduction and ordered full tests/receipt -> predictor-only preflight -> tiny integration -> preparation/validation/independent audit -> freeze -> bounded transfer -> report commands. Never overwrite old frozen evidence. Export original test source/receipt, config/protocol/selection/access/provenance, all outputs and closed-log hashes; update current context and AI/API attribution. Complete only after requirement/scenario/task audits, preservation, meaningful tests, compile checks, strict OpenSpec validation and report manifests pass. Further tuning needs another study.

## Risks / Trade-offs

- Bagging may reduce useful rare-fraud signal -> balanced/unweighted and all seed controls, unchanged whole-training rows, retain negative results.
- Three seeds may yield small unstable gains -> fixed seeds, visible constituents/diversity, conditional uncertainty and explicit training-seed exclusions.
- Parameter aliases or ineffective sampling can invalidate the hypothesis -> canonical parameters, actual booster audit and inactive-sampling miniature controls.
- Reference reuse spans source formats -> individual exact-recipe audits and no old-grid dependency.
- Consumed data informed this proposal -> disclose adaptation and prohibit independent-confirmation claims/hindsight selection.

## Migration Plan

Finish prospective declarations/preservation enumeration, implement isolated modules, verify meaningful fixtures and actual resource/tiny gates, then execute the entire chronological finite grid. Independently audit before freeze; transfer only predetermined dependencies, finish checked uncertainty/figures/reproduction, and synchronize/archive after actual completion. Failed gates and all prior studies remain intact. No material design question is deferred.
