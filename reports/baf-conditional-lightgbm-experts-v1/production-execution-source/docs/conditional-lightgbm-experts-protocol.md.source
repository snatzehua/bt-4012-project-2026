# Conditional LightGBM experts: prospective protocol v1

Written 12 October 2026 before any new model or proposed score evaluation. Settings are analyst-selected, not group-approved. The executable configuration is `configs/conditional-lightgbm-experts.json`. Implementation status is tracked separately in OpenSpec tasks; this document is not a completion claim.

## Context

See `proposal.md` for motivation. The seventeenth completed study selected `bagged_anchor_balanced_logit_a25/pooled`, macro development F2 0.3308809618; its consumed-month F2 0.372240 failed the tree-fusion and unweighted comparisons. Its original selection SHA is `ec308166981ad973ab0fa6bee693e21d7bc8fe0e351bed32f4cb8b02eb8b8d0c`. All predecessor sources, software, selections and access histories remain frozen.

`planning-evidence.json` records prospective enumeration, training-only group counts and recomputation of **old** development diagnostics. It contains zero new model fits and zero evaluations of proposed compositions or policies. For the current winner, hindsight-optimal threshold F2 on months 5/6 averages 0.332033 versus achieved 0.330881. Optimal flagged fractions across months 4/5/6 are 0.043300/0.027874/0.043488. This is an optimistic diagnostic for the fixed scores, not a deployable policy or an upper bound on other models. Nonmonotonic fractions give little support to the alternative of extrapolating a threshold/review trend. Descriptive group errors motivate partitioning but cannot establish its benefit. The partitions are adaptive analyst choices after earlier development results.

The planning helper's first attempt used the parent's policy-only mapping where role pairs were needed and stopped before fitting anything. Preserve `planning-initial.py.source` and `planning-initial.log`; the corrected enumeration and its successful log are separate evidence.

## Goals / Non-Goals

**Goals:** Test whether conditional discrimination improves actual positive-class F2, retaining chronological selection, original global preprocessing/class costs, explicit global fallbacks, capacity controls and independent reconstruction. Keep every negative result and the actual cost of fitting and auditing visible.

**Non-Goals:** No new features, learned routing, subgroup-specific thresholds, locally rebalanced costs, calibration, resampling, early stopping, hyperparameter search or restored independent holdout. No 1% quota or precision floor. Partitioning is not a matched-size recency experiment and does not replace the completed matched-K study. No assumption that larger AP or a successful finite run achieves the performance objective.

## Decisions

### 1. Isolated source and preservation contract

Use three new modules: `baf/conditional_experts.py` for models and scores, `baf/conditional_expert_study.py` for execution/auditing, and `baf/conditional_expert_report.py` for uncertainty/reporting. Add `configs/conditional-lightgbm-experts.json`, `docs/conditional-lightgbm-experts-protocol.md`, `tests/test_conditional_experts.py` and a separate fit-accounting runner. Do not change frozen production modules or predecessor tests/runners. New destinations are ignored `outputs/baf-conditional-lightgbm-experts-v1/` and exported `reports/baf-conditional-lightgbm-experts-v1/`.

Extend the parent's 55-source closure to 58 by those three modules. Planning evidence enumerates 1,053 mandatory preservation paths, seventeen original/exported selection pairs and 4,369 historical ignored files. Mandatory source/config/software/protocol/selection/report evidence must survive reproduction; optional ignored caches may be wholly absent. Compare all present historical files against the original snapshot. Preserve separate Newton fit/analysis histories and the stopped L-BFGS attempt. Config, protocol, executable source and software identities freeze before execution; retain subsequent test/reproduction receipts without rewriting fit identities.

Keep six operating references with their original policies:

| Role | Canonical strategy | Policy |
|---|---|---|
| bagged_anchor | bagged_anchor_balanced_logit_a25 | pooled |
| latest_fusion | interaction_blend_spline_c1p0_balanced_a10 | latest |
| recent_fusion | margin_blend_spline_c0p1_unweighted_a25 | latest |
| fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled |
| balanced | baseline_balanced | pooled |
| unweighted | baseline_lgbm | pooled |

Their required DAG is eight bases and thirteen full-population scores per month: three balanced bagged seeds 17/29/43, XGBoost, balanced/unweighted original LightGBM and two additive spline-logistic sources; five derived scores include the older fusions, balanced bagged mean-logit and current anchor. Inspect only required individual caches in declared own/parent/older-reference order. Verify recipe, training IDs, preprocessing, software, warning/numerical validity, hashes and saved inference. Partial, corrupt or incompatible inspected sets fail without silent fallback. Whole absence permits only exact individual reproduction in the new destination under original gates. No call may prepare an unrelated historical search grid. Current-cache development expects 24 base and fifteen derived reuses. Fresh absence can require nine bagged, six spline and nine other tree fits; record them separately from the new method.

### 2. Fixed local populations and shared global representation

Declare canonical `payment_type` levels AA/AB/AC/AD/AE and `housing_status` BA/BB/BC/BD/BE/BF/BG. For each earlier training population and both balanced/unweighted variants, a level receives an expert only with at least **10,000 training rows, 100 fraud and 100 legitimate rows**. Membership uses the original predictor value; missing and unexpected levels cannot acquire an expert from a future batch. Sparse or absent levels use the original same-weight global 200-round model. A failed eligible fit is an error, never a fallback reason.

Planning verifies four eligible payment levels and five eligible housing levels before each of months 4/5/6/7 using months strictly earlier than the named month. AE/BF/BG are ineligible. These are eligibility observations, not model results; executable auditing must reconstruct counts. Retain all twelve canonical level records per weighting/fold, including zero counts.

Reuse the verified global unweighted 200-round preprocessor for that earlier training population; independently confirm the balanced model has the same fitted statistics, categories, feature order and population. Apply it once to global training data and slice rows for local fitting, maintaining the global CSR feature schema. Keep the partition's constant one-hot columns rather than introducing another feature-selection change. Local experts do not refit medians, missing indicators or vocabularies.

Original feature handling is authoritative in `baf/common.py`: exclude `fraud_bool`, `month` and constant `device_fraud_count`; stable IDs are alignment keys, never predictors. Decode -1 in `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes` and `device_distinct_emails_8w` to missing; negative `intended_balcon_amount` is missing. Use earlier global medians, missing indicators with empty features retained, and one-hot encoding with unknown categories ignored for payment/employment/housing/source/device OS. Trees receive no numerical scaling; the preserved logistic references retain their own original scaled/spline preprocessing. Share no fitted objects across different eligible training months.

For each future application, use its partition expert if eligible, otherwise global fallback. Persist expert predictions only for matching future member IDs, and independently stitch exactly one score for every full-population ID. Reject overlaps, omissions, duplicate IDs and mislabeled subset artifacts. Empty inference subsets are valid zero-length artifacts with metadata. Changing current labels, row order or unrelated rows must not change an application's score.

Alternative local preprocessing was rejected because it changes representation along with population specialization. A learned router or chosen subgroup threshold would add another adaptively fitted layer; this study uses fixed observable memberships and one global operating policy.

### 3. Whole-training costs and fixed tree recipes

Retain the original unbagged classifier recipe: 200 rounds, learning rate 0.05, fifteen leaves, minimum child samples 100, seed 2026, six threads, deterministic column-wise fitting and verbosity -1, with all other effective parameters matched to the original factory. No sampling or early stopping. Add same-weight global **400-round** controls, changing only round count.

Compute class costs from the **whole earlier training population**: unweighted costs are one; balanced costs are `N/(2*N0)` and `N/(2*N1)`. For new local and capacity models set `class_weight=None` and pass those exact per-row costs as `sample_weight`. Never recompute balanced costs from local counts or multiply them by another class weight. Persist global counts, weight identities/hashes, member IDs, requested/effective parameters, feature names, fitted rounds, warnings, model hashes and inference reconstruction at absolute tolerance 1e-12, relative tolerance zero.

The [official LightGBM classifier API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) documents inverse-frequency balanced weights and multiplication with sample weights. Miniature gates must establish equivalence between the original global class-weight pipeline and the explicit-global-weight 200-round model before production work. Pin behavior to the installed version in the preserved 27-package environment, rather than assuming current documentation substitutes for executable checks. Scores are probability-like outputs, not established calibrated fraud risks.

Global400 is a nominal work control: global200 plus one local200 prediction has up to 400 rounds per routed row, and training row-round exposures are at most `400*N` when all rows have experts. It does **not** match total stored trees, histogram construction, memory or measured runtime. Record actual row-round exposures, stored tree counts, per-row inference work, wall time and RSS. The expert method changes training populations and total model capacity together; do not claim a pure causal population effect.

### 4. Canonical scores, selection and complete effects

Select from six original references, two capacity400 models, four pure experts (two partitions by two weightings) and twelve blends. A blend is exact aligned float64 arithmetic `(1-alpha)*global200 + alpha*expert`, with alpha 0.25/0.5/0.75. Alpha zero is the original balanced/unweighted reference; alpha one is the declared pure expert. Canonical IDs and all 72 strategy/policy cases are in planning evidence. No extra endpoint evaluation, transform, clipping or batch normalization is introduced. Reject invalid/nonfinite probabilities; existing reference margins retain their original domains.

Each model for month m trains on all months before m. Generate development scores for 4/5/6. Fit `pooled`, `latest` and `rank_fraction` operating policies only on earlier out-of-time predictions: month 5 uses 4; month 6 uses 4/5 or latest 5. Final rules use 4/5/6 or latest 6. Rank fraction is learned to optimize F2, not a fixed quota. Preserve exact rational count/ceiling, stable-ID score ties, largest-threshold/smallest-count policy ties and existing deterministic strategy ordering. Choose maximum actual macro month-5/6 F2, then recall, precision and lexical IDs; AP, pooled results and group errors cannot choose a different winner. F2 is `5*TP/(5*TP+4*FN+FP)`, with explicit zero-denominator handling.

Prospective available-cache accounting:

| Output | Count |
|---|---:|
| New local / capacity full development fits | 54 / 6 |
| Required reference base / derived results | 24 / 15 |
| Full-population physical scores | 93 |
| Separate expert-member subset scores | 54 |
| Eligibility / fallback model records | 72 / 18 |
| Stitched pure / new blend scores | 12 / 36 |
| Selectable monthly scores | 72 |
| Forward operating / macro / pooled rows | 144 / 72 / 72 |
| Unique matched contrasts | 738 |
| Routing / stratum error rows | 96 / 2304 |

The 93 full scores comprise 39 old DAG scores, six capacity scores, twelve stitched scores and 36 blends. Do not count subsets as predictions on comparable populations. Routing diagnostics include every known level and explicit missing/unseen buckets: sixteen buckets across both partitions, two weightings and three months give 96 rows. Error diagnostics use the same sixteen buckets by 24 strategies, three policies and two evaluation months: 2,304 rows, including empty groups.

All 738 contrast identities are enumerated: 432 against six fixed references; 48 nonpooled versus own pooled; six capacity400 versus200; twelve each pure expert versus200 and400; 36 each blend versus200,400 and pure; 72 higher-alpha effects among 0.25/0.5/0.75/1; 24 matched weighting; 24 matched partition. Preserve ties/negative effects and canonical identities. Report F2, precision, recall, flagged fraction, TP/FP/FN/TN, missed fraud, legitimate flags and descriptive AP. Current-month group outcomes are report diagnostics only.

### 5. Gates, independent evidence and actual fit accounting

Verify Base SHA `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`, Python 3.12.9 and all 27 pins. Run the unchanged conservative predictor-only preflight on the actual host for the largest eligible global population (903,157 rows) and at most 157 source features, with 8,192-row chunks and a 70%-RAM ceiling. Preserve and audit exported nine-fit sampling and eight-fit ancestry gates; present optional ignored files must pass whole-set/hash/inference checks, and whole absence does not invalidate exported evidence. No new old-gate fits are planned.

Add a distinct **sixteen-fit production miniature gate**: two original global200 fits, two explicit-weight global200 equivalence fits, two global400 fits, eight local experts (two weightings by two partitions by two eligible groups), and two deterministic repeats (one balanced expert and balanced global400). Synthetic earlier training uses two months of 12,000 rows, two groups per partition each reaching 12,000 rows and the original fraud/legitimate minima; do not lower eligibility thresholds for the gate. A 12,000-row future month includes missing/unseen routing cases and differentiated group prevalences. Verify global weights rather than local balance, shared preprocessing, all routing reasons, probability domains, effective rounds, repeatability, warnings and save/load inference. A numerical/resource/persistence failure retains evidence and stops; never relax the recipe after outcomes.

The new methodological runner instruments actual classifier fit calls and separates this study's tiny tests from historical modules. Plan ten current tiny fits per complete invocation: four new expert/capacity formats and six required old formats (two bagged, two spline, XGBoost, original LightGBM). Audit every cache reuse without refitting. Retain failures/retries/source receipts and compare ledgers with observed calls; never infer historical counts from the expected current count. Tests cover all 72 winner/policy DAGs, missing/partial cache behavior, temporal/label/membership invariance, global costs, score stitching/arithmetic, aliases, tie boundaries and complete factor/grid corruption fixtures. Independent development auditing reconstructs all 84 saved models with current cache availability, all 93 scores and 54 subsets, rules/masks/metrics, complete tables and the actual winner before freeze.

### 6. Predetermined transfer, uncertainty and acceptance

Always freeze selected, six fixed references and same-strategy pooled: eight visible roles with reference precedence. A capacity winner adds same-weight global200 and opposite-weight capacity400 at its policy: ten roles. A pure/blend winner adds nine roles: same-weight global200, global400 and same-partition pure expert; opposite weighting at selected alpha; opposite partition at selected alpha; and alpha0.25/0.5/0.75/1 at selected partition/weight/policy. This gives seventeen roles including visible aliases. Deduplicate actual dependencies, never the display contract.

All 72 cases are enumerated before fitting. Each needs the old eight bases/thirteen scores. A reference winner needs no new model; a capacity winner needs two capacity models and fifteen total full scores. A pure winner needs at most twenty full scores; a blend at most 22. With observed earlier-than-7 eligibility, payment winners need thirteen local plus one capacity fit; housing winners fourteen local plus one capacity. Generic all-level eligibility could require twenty new fits, so freeze **actual** training-only eligibility and cache/type bounds rather than applying the smaller observed bound blindly. Whole-absence reference reproduction adds at most three bagged, two spline and three other tree fits. Current eight final reference bases are present; availability must be audited again at freeze.

Freeze winner, config/software/source/protocol identities, exact role aliases, earlier rules, membership/eligibility, cache availability, per-type work bounds and access record before further month-7 scoring. Transfer only the deduplicated frozen DAG. Do not evaluate losing grids, hindsight month-7 thresholds or replace the winner using transfer outcomes. Month 7 is consumed exploratory transfer and cannot confirm independent generalization.

Use seed 1027 and 500 paired fixed-decision row draws for each of months 5,6,7. Independently reconstruct all 1,500 draws and percentile contrasts: eight/ten/seventeen roles imply 21/27/48 development interval rows and 7/9/16 consumed rows, retaining aliases and fixed role identities. Conditional intervals omit retraining, adaptive source/partition/selection search, temporal/generative variation and multiplicity. They cannot make repeated adaptive development search independent.

Broader acceptance requires positive measured selected-minus-reference F2 against **all six fixed references** in both macro development and consumed transfer. Clear-gain statements additionally require positive conditional interval lower bounds in the named scope. A complete negative finite study leaves the performance objective active. No trade of higher recall alone, AP or another objective can satisfy it.

Figure validation covers 72 operating points, twelve alpha curves (partition by weighting by policy) with five canonical endpoints/weights each (60 rows), and six global400 control lines. Verify plotted values, legends and pixel-identical regeneration, then inspect the rendered image. Findings must explain reference failures, group errors, measured work and conditional uncertainty.

## Risks / Trade-offs

- Adaptive partitions and seventeen previous searches can overfit development evidence → preserve negative results, freeze the finite search, disclose adaptivity and consumed transfer, and avoid independent-confirmation language.
- Smaller subgroup samples may worsen discrimination or stability → retain fixed whole-training costs, original minimum populations, both partitions and global200/global400 controls.
- More models may improve capacity while increasing storage/runtime → measure actual work and describe the composite method; no equal-runtime claim from round counts.
- Corrupt caches or hidden feature differences can produce apparent gains → individually reconstruct provenance, preprocessing, membership and saved inference; fail closed on partial sets.
- Resource or numerical limits may stop the study → run unchanged gates first, retain partial work/failures and require a separate study for a changed recipe.

## Migration Plan

After a separate apply request, create isolated implementation/config/protocol/tests, establish gates and independent audit outputs, then run the declared development study. Freeze before bounded consumed transfer, reconstruct uncertainty and reporting, verify clean-copy reproduction and preservation, and only then synchronize/archive. Shared README/project context must distinguish proposed artifacts from measured completion throughout. A fresh copy restores all seventeen exact selections and rejects changed restores, checks pins/data, and runs the full methodological suite without prior model/prediction caches; this does not claim full-size retraining in the copy. Publish ordered commands for actual individual absent-source reproduction and source gates. Retain closed logs, cumulative fit ledger, original/generated/final report manifests and run hashes. Attribute Codex and official API sources; no public notebook is copied. Rollback means leaving a failed isolated study visibly incomplete without altering predecessors.
