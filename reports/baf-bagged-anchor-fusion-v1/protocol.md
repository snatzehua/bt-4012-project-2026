# Fixed bagged-ensemble additions to the preserved F2 anchor

This separate exploratory study implements `test-bagged-anchor-fusion-f2` under `configs/bagged-anchor-fusion.json`. The revised proposal motivates the model and class-imbalance comparisons; lecturer feedback motivates explicit preprocessing and matched controls. The design is analyst-selected, adaptive to prior development results, and open to group review. Completing this finite experiment does not establish a performance gain.

The objective is positive-class F2, beta=2: `5TP/(5TP+4FN+FP)`. This gives missed fraud four times the denominator penalty of false alarms. It is not class-support-weighted F1. There is **no fixed review quota or precision floor**. All reports must retain precision, recall, review load, false alarms and missed fraud alongside F2.

## Data and immutable ancestry

Use BAF Base Kaggle version 2 at ignored `data/raw/baf/Base.csv`, SHA256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. The same Python 3.12.9 and 27 pins in the three requirements files apply. No dependency changes are declared. Source descriptions and limitations remain those of `docs/experiment-protocol.md` and the source-model protocols.

Models predicting development months 4/5/6 use months 0 through 3/4/5 respectively. Preprocessing uses only that model's eligible training population. Month 7 has already been consumed: any further scoring is exploratory and cannot restore an independent holdout. Models transferred to month 7 use months 0-6.

The three new modules are `baf/bagged_anchor_scores.py`, `baf/bagged_anchor_study.py` and `baf/bagged_anchor_report.py`, extending the frozen parent's 52 paths to 55. Destinations are `outputs/baf-bagged-anchor-fusion-v1/` and `reports/baf-bagged-anchor-fusion-v1/`. Preserve the exact 735-path mandatory union, sixteen original/exported selection pairs and all 3732 historical ignored files identified by prospective planning evidence. Separate Newton fit and analysis histories and the stopped L-BFGS attempt remain immutable.

The new destination is inspected first, then the current bagged-study parent. Only old reference bases may inspect their individually declared older origins, in the preserved order, after whole absence. Partial or incompatible inspected caches fail immediately without overwrite or fallback. Wholly absent required sources may reproduce only their exact individual recipes in the new destination under unchanged numerical, resource, warning and persistence gates. Never rerun an ancestor grid or inspect unrelated recipes.

## Exact sources and preprocessing

The five reference bases are `xgb_d4_w1`, `baseline_balanced`, `baseline_lgbm`, `spline_c0p1_unweighted` and `spline_c1p0_balanced`. Their original source/config/software/training/preprocessing/numerical identities remain fixed. The existing anchor's name does not require an interaction-feature model.

The six bagged bases cross balanced/unweighted training with bagging seeds 17/29/43. Retain every eligible training row, original LightGBM preprocessing, 200 rounds, 15 leaves, learning rate .05, minimum child size 100, model seed 2026, six threads, deterministic CPU column-wise fitting, fraction .8 and frequency 1. Public aliases are `subsample=.8`, `subsample_freq=1`, `bagging_seed=<seed>` without conflicting aliases. Balanced class weights use whole-training class counts; individual internal bags are not exactly class matched.

Original LightGBM numeric inputs retain training-only median imputation and missingness indicators; five categorical fields retain training-only one-hot encoding with unknown-category handling. Numeric sentinels follow `features` in the original study; month, target and constant device-fraud count are excluded. XGBoost retains its original native input adapter. Newton spline models retain their original knots, scaling, category handling, solver and independently checked objective/gradient. Independently reconstruct actual training statistics, categories, knots, feature roles, parameters, row identities and class weights; saved inference must match within absolute 1e-12, relative zero.

There are eleven bases per month and 33 development base results. All are available in this checkout, so expected new full development fits are zero. A clean absence can require eighteen bagged and fifteen old-reference fits. Actual per-type fits, reuses, origins, elapsed time and RSS must be recorded rather than inferred from declaration counts.

## Fixed score arithmetic

Let `T = .5*logit(xgb_d4_w1) + .5*logit(baseline_balanced)` and preserve the existing anchor `H = .1*logit(spline_c1p0_balanced) + .9*T`. H's canonical ID is `interaction_blend_spline_c1p0_balanced_a10`. H is a finite margin, never a probability input to another logit.

For each weighting, preserve seed order 17/29/43 and float64 arithmetic:

- Probability mean `P=(p17+p29+p43)/3`.
- Mean-logit margin `L=(logit(p17)+logit(p29)+logit(p43))/3`.
- `logit(p)=log(p)-log1p(-p)` after clipping probability-like inputs to `[1e-6,1-1e-6]`.
- Add either `Gprob=logit(P)` or `Glogit=L` with `S=(1-alpha)*H+alpha*G`, alpha in .1/.25/.5.

The twelve blend IDs are `bagged_anchor_<balanced|unweighted>_<prob|logit>_a10/a25/a50`. Retain the five original references, six native seeds, four parent aggregates and ten canonical logit controls: six seed logits, two probability-mean logits and the two unbagged baseline logits. This is **37 selectable strategies**. Do not logit L. Lambda zero and one on figures use the existing H and G IDs; neither creates an extra evaluated strategy.

Exactly declared unique members must have identical row/label alignment. Reject invalid probabilities, nonfinite margins, incorrect dimensions, undeclared weights and changed arithmetic. Composition uses no current labels, batch statistics, ranks, normalization or learned weights; unrelated rows cannot affect a row's score. Margins are not calibrated probabilities or operational costs. Clipping can create equivalences and ties.

Export 60 diagnostic expression-leaf rows: each of six Glogit blends has three anchor source logits plus three seed logits; each of six Gprob blends has three anchor source logits plus a nonlinear logit-probability-mean atom. Expansions must agree within 1e-12 but **must not replace the declared evaluation association**.

## Policies, complete grid and comparisons

For month 5, fit rules using month 4 predictions/labels only. For month 6, pooled/rank-fraction use months 4/5; latest uses month 5. Final rules use development months 4/5/6 or latest month 6. All compared models predict identical future applications. Select actual macro month-5/6 F2, then recall, precision and deterministic candidate/policy IDs. AP and pooled F2 do not choose a different winner. Preserve largest-threshold, smallest-count, stable-row-ID and exact rational-ceiling tie rules.

The fixed reference policies and macro development F2 values are:

| Role | Candidate | Policy | F2 |
| --- | --- | --- | ---: |
| latest_fusion | interaction_blend_spline_c1p0_balanced_a10 | latest | .3307932536724773 |
| recent_fusion | margin_blend_spline_c0p1_unweighted_a25 | latest | .3290099447918403 |
| fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | .3289065701775199 |
| balanced | baseline_balanced | pooled | .32780157809618937 |
| unweighted | baseline_lgbm | pooled | .3226536849212962 |

Each month has 40 physical scores: eleven bases, seven preserved derived scores, ten new logits and twelve blends. Development therefore has 120 physical / 111 selectable scores: 33 bases, 21 preserved derived results and 66 new calculations. Three policies yield 222 forward operating rows, 111 macro summaries, 111 pooled-confusion summaries and 111 monthly-score summaries.

Retain all 947 contrast identities: 555 versus fixed references; 74 nonpooled versus own pooled; 36 blend versus anchor; 36 blend versus added margin; 36 higher versus lower alpha (all three pairs); 18 blend balanced versus unweighted; 18 blend Glogit versus Gprob; 30 logit versus native; six Glogit versus Gprob controls; fifteen native matched-weighting; fifteen logit matched-weighting including baselines; eighteen probability mean versus native seed; eighteen mean-logit versus logit seed; eighteen logit-probability-mean versus logit seed; 54 randomized versus matched-representation unbagged controls. Keep ties, aliases and negative effects.

Six diagnostic pairs are H versus each of four G margins and Gprob versus Glogit within each weighting. Export eighteen development diversity rows and 36 month-5/6 decision-overlap rows across policies, including caught/missed fraud and legitimate applications. Constant correlations need null values and a reason. Export 39 clipping rows (eleven bases plus two probability means per month), eighteen bagged preprocessing/parameter blocks, fifteen old-reference numerical/preprocessing blocks, all 33 source audit records and the 60 expression rows. Diagnostics cannot change the finite search.

## Gates, freeze and bounded transfer

Before full reproduction, pass the full methodological suite and an actual-host predictor-only allocation preflight for months 0-6, unchanged 70%-RAM allowance and 8192-row verification chunks. Verify the preserved nine-fit bagging miniature and older source ancestry. Present optional old miniature files must match; whole absence must not trigger historical fitting. Zero new production miniature fits are declared.

Each full current-study test invocation must include six genuine tiny source-cache fits: two bagged weightings, balanced/unweighted Newton spline, XGBoost and original LightGBM. Exercise absence, persistence, independent auditing, reuse without fitting, partial/corrupt rejection and failure retention. Count historical-module fits and retries separately. Verify all 111 winner/policy cases and the entire analytical grid with meaningful key/value/provenance corruption fixtures. Full-data audits must reconstruct actual sources and cannot be replaced by dummy metadata tests.

Before further month-7 access, independently audit every score, policy, mask, metric, factor and actual macro winner; freeze source/config/software/protocol identity, development hashes, roles/rules, exact DAG and cache/type-specific fit bounds. Original references have winner precedence. Keep aliases visible while deduplicating work.

| Winner type | Cases | Visible roles | Maximum bases | Maximum scores | Maximum absent bagged fits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original reference | 15 | 7 | 5 | 8 | 0 |
| Baseline logit | 6 | 9 | 5 | 10 | 0 |
| Bagged native/logit seed | 36 | 21 | 9 | 20 | 4 |
| Bagged aggregate/logit | 18 | 21 | 11 | 23 | 6 |
| Blend | 36 | 26 | 11 | 28 | 6 |

All exact role mappings and tighter DAG bounds are in planning evidence. Always retain selected, five references and same-strategy pooled. Baseline-logit winners add native and opposite-weight logit. Other winners retain three native/logit seed pairs, three aggregate representations and two unbagged representations. Single/aggregate winners additionally retain native/transformed/opposite-weight controls. Blend winners additionally retain anchor, native and margin added member, matched weighting/form and all three alpha controls. Only those roles transfer. No losing-grid sweep, hindsight cutoff or winner replacement is allowed.

Current final caches have all five old references and no bagged bases. Exact frozen bounds allow only required absent bagged sources; fresh ancestry absence may separately require two additive and three old-tree reproductions. Maxima do not authorize unnecessary fits.

## Uncertainty, findings and completion

Use seed 1027 and 500 paired fixed-decision draws for each of months 5/6/7: 1000 development and 500 consumed. Seven/nine/21/26 roles yield eighteen/24/60/75 development intervals and six/eight/twenty/25 consumed intervals. Independently reconstruct every draw/quantile and retain aliases. Conditional row intervals exclude training-seed/retraining, adaptive source/search/selection, temporal/generative uncertainty and multiplicity correction.

Findings retain all five reference comparisons and all matched effects, real fit/reuse/gate/failure counts, precision/recall/workload and uncertainty limitations. Figures include all 111 operating points and matched alpha curves with canonical endpoints; verify values, legends, regeneration and visual rendering.

Broader goal achievement requires positive measured F2 differences versus **all five references in both development and consumed transfer**. Clear-gain claims additionally require positive conditional lower bounds on the named scope. A negative finite experiment leaves the goal active. No consumed result is independent confirmation.

Reproduction must verify data/pins, sixteen exact selection restores rejecting changed records, absent individual source preparation and ordered tests/receipt, resource/source-ancestry gates, development preparation/validation/independent audit, freeze, bounded transfer and reporting. Check commands in an isolated no-prior-model/prediction copy. Preserve execution/test/access identities and closed-log run/report hashes. Synchronize/archive only after all fourteen tasks and seven requirements have actual verified outputs, compile/tests and strict OpenSpec validation pass.

Codex assisted with prospective design, isolated implementation, verification and interpretation. Reused APIs include NumPy, pandas, SciPy, scikit-learn, LightGBM, XGBoost, joblib and threadpoolctl; source attribution and versions remain in the prior protocols and pinned requirements. The group must review and explain the submitted work.
