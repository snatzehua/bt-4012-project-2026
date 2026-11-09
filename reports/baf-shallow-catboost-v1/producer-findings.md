# Shallow CatBoost study findings

Selected **bagged_anchor_balanced_logit_a25 / pooled**, macro chronological F2 **0.330880962**. Measured improvement against all six fixed references in development and consumed transfer: **False**. Completing this finite search does not itself achieve the broader performance goal.

Twelve native recipes cross depth 3/4/6, 200/400 rounds and fixed positive cost 1/100. Thirty-six fixed raw-margin blends and six references yield 54 strategies and 162 policy cases. Positive-class F2 uses beta=2 without a review quota or precision floor. All settings are adaptive analyst selections, open to group review.

All 29 raw predictors use training-only numerical medians and native categorical inputs. RawFormulaVal scores and fixed blends are finite margins, not calibrated fraud probabilities. Original references retain their frozen representations. Depth/round comparisons do not equalize runtime, storage or categorical-statistic work.

Models for months 4/5/6 use strictly earlier training rows. Month-5 rules use month 4; month-6 rules use months 4/5 or latest 5. Selection is actual macro F2 across months 5/6. Transfer roles and exact dependencies were frozen before further month-7 access. Month 7 is an already-consumed exploratory evaluation, not independent confirmation.

| Fixed reference | Macro F2 difference | Consumed F2 difference |
| --- | ---: | ---: |
| bagged_anchor | +0.000000000 | +0.000000000 |
| latest_fusion | +0.000087708 | +0.000705952 |
| recent_fusion | +0.001871017 | +0.001992179 |
| fusion | +0.001974392 | -0.000454375 |
| balanced | +0.003079384 | +0.007831191 |
| unweighted | +0.008227277 | -0.004644674 |

All operating results, matched effects including negative values and ties, score/diversity/overlap diagnostics, requested/effective parameters, preprocessing and actual model work are exported beside this report. Current/historical methodological test calls, fourteen production miniature fits, full native fits, exact reference reproductions and verified reuses are separately recorded in their ledgers.

The 1500 paired row-bootstrap draws keep policies and decisions fixed. Intervals are conditional on the observed synthetic cases and exclude retraining, adaptive recipe/source/policy/selection search, temporal/generative variation and multiple-comparison uncertainty. Clear-gain claims require positive lower bounds in the named scope. Repeated consumed-month access further limits generalization claims.

Codex assisted with design, implementation, tests and interpretation. Primary documentation: [CatBoost raw predictions](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict), [training parameters](https://catboost.ai/docs/en/references/training-parameters/common), [margin/probability output](https://catboost.ai/docs/en/concepts/output-data_model-value-output), and [native serialization](https://catboost.ai/docs/en/concepts/python-reference_catboostranker_save_model). No public notebook was copied.
