# Conditional LightGBM expert study

Selected **bagged_anchor_balanced_logit_a25/pooled**, macro chronological development F2 **0.330880962**.
Consumed month-7 F2 is **0.372239748**. Positive-class F2 beta=2 has no review quota or precision floor.

Measured improvements against all six references on both scopes: **False**. Finite study completion alone does not establish the broader objective. Clear gains additionally require positive conditional interval lower bounds in the named scope.

## Required comparisons

| reference | development_difference | consumed_difference | development_pass | consumed_pass |
| --- | --- | --- | --- | --- |
| bagged_anchor | 0.000000 | 0.000000 | False | False |
| latest_fusion | 0.000088 | 0.000706 | True | True |
| recent_fusion | 0.001871 | 0.001992 | True | True |
| fusion | 0.001974 | -0.000454 | True | False |
| balanced | 0.003079 | 0.007831 | True | True |
| unweighted | 0.008227 | -0.004645 | True | False |

## Operating results

| candidate | policy | f2 | f1 | f3 | precision | recall | flagged_fraction | average_precision | tp | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bagged_anchor_balanced_logit_a25 | pooled | 0.330881 | 0.237872 | 0.381426 | 0.162720 | 0.451466 | 0.035935 | 0.189225 | 646.500000 | 3395.000000 | 784.000000 |
| interaction_blend_spline_c1p0_balanced_a10 | latest | 0.330793 | 0.234529 | 0.384271 | 0.158659 | 0.459837 | 0.037598 | 0.189693 | 658.500000 | 3569.000000 | 772.000000 |
| interaction_blend_spline_c1p0_balanced_a10 | pooled | 0.330716 | 0.234546 | 0.384105 | 0.158702 | 0.459493 | 0.037552 | 0.189693 | 658.000000 | 3564.500000 | 772.500000 |
| bagged_anchor_balanced_logit_a25 | latest | 0.330347 | 0.237257 | 0.381050 | 0.162199 | 0.451466 | 0.036092 | 0.189225 | 646.500000 | 3412.000000 | 784.000000 |
| margin_blend_spline_c0p1_unweighted_a25 | latest | 0.329010 | 0.231593 | 0.383368 | 0.155537 | 0.460249 | 0.038074 | 0.190366 | 659.000000 | 3628.500000 | 771.500000 |
| logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | 0.328907 | 0.240464 | 0.375839 | 0.166891 | 0.439609 | 0.034181 | 0.188626 | 629.500000 | 3214.000000 | 801.000000 |
| margin_blend_spline_c0p1_unweighted_a25 | pooled | 0.328205 | 0.230378 | 0.383125 | 0.154440 | 0.461284 | 0.038527 | 0.190366 | 660.500000 | 3676.000000 | 770.000000 |
| baseline_balanced | pooled | 0.327802 | 0.240101 | 0.373971 | 0.166693 | 0.436170 | 0.033753 | 0.183160 | 624.500000 | 3175.000000 | 806.000000 |
| logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | 0.327582 | 0.233693 | 0.380469 | 0.159761 | 0.456850 | 0.038092 | 0.188626 | 654.500000 | 3612.000000 | 776.000000 |
| baseline_balanced | latest | 0.327170 | 0.240096 | 0.372824 | 0.166913 | 0.434101 | 0.033512 | 0.183160 | 621.500000 | 3152.000000 | 809.000000 |
| expert_blend_housing_status_balanced_a25 | latest | 0.324795 | 0.234192 | 0.373069 | 0.160013 | 0.438439 | 0.034835 | 0.183400 | 627.500000 | 3308.000000 | 803.000000 |
| expert_blend_payment_type_balanced_a25 | pooled | 0.324398 | 0.233289 | 0.373880 | 0.159624 | 0.442405 | 0.035907 | 0.183083 | 633.500000 | 3405.000000 | 797.000000 |
| capacity400_balanced | rank_fraction | 0.324248 | 0.227054 | 0.378857 | 0.151821 | 0.456519 | 0.038002 | 0.184921 | 652.500000 | 3686.000000 | 778.000000 |
| logit_mean_xgb_d4_w1_baseline_balanced_a50 | rank_fraction | 0.323872 | 0.229227 | 0.376356 | 0.154697 | 0.450255 | 0.036859 | 0.188626 | 643.500000 | 3568.000000 | 787.000000 |
| capacity400_balanced | pooled | 0.323860 | 0.234671 | 0.372087 | 0.161859 | 0.438805 | 0.035344 | 0.184921 | 628.500000 | 3341.500000 | 802.000000 |

| role | candidate | policy | eval_month | parameters | precision | recall | f1 | f2 | mcc | balanced_accuracy | fpr | accuracy | flagged | tp | fp | fn | tn | f3 | flagged_fraction | n | fraud | average_precision | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected | bagged_anchor_balanced_logit_a25 | pooled | 7 | {"beta": 2, "kind": "pooled", "source_months": [4, 5, 6], "threshold": 0.12296562209473805} | 0.186414 | 0.495798 | 0.270953 | 0.372240 | 0.287751 | 0.731707 | 0.032385 | 0.960658 | 3798 | 708 | 3090 | 720 | 92325 | 0.425225 | 0.039218 | 96843 | 1428 | 0.223096 | 0.898914 |
| bagged_anchor | bagged_anchor_balanced_logit_a25 | pooled | 7 | {"beta": 2, "kind": "pooled", "source_months": [4, 5, 6], "threshold": 0.12296562209473805} | 0.186414 | 0.495798 | 0.270953 | 0.372240 | 0.287751 | 0.731707 | 0.032385 | 0.960658 | 3798 | 708 | 3090 | 720 | 92325 | 0.425225 | 0.039218 | 96843 | 1428 | 0.223096 | 0.898914 |
| latest_fusion | interaction_blend_spline_c1p0_balanced_a10 | latest | 7 | {"beta": 2, "kind": "latest", "source_months": [6], "threshold": -0.31412114828661464} | 0.194886 | 0.480392 | 0.277284 | 0.371534 | 0.290259 | 0.725345 | 0.029702 | 0.963074 | 3520 | 686 | 2834 | 742 | 92581 | 0.419008 | 0.036347 | 96843 | 1428 | 0.223937 | 0.899297 |
| recent_fusion | margin_blend_spline_c0p1_unweighted_a25 | latest | 7 | {"beta": 2, "kind": "latest", "source_months": [6], "threshold": -1.1092238992049697} | 0.200839 | 0.469188 | 0.281276 | 0.370248 | 0.291621 | 0.720623 | 0.027941 | 0.964644 | 3336 | 670 | 2666 | 758 | 92749 | 0.413887 | 0.034448 | 96843 | 1428 | 0.224617 | 0.899573 |
| fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | 7 | {"beta": 2, "kind": "pooled", "source_months": [4, 5, 6], "threshold": -0.5732084408730248} | 0.192414 | 0.486695 | 0.275794 | 0.372694 | 0.290135 | 0.728061 | 0.030572 | 0.962310 | 3612 | 695 | 2917 | 733 | 92498 | 0.422133 | 0.037297 | 96843 | 1428 | 0.223147 | 0.898990 |
| balanced | baseline_balanced | pooled | 7 | {"beta": 2, "kind": "pooled", "source_months": [4, 5, 6], "threshold": 0.8395293330571296} | 0.181747 | 0.486695 | 0.264661 | 0.364409 | 0.280924 | 0.726951 | 0.032794 | 0.960121 | 3824 | 695 | 3129 | 733 | 92286 | 0.416767 | 0.039487 | 96843 | 1428 | 0.212249 | 0.896604 |
| unweighted | baseline_lgbm | pooled | 7 | {"beta": 2, "kind": "pooled", "source_months": [4, 5, 6], "threshold": 0.046095701655237555} | 0.168248 | 0.546218 | 0.257256 | 0.376884 | 0.285563 | 0.752903 | 0.040413 | 0.953492 | 4636 | 780 | 3856 | 648 | 91559 | 0.446020 | 0.047871 | 96843 | 1428 | 0.218553 | 0.897189 |
| same_strategy | bagged_anchor_balanced_logit_a25 | pooled | 7 | {"beta": 2, "kind": "pooled", "source_months": [4, 5, 6], "threshold": 0.12296562209473805} | 0.186414 | 0.495798 | 0.270953 | 0.372240 | 0.287751 | 0.731707 | 0.032385 | 0.960658 | 3798 | 708 | 3090 | 720 | 92325 | 0.425225 | 0.039218 | 96843 | 1428 | 0.223096 | 0.898914 |

## Matched effects

| contrast | rows | minimum | maximum | positive | negative | tied |
| --- | --- | --- | --- | --- | --- | --- |
| blend_vs200 | 36 | -0.015567 | 0.003563 | 4 | 32 | 0 |
| blend_vs400 | 36 | -0.010600 | 0.008161 | 16 | 20 | 0 |
| blend_vspure | 36 | -0.000083 | 0.014388 | 35 | 1 | 0 |
| capacity400_vs200 | 6 | -0.010535 | 0.005477 | 1 | 5 | 0 |
| expert_vs200 | 12 | -0.015572 | -0.002456 | 0 | 12 | 0 |
| expert_vs400 | 12 | -0.013509 | 0.000361 | 1 | 11 | 0 |
| fixed_reference | 432 | -0.024933 | 0.008227 | 36 | 390 | 6 |
| higher_alpha | 72 | -0.014388 | 0.002706 | 8 | 64 | 0 |
| matched_partition | 24 | -0.003692 | 0.009874 | 12 | 12 | 0 |
| matched_weighting | 24 | -0.008405 | 0.006704 | 17 | 7 | 0 |
| own_policy | 48 | -0.009213 | 0.002494 | 11 | 36 | 1 |

All negative effects and ties are retained. Raw group counts, routing and errors are exported in eligibility.csv, routing.csv and stratum_errors.csv; these diagnostics did not select new subgroups or thresholds. Global400 controls change nominal round capacity only. They do not equalize runtime, stored trees or histogram costs. Partitioning changes population and total capacity together, so effects do not isolate a pure causal population benefit.

## Actual work

Development audited 84 model results, including 24 original-base reuses, 54 local fits and 6 global400 fits. Exact absent-source reproduction: bagged 0, spline 0, other trees 0. New conditional training row-rounds 3227882000, stored trees 13200, original fit seconds 116.595. Sixty new-method fits are separate from sixteen production miniature fits and current/historical methodological test calls.

Shared pure/blend computation uses 568014800 actual prediction row-rounds across the declared conditional methods; per-bucket global/local work is in conditional_inference_work.csv. Shared source scores are not charged again to each blend. Original fit times and process high-water RSS are measured, not isolated incremental resource costs.

Transfer retained 8 visible roles, 8 model results, 13 full scores and 0 local subset artifacts under frozen bounds. No losing grid was evaluated on month 7. See preparation.json, transfer_complete.json and model provenance for deduplicated work and aliases.

## Conditional uncertainty and limitations

All 1500 paired fixed-decision row-bootstrap draws were independently reconstructed. Intervals omit retraining, adaptive source/partition/policy/selection search, temporal and generative variation, and multiplicity. Seventeen earlier studies informed this exploratory search; partition choices are adaptive analyst choices. Month 7 was already consumed and supplies no independent confirmation. Synthetic BAF and two development evaluation months limit generalization. Settings are analyst-selected, not group decisions.

Codex assisted with design, implementation, experiments, audits and this report. Original source recipes and official API references are retained in the protocol; no public notebook was copied. Preserve frozen execution/access identities and all predecessor studies. Further tuning requires a separate study.
