# Spline-score fusion findings

Selected **margin_blend_spline_c0p1_unweighted_a25 / latest**, macro forward F2 **0.329010**. Best blend: **margin_blend_spline_c0p1_unweighted_a25 / latest**, F2 **0.329010**. Positive measured F2 differences against fusion, balanced and unweighted on both development and consumed transfer: **False**. Search completion alone does not achieve the broader performance goal.

Four frozen C=0.1 Newton models (linear/spline crossed with unweighted/balanced training) were combined with the existing fixed tree fusion at Newton weights 0.1/0.25/0.5. Only Newton probability-like scores received a clipped logit, epsilon 1e-6; the existing fusion is already a margin. Composition is fixed row-wise and uses neither labels nor batch statistics. Scores are not asserted calibrated probabilities. C/source choices are adaptive to earlier exploratory results.

Development used **21 verified model/score reuses**, **0 new fits**, 51 derived results and zero miniature model fits. Every saved base model reproduced inference; Newton representation, independent objective/gradient and fallback evidence were checked under the frozen recipes. There are 72 score artifacts, 69 selectable scores, 138 forward operating rows, 69 macro and pooled summaries, 433 contrasts, 12 clipping/diversity rows each and 24 joint-error rows. All eleven completed identities and the stopped attempt remain preserved.

Month-5 policies learn on month 4; month-6 policies on months 4/5 or latest month 5. F2 has beta=2, without a review quota or precision floor. Selection uses actual macro F2, then recall, precision and deterministic IDs. Pooled F2, AP, correlations and joint errors are descriptive.

## Forward results

| candidate | policy | f2 | precision | recall | flagged_fraction | average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| margin_blend_spline_c0p1_unweighted_a25 | latest | 0.329010 | 0.155537 | 0.460249 | 0.038074 | 0.190366 |
| logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | 0.328907 | 0.166891 | 0.439609 | 0.034181 | 0.188626 |
| margin_blend_spline_c0p1_balanced_a10 | latest | 0.328810 | 0.157229 | 0.459771 | 0.038164 | 0.189627 |
| margin_blend_spline_c0p1_balanced_a10 | pooled | 0.328810 | 0.158112 | 0.457012 | 0.037605 | 0.189627 |
| margin_blend_linear_c0p1_balanced_a10 | pooled | 0.328502 | 0.154029 | 0.463381 | 0.038891 | 0.188763 |
| margin_blend_spline_c0p1_unweighted_a25 | pooled | 0.328205 | 0.154440 | 0.461284 | 0.038527 | 0.190366 |
| margin_blend_spline_c0p1_unweighted_a10 | pooled | 0.327892 | 0.161781 | 0.450959 | 0.036747 | 0.189627 |
| baseline_balanced | pooled | 0.327802 | 0.166693 | 0.436170 | 0.033753 | 0.183160 |
| margin_blend_linear_c0p1_unweighted_a25 | latest | 0.327762 | 0.155930 | 0.459138 | 0.038343 | 0.187625 |
| margin_blend_linear_c0p1_balanced_a10 | latest | 0.327758 | 0.159691 | 0.446140 | 0.035591 | 0.188763 |
| margin_blend_linear_c0p1_unweighted_a25 | pooled | 0.327752 | 0.156586 | 0.457069 | 0.037923 | 0.187625 |
| logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | 0.327582 | 0.159761 | 0.456850 | 0.038092 | 0.188626 |
| margin_blend_spline_c0p1_balanced_a25 | pooled | 0.327539 | 0.163489 | 0.448153 | 0.036348 | 0.190397 |
| margin_blend_spline_c0p1_balanced_a25 | latest | 0.327412 | 0.163596 | 0.447464 | 0.036246 | 0.190397 |
| baseline_balanced | latest | 0.327170 | 0.166913 | 0.434101 | 0.033512 | 0.183160 |
| margin_blend_linear_c0p1_unweighted_a10 | latest | 0.326694 | 0.171737 | 0.427425 | 0.032449 | 0.188703 |
| margin_blend_linear_c0p1_unweighted_a10 | pooled | 0.326694 | 0.171737 | 0.427425 | 0.032449 | 0.188703 |
| margin_blend_spline_c0p1_balanced_a25 | rank_fraction | 0.326660 | 0.150220 | 0.464795 | 0.039041 | 0.190397 |
| margin_blend_spline_c0p1_unweighted_a10 | latest | 0.326607 | 0.158884 | 0.456822 | 0.038342 | 0.189627 |
| margin_blend_spline_c0p1_unweighted_a50 | rank_fraction | 0.326600 | 0.154425 | 0.454231 | 0.037080 | 0.189952 |
| margin_blend_spline_c0p1_unweighted_a50 | pooled | 0.326129 | 0.163873 | 0.445654 | 0.036236 | 0.189952 |
| margin_blend_spline_c0p1_unweighted_a10 | rank_fraction | 0.325340 | 0.149571 | 0.462707 | 0.038998 | 0.189627 |
| margin_blend_spline_c0p1_balanced_a50 | rank_fraction | 0.324981 | 0.152286 | 0.454519 | 0.037614 | 0.190067 |
| margin_blend_linear_c0p1_balanced_a25 | pooled | 0.324454 | 0.158904 | 0.446822 | 0.036822 | 0.187570 |
| margin_blend_linear_c0p1_balanced_a25 | latest | 0.324353 | 0.158698 | 0.447166 | 0.036924 | 0.187570 |
| margin_blend_linear_c0p1_unweighted_a10 | rank_fraction | 0.324082 | 0.158210 | 0.443206 | 0.035483 | 0.188703 |
| logit_mean_xgb_d4_w1_baseline_balanced_a50 | rank_fraction | 0.323872 | 0.154697 | 0.450255 | 0.036859 | 0.188626 |
| margin_blend_linear_c0p1_unweighted_a25 | rank_fraction | 0.323804 | 0.142788 | 0.478472 | 0.042463 | 0.187625 |
| margin_blend_spline_c0p1_unweighted_a50 | latest | 0.323528 | 0.158921 | 0.455309 | 0.039051 | 0.189952 |
| margin_blend_spline_c0p1_balanced_a50 | pooled | 0.323473 | 0.166584 | 0.438921 | 0.035673 | 0.190067 |
| margin_blend_spline_c0p1_unweighted_a25 | rank_fraction | 0.323057 | 0.145341 | 0.471623 | 0.041219 | 0.190366 |
| baseline_lgbm | pooled | 0.322654 | 0.145630 | 0.469348 | 0.041789 | 0.182014 |
| margin_blend_spline_c0p1_balanced_a50 | latest | 0.322571 | 0.163819 | 0.445817 | 0.037379 | 0.190067 |
| margin_blend_linear_c0p1_balanced_a25 | rank_fraction | 0.322551 | 0.142191 | 0.475264 | 0.042237 | 0.187570 |
| margin_blend_linear_c0p1_balanced_a50 | rank_fraction | 0.322530 | 0.146228 | 0.465111 | 0.040275 | 0.180998 |
| margin_blend_spline_c0p1_balanced_a10 | rank_fraction | 0.322369 | 0.145262 | 0.468022 | 0.040797 | 0.189627 |
| margin_blend_linear_c0p1_unweighted_a50 | rank_fraction | 0.322354 | 0.140436 | 0.481633 | 0.043493 | 0.181697 |
| logit_spline_c0p1_unweighted | rank_fraction | 0.322264 | 0.150523 | 0.452966 | 0.037916 | 0.186404 |
| spline_c0p1_unweighted | rank_fraction | 0.322264 | 0.150523 | 0.452966 | 0.037916 | 0.186404 |
| baseline_lgbm | latest | 0.321424 | 0.144608 | 0.469348 | 0.042196 | 0.182014 |
| margin_blend_linear_c0p1_balanced_a10 | rank_fraction | 0.321183 | 0.140280 | 0.479583 | 0.043396 | 0.188763 |
| margin_blend_linear_c0p1_unweighted_a50 | pooled | 0.320317 | 0.153871 | 0.444810 | 0.037493 | 0.181697 |
| logit_spline_c0p1_balanced | rank_fraction | 0.320217 | 0.153218 | 0.442143 | 0.036346 | 0.186464 |
| spline_c0p1_balanced | rank_fraction | 0.320217 | 0.153218 | 0.442143 | 0.036346 | 0.186464 |
| logit_spline_c0p1_unweighted | pooled | 0.319745 | 0.155748 | 0.447023 | 0.038270 | 0.186404 |
| spline_c0p1_unweighted | pooled | 0.319745 | 0.155748 | 0.447023 | 0.038270 | 0.186404 |
| margin_blend_linear_c0p1_balanced_a50 | pooled | 0.319591 | 0.166382 | 0.421679 | 0.033052 | 0.180998 |
| margin_blend_linear_c0p1_unweighted_a50 | latest | 0.319464 | 0.147602 | 0.464465 | 0.041865 | 0.181697 |
| baseline_balanced | rank_fraction | 0.318771 | 0.149056 | 0.450543 | 0.038377 | 0.183160 |
| baseline_lgbm | rank_fraction | 0.318729 | 0.137376 | 0.480665 | 0.044290 | 0.182014 |
| logit_spline_c0p1_unweighted | latest | 0.316889 | 0.147446 | 0.474610 | 0.045055 | 0.186404 |
| spline_c0p1_unweighted | latest | 0.316889 | 0.147446 | 0.474610 | 0.045055 | 0.186404 |
| margin_blend_linear_c0p1_balanced_a50 | latest | 0.316805 | 0.154662 | 0.451335 | 0.040096 | 0.180998 |
| logit_spline_c0p1_balanced | pooled | 0.314454 | 0.165450 | 0.421403 | 0.034476 | 0.186464 |
| spline_c0p1_balanced | pooled | 0.314454 | 0.165450 | 0.421403 | 0.034476 | 0.186464 |
| logit_spline_c0p1_balanced | latest | 0.310590 | 0.153727 | 0.458300 | 0.043591 | 0.186464 |
| spline_c0p1_balanced | latest | 0.310590 | 0.153727 | 0.458300 | 0.043591 | 0.186464 |
| linear_c0p1_unweighted | latest | 0.307896 | 0.143952 | 0.439725 | 0.040010 | 0.157733 |
| logit_linear_c0p1_unweighted | latest | 0.307896 | 0.143952 | 0.439725 | 0.040010 | 0.157733 |
| linear_c0p1_unweighted | pooled | 0.305998 | 0.147037 | 0.424208 | 0.037292 | 0.157733 |
| logit_linear_c0p1_unweighted | pooled | 0.305998 | 0.147037 | 0.424208 | 0.037292 | 0.157733 |
| linear_c0p1_unweighted | rank_fraction | 0.304713 | 0.128789 | 0.468722 | 0.046264 | 0.157733 |
| logit_linear_c0p1_unweighted | rank_fraction | 0.304713 | 0.128789 | 0.468722 | 0.046264 | 0.157733 |
| linear_c0p1_balanced | latest | 0.303622 | 0.147157 | 0.430062 | 0.039297 | 0.155346 |
| logit_linear_c0p1_balanced | latest | 0.303622 | 0.147157 | 0.430062 | 0.039297 | 0.155346 |
| linear_c0p1_balanced | rank_fraction | 0.301976 | 0.130678 | 0.456969 | 0.044555 | 0.155346 |
| logit_linear_c0p1_balanced | rank_fraction | 0.301976 | 0.130678 | 0.456969 | 0.044555 | 0.155346 |
| linear_c0p1_balanced | pooled | 0.299548 | 0.157323 | 0.390062 | 0.031897 | 0.155346 |
| logit_linear_c0p1_balanced | pooled | 0.299548 | 0.157323 | 0.390062 | 0.031897 | 0.155346 |

## Interpretation and matched effects

Original versus logit singles separate the clipping/transformation effect from adding a second member. Every blend is compared with both common-margin members at the same policy; all representation, training-weight and blend-weight contrasts retain negatives and ties. Non-pooled versus own pooled comparisons isolate policy changes. The operating figure's right panel holds the policy type fixed at pooled. Complete numeric effects are in contrasts.csv; score correlation and joint-error overlap are descriptive and did not change weights or selection.

| contrast | policy | reference | reference_policy | f2_difference |
| --- | --- | --- | --- | --- |
| old pooled reference | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.000701 |
| old pooled reference | pooled | baseline_balanced | pooled | 0.000403 |
| old pooled reference | pooled | baseline_lgbm | pooled | 0.005551 |
| blend minus common-margin member | pooled | logit_spline_c0p1_unweighted | pooled | 0.008460 |
| blend minus common-margin member | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.000701 |
| old pooled reference | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | 0.000103 |
| old pooled reference | latest | baseline_balanced | pooled | 0.001208 |
| old pooled reference | latest | baseline_lgbm | pooled | 0.006356 |
| policy minus own pooled | latest | margin_blend_spline_c0p1_unweighted_a25 | pooled | 0.000805 |
| blend minus common-margin member | latest | logit_spline_c0p1_unweighted | latest | 0.012121 |
| blend minus common-margin member | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | 0.001428 |
| old pooled reference | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.005850 |
| old pooled reference | rank_fraction | baseline_balanced | pooled | -0.004745 |
| old pooled reference | rank_fraction | baseline_lgbm | pooled | 0.000403 |
| policy minus own pooled | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | pooled | -0.005148 |
| blend minus common-margin member | rank_fraction | logit_spline_c0p1_unweighted | rank_fraction | 0.000793 |
| blend minus common-margin member | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | rank_fraction | -0.000815 |
| spline minus linear blend | pooled | margin_blend_linear_c0p1_unweighted_a25 | pooled | 0.000453 |
| spline minus linear blend | latest | margin_blend_linear_c0p1_unweighted_a25 | latest | 0.001248 |
| spline minus linear blend | rank_fraction | margin_blend_linear_c0p1_unweighted_a25 | rank_fraction | -0.000747 |
| higher minus lower Newton weight | pooled | margin_blend_spline_c0p1_unweighted_a10 | pooled | 0.000313 |
| higher minus lower Newton weight | latest | margin_blend_spline_c0p1_unweighted_a10 | latest | 0.002403 |
| higher minus lower Newton weight | rank_fraction | margin_blend_spline_c0p1_unweighted_a10 | rank_fraction | -0.002283 |

## Conditional development differences

| scope | reference | difference | ci_low | ci_high | replicates |
| --- | --- | --- | --- | --- | --- |
| 5 | fusion | -0.000576 | -0.007309 | 0.005420 | 500 |
| 5 | balanced | 0.000749 | -0.006548 | 0.008579 | 500 |
| 5 | unweighted | 0.000883 | -0.006898 | 0.008136 | 500 |
| 5 | same_model | 0.000000 | 0.000000 | 0.000000 | 500 |
| 5 | matched_representation | -0.001419 | -0.008023 | 0.004786 | 500 |
| 5 | matched_weight | 0.000527 | -0.006208 | 0.006917 | 500 |
| 5 | transformed_newton_member | 0.009142 | -0.000005 | 0.019113 | 500 |
| 5 | fusion_member | -0.000576 | -0.007309 | 0.005420 | 500 |
| 5 | raw_newton_member | 0.009142 | -0.000005 | 0.019113 | 500 |
| 6 | fusion | 0.000783 | -0.005143 | 0.006285 | 500 |
| 6 | balanced | 0.001668 | -0.006011 | 0.009270 | 500 |
| 6 | unweighted | 0.011829 | 0.005064 | 0.019031 | 500 |
| 6 | same_model | 0.001610 | 0.000072 | 0.003127 | 500 |
| 6 | matched_representation | 0.003916 | -0.001604 | 0.009046 | 500 |
| 6 | matched_weight | 0.002670 | 0.000224 | 0.004859 | 500 |
| 6 | transformed_newton_member | 0.015100 | 0.006072 | 0.024000 | 500 |
| 6 | fusion_member | 0.003432 | -0.001519 | 0.007868 | 500 |
| 6 | raw_newton_member | 0.015100 | 0.006072 | 0.024000 | 500 |
| macro | fusion | 0.000103 | -0.004381 | 0.004735 | 500 |
| macro | balanced | 0.001208 | -0.003665 | 0.006420 | 500 |
| macro | unweighted | 0.006356 | 0.001273 | 0.011306 | 500 |
| macro | same_model | 0.000805 | 0.000036 | 0.001564 | 500 |
| macro | matched_representation | 0.001248 | -0.003024 | 0.005684 | 500 |
| macro | matched_weight | 0.001598 | -0.001981 | 0.004948 | 500 |
| macro | transformed_newton_member | 0.012121 | 0.005576 | 0.019190 | 500 |
| macro | fusion_member | 0.001428 | -0.002765 | 0.005136 | 500 |
| macro | raw_newton_member | 0.012121 | 0.005576 | 0.019190 | 500 |

## Already-consumed month 7

| role | candidate | policy | f2 | precision | recall | flagged_fraction | tp | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected | margin_blend_spline_c0p1_unweighted_a25 | latest | 0.370248 | 0.200839 | 0.469188 | 0.034448 | 670 | 2666 | 758 |
| fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | 0.372694 | 0.192414 | 0.486695 | 0.037297 | 695 | 2917 | 733 |
| balanced | baseline_balanced | pooled | 0.364409 | 0.181747 | 0.486695 | 0.039487 | 695 | 3129 | 733 |
| unweighted | baseline_lgbm | pooled | 0.376884 | 0.168248 | 0.546218 | 0.047871 | 780 | 3856 | 648 |
| same_model | margin_blend_spline_c0p1_unweighted_a25 | pooled | 0.370833 | 0.195310 | 0.478291 | 0.036110 | 683 | 2814 | 745 |
| matched_representation | margin_blend_linear_c0p1_unweighted_a25 | latest | 0.371510 | 0.199646 | 0.473389 | 0.034964 | 676 | 2710 | 752 |
| matched_weight | margin_blend_spline_c0p1_balanced_a25 | latest | 0.369139 | 0.201269 | 0.466387 | 0.034169 | 666 | 2643 | 762 |
| transformed_newton_member | logit_spline_c0p1_unweighted | latest | 0.356534 | 0.203505 | 0.439076 | 0.031814 | 627 | 2454 | 801 |
| fusion_member | logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | 0.372369 | 0.191116 | 0.488095 | 0.037659 | 697 | 2950 | 731 |
| raw_newton_member | spline_c0p1_unweighted | latest | 0.356534 | 0.203505 | 0.439076 | 0.031814 | 627 | 2454 | 801 |

Only the frozen winner-type roles transferred with deduplicated work: 3 base reuses and 3 new fits, including 3 Newton fits. Freeze/access hashes and exact dependencies are exported. No losing-grid sweep or hindsight threshold was evaluated.

| scope | reference | difference | ci_low | ci_high | replicates |
| --- | --- | --- | --- | --- | --- |
| 7_consumed | fusion | -0.002447 | -0.007202 | 0.003564 | 500 |
| 7_consumed | balanced | 0.005839 | -0.002027 | 0.013381 | 500 |
| 7_consumed | unweighted | -0.006637 | -0.015719 | 0.003520 | 500 |
| 7_consumed | same_model | -0.000585 | -0.004208 | 0.002978 | 500 |
| 7_consumed | matched_representation | -0.001263 | -0.008186 | 0.005556 | 500 |
| 7_consumed | matched_weight | 0.001109 | -0.002084 | 0.004410 | 500 |
| 7_consumed | transformed_newton_member | 0.013714 | 0.003402 | 0.022825 | 500 |
| 7_consumed | fusion_member | -0.002121 | -0.007602 | 0.003912 | 500 |
| 7_consumed | raw_newton_member | 0.013714 | 0.003402 | 0.022825 | 500 |

Uncertainty uses 500 paired fixed-decision draws per month, seed 1027; all 1,000 development and 500 consumed draws were independently reconstructed by direct sampled Boolean rows. Intervals condition on fixed models/policies and exclude adaptive source/C choice, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity adjustment. Month 7 was consumed before this study and cannot supply independent confirmation. Analyst-selected settings remain open to group review.

## Reproduction and assistance

See README's ordered commands and protocol.md. Configuration, original fit/analysis ancestry, package/Python/data hashes, model/prediction parents, preservation presence, frozen rules, transfer counts and checked tables are exported. Codex assisted with the study design, implementation, numerical/provenance audits, diagnostics, reporting and tests, reusing preserved repository interfaces and pinned public-library APIs. The group must explain and verify any submitted analysis.
