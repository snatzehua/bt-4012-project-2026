# Complete matched effects

All342 predeclared comparisons are retained in `contrasts.csv`, including ties and negative results. Counts below use absolute tolerance1e-12 for numerical equality. These summarize a finite adaptive search and do not establish independent population effects. Every row names its exact candidate/policy/reference factors; weighting, seed and aggregation interpretations apply only to otherwise-matched rows.

| contrast | comparisons | positive | tied | negative | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- |
| aggregate minus member | 36 | 26 | 0 | 10 | -0.006153 | 0.006945 |
| balanced minus unweighted | 15 | 13 | 0 | 2 | -0.008344 | 0.012523 |
| fixed reference balanced | 45 | 6 | 1 | 38 | -0.013493 | 0.002992 |
| fixed reference fusion | 45 | 3 | 1 | 41 | -0.014598 | 0.001887 |
| fixed reference latest_fusion | 45 | 0 | 1 | 44 | -0.016485 | 0.000000 |
| fixed reference recent_fusion | 45 | 2 | 1 | 42 | -0.014701 | 0.001783 |
| fixed reference unweighted | 45 | 27 | 1 | 17 | -0.008345 | 0.008140 |
| logit mean minus probability mean | 6 | 5 | 0 | 1 | -0.000632 | 0.002173 |
| policy minus own pooled | 30 | 10 | 0 | 20 | -0.009031 | 0.009810 |
| randomized minus unbagged | 30 | 11 | 0 | 19 | -0.007115 | 0.009156 |

## All declared factors

| contrast | candidate | policy | reference | reference_policy | f2_difference | recall_difference | precision_difference | flagged_fraction_difference | average_precision_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed reference latest_fusion | interaction_blend_spline_c1p0_balanced_a10 | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.000078 | -0.000345 | 0.000043 | -0.000046 | 0.000000 |
| fixed reference recent_fusion | interaction_blend_spline_c1p0_balanced_a10 | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | 0.001706 | -0.000756 | 0.003165 | -0.000522 | -0.000673 |
| fixed reference fusion | interaction_blend_spline_c1p0_balanced_a10 | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | 0.001809 | 0.019884 | -0.008189 | 0.003371 | 0.001067 |
| fixed reference balanced | interaction_blend_spline_c1p0_balanced_a10 | pooled | baseline_balanced | pooled | 0.002914 | 0.023323 | -0.007991 | 0.003800 | 0.006533 |
| fixed reference unweighted | interaction_blend_spline_c1p0_balanced_a10 | pooled | baseline_lgbm | pooled | 0.008062 | -0.009855 | 0.013072 | -0.004237 | 0.007679 |
| fixed reference latest_fusion | interaction_blend_spline_c1p0_balanced_a10 | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed reference recent_fusion | interaction_blend_spline_c1p0_balanced_a10 | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | 0.001783 | -0.000412 | 0.003122 | -0.000476 | -0.000673 |
| fixed reference fusion | interaction_blend_spline_c1p0_balanced_a10 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | 0.001887 | 0.020229 | -0.008232 | 0.003417 | 0.001067 |
| fixed reference balanced | interaction_blend_spline_c1p0_balanced_a10 | latest | baseline_balanced | pooled | 0.002992 | 0.023667 | -0.008034 | 0.003846 | 0.006533 |
| fixed reference unweighted | interaction_blend_spline_c1p0_balanced_a10 | latest | baseline_lgbm | pooled | 0.008140 | -0.009510 | 0.013029 | -0.004190 | 0.007679 |
| policy minus own pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | interaction_blend_spline_c1p0_balanced_a10 | pooled | 0.000078 | 0.000345 | -0.000043 | 0.000046 | 0.000000 |
| fixed reference latest_fusion | interaction_blend_spline_c1p0_balanced_a10 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.008877 | 0.007859 | -0.013421 | 0.003248 | 0.000000 |
| fixed reference recent_fusion | interaction_blend_spline_c1p0_balanced_a10 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.007094 | 0.007448 | -0.010299 | 0.002772 | -0.000673 |
| fixed reference fusion | interaction_blend_spline_c1p0_balanced_a10 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.006990 | 0.028088 | -0.021653 | 0.006665 | 0.001067 |
| fixed reference balanced | interaction_blend_spline_c1p0_balanced_a10 | rank_fraction | baseline_balanced | pooled | -0.005885 | 0.031527 | -0.021455 | 0.007094 | 0.006533 |
| fixed reference unweighted | interaction_blend_spline_c1p0_balanced_a10 | rank_fraction | baseline_lgbm | pooled | -0.000737 | -0.001651 | -0.000392 | -0.000942 | 0.007679 |
| policy minus own pooled | interaction_blend_spline_c1p0_balanced_a10 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | pooled | -0.008799 | 0.008204 | -0.013464 | 0.003294 | 0.000000 |
| fixed reference latest_fusion | margin_blend_spline_c0p1_unweighted_a25 | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.002588 | 0.001446 | -0.004219 | 0.000929 | 0.000673 |
| fixed reference recent_fusion | margin_blend_spline_c0p1_unweighted_a25 | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.000805 | 0.001034 | -0.001097 | 0.000453 | 0.000000 |
| fixed reference fusion | margin_blend_spline_c0p1_unweighted_a25 | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.000701 | 0.021675 | -0.012451 | 0.004346 | 0.001740 |
| fixed reference balanced | margin_blend_spline_c0p1_unweighted_a25 | pooled | baseline_balanced | pooled | 0.000403 | 0.025114 | -0.012253 | 0.004775 | 0.007206 |
| fixed reference unweighted | margin_blend_spline_c0p1_unweighted_a25 | pooled | baseline_lgbm | pooled | 0.005551 | -0.008064 | 0.008810 | -0.003261 | 0.008352 |
| fixed reference latest_fusion | margin_blend_spline_c0p1_unweighted_a25 | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.001783 | 0.000412 | -0.003122 | 0.000476 | 0.000673 |
| fixed reference recent_fusion | margin_blend_spline_c0p1_unweighted_a25 | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed reference fusion | margin_blend_spline_c0p1_unweighted_a25 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | 0.000103 | 0.020640 | -0.011354 | 0.003893 | 0.001740 |
| fixed reference balanced | margin_blend_spline_c0p1_unweighted_a25 | latest | baseline_balanced | pooled | 0.001208 | 0.024079 | -0.011156 | 0.004322 | 0.007206 |
| fixed reference unweighted | margin_blend_spline_c0p1_unweighted_a25 | latest | baseline_lgbm | pooled | 0.006356 | -0.009099 | 0.009907 | -0.003714 | 0.008352 |
| policy minus own pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | margin_blend_spline_c0p1_unweighted_a25 | pooled | 0.000805 | -0.001034 | 0.001097 | -0.000453 | 0.000000 |
| fixed reference latest_fusion | margin_blend_spline_c0p1_unweighted_a25 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.007736 | 0.011786 | -0.013318 | 0.003621 | 0.000673 |
| fixed reference recent_fusion | margin_blend_spline_c0p1_unweighted_a25 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.005953 | 0.011374 | -0.010196 | 0.003145 | 0.000000 |
| fixed reference fusion | margin_blend_spline_c0p1_unweighted_a25 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.005850 | 0.032014 | -0.021550 | 0.007038 | 0.001740 |
| fixed reference balanced | margin_blend_spline_c0p1_unweighted_a25 | rank_fraction | baseline_balanced | pooled | -0.004745 | 0.035453 | -0.021352 | 0.007467 | 0.007206 |
| fixed reference unweighted | margin_blend_spline_c0p1_unweighted_a25 | rank_fraction | baseline_lgbm | pooled | 0.000403 | 0.002275 | -0.000289 | -0.000570 | 0.008352 |
| policy minus own pooled | margin_blend_spline_c0p1_unweighted_a25 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | pooled | -0.005148 | 0.010340 | -0.009099 | 0.002692 | 0.000000 |
| fixed reference latest_fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.001887 | -0.020229 | 0.008232 | -0.003417 | -0.001067 |
| fixed reference recent_fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.000103 | -0.020640 | 0.011354 | -0.003893 | -0.001740 |
| fixed reference fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed reference balanced | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | baseline_balanced | pooled | 0.001105 | 0.003439 | 0.000198 | 0.000429 | 0.005465 |
| fixed reference unweighted | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | baseline_lgbm | pooled | 0.006253 | -0.029739 | 0.021261 | -0.007607 | 0.006612 |
| fixed reference latest_fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.003211 | -0.002987 | 0.001103 | 0.000494 | -0.001067 |
| fixed reference recent_fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.001428 | -0.003399 | 0.004224 | 0.000018 | -0.001740 |
| fixed reference fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.001325 | 0.017241 | -0.007130 | 0.003911 | 0.000000 |
| fixed reference balanced | logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | baseline_balanced | pooled | -0.000220 | 0.020680 | -0.006932 | 0.004339 | 0.005465 |
| fixed reference unweighted | logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | baseline_lgbm | pooled | 0.004928 | -0.012498 | 0.014132 | -0.003697 | 0.006612 |
| policy minus own pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.001325 | 0.017241 | -0.007130 | 0.003911 | 0.000000 |
| fixed reference latest_fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.006922 | -0.009582 | -0.003962 | -0.000739 | -0.001067 |
| fixed reference recent_fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.005138 | -0.009994 | -0.000840 | -0.001215 | -0.001740 |
| fixed reference fusion | logit_mean_xgb_d4_w1_baseline_balanced_a50 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.005035 | 0.010646 | -0.012194 | 0.002678 | 0.000000 |
| fixed reference balanced | logit_mean_xgb_d4_w1_baseline_balanced_a50 | rank_fraction | baseline_balanced | pooled | -0.003930 | 0.014085 | -0.011996 | 0.003107 | 0.005465 |
| fixed reference unweighted | logit_mean_xgb_d4_w1_baseline_balanced_a50 | rank_fraction | baseline_lgbm | pooled | 0.001218 | -0.019093 | 0.009067 | -0.004930 | 0.006612 |
| policy minus own pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.005035 | 0.010646 | -0.012194 | 0.002678 | 0.000000 |
| fixed reference latest_fusion | baseline_balanced | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.002992 | -0.023667 | 0.008034 | -0.003846 | -0.006533 |
| fixed reference recent_fusion | baseline_balanced | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.001208 | -0.024079 | 0.011156 | -0.004322 | -0.007206 |
| fixed reference fusion | baseline_balanced | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.001105 | -0.003439 | -0.000198 | -0.000429 | -0.005465 |
| fixed reference balanced | baseline_balanced | pooled | baseline_balanced | pooled | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed reference unweighted | baseline_balanced | pooled | baseline_lgbm | pooled | 0.005148 | -0.033178 | 0.021063 | -0.008036 | 0.001147 |
| fixed reference latest_fusion | baseline_balanced | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.003624 | -0.025736 | 0.008255 | -0.004086 | -0.006533 |
| fixed reference recent_fusion | baseline_balanced | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.001840 | -0.026148 | 0.011377 | -0.004562 | -0.007206 |
| fixed reference fusion | baseline_balanced | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.001737 | -0.005508 | 0.000022 | -0.000669 | -0.005465 |
| fixed reference balanced | baseline_balanced | latest | baseline_balanced | pooled | -0.000632 | -0.002069 | 0.000221 | -0.000240 | 0.000000 |
| fixed reference unweighted | baseline_balanced | latest | baseline_lgbm | pooled | 0.004516 | -0.035247 | 0.021284 | -0.008277 | 0.001147 |
| policy minus own pooled | baseline_balanced | latest | baseline_balanced | pooled | -0.000632 | -0.002069 | 0.000221 | -0.000240 | 0.000000 |
| fixed reference latest_fusion | baseline_balanced | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.012023 | -0.009295 | -0.009603 | 0.000779 | -0.006533 |
| fixed reference recent_fusion | baseline_balanced | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.010239 | -0.009706 | -0.006481 | 0.000303 | -0.007206 |
| fixed reference fusion | baseline_balanced | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.010136 | 0.010934 | -0.017835 | 0.004196 | -0.005465 |
| fixed reference balanced | baseline_balanced | rank_fraction | baseline_balanced | pooled | -0.009031 | 0.014373 | -0.017637 | 0.004624 | 0.000000 |
| fixed reference unweighted | baseline_balanced | rank_fraction | baseline_lgbm | pooled | -0.003883 | -0.018805 | 0.003426 | -0.003412 | 0.001147 |
| policy minus own pooled | baseline_balanced | rank_fraction | baseline_balanced | pooled | -0.009031 | 0.014373 | -0.017637 | 0.004624 | 0.000000 |
| fixed reference latest_fusion | baseline_lgbm | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.008140 | 0.009510 | -0.013029 | 0.004190 | -0.007679 |
| fixed reference recent_fusion | baseline_lgbm | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.006356 | 0.009099 | -0.009907 | 0.003714 | -0.008352 |
| fixed reference fusion | baseline_lgbm | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.006253 | 0.029739 | -0.021261 | 0.007607 | -0.006612 |
| fixed reference balanced | baseline_lgbm | pooled | baseline_balanced | pooled | -0.005148 | 0.033178 | -0.021063 | 0.008036 | -0.001147 |
| fixed reference unweighted | baseline_lgbm | pooled | baseline_lgbm | pooled | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed reference latest_fusion | baseline_lgbm | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.009369 | 0.009510 | -0.014051 | 0.004597 | -0.007679 |
| fixed reference recent_fusion | baseline_lgbm | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.007586 | 0.009099 | -0.010929 | 0.004121 | -0.008352 |
| fixed reference fusion | baseline_lgbm | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.007483 | 0.029739 | -0.022283 | 0.008014 | -0.006612 |
| fixed reference balanced | baseline_lgbm | latest | baseline_balanced | pooled | -0.006378 | 0.033178 | -0.022085 | 0.008443 | -0.001147 |
| fixed reference unweighted | baseline_lgbm | latest | baseline_lgbm | pooled | -0.001230 | 0.000000 | -0.001022 | 0.000407 | 0.000000 |
| policy minus own pooled | baseline_lgbm | latest | baseline_lgbm | pooled | -0.001230 | 0.000000 | -0.001022 | 0.000407 | 0.000000 |
| fixed reference latest_fusion | baseline_lgbm | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.012064 | 0.020827 | -0.021282 | 0.006691 | -0.007679 |
| fixed reference recent_fusion | baseline_lgbm | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.010281 | 0.020416 | -0.018161 | 0.006215 | -0.008352 |
| fixed reference fusion | baseline_lgbm | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.010177 | 0.041056 | -0.029515 | 0.010108 | -0.006612 |
| fixed reference balanced | baseline_lgbm | rank_fraction | baseline_balanced | pooled | -0.009072 | 0.044495 | -0.029317 | 0.010537 | -0.001147 |
| fixed reference unweighted | baseline_lgbm | rank_fraction | baseline_lgbm | pooled | -0.003924 | 0.011317 | -0.008253 | 0.002501 | 0.000000 |
| policy minus own pooled | baseline_lgbm | rank_fraction | baseline_lgbm | pooled | -0.003924 | 0.011317 | -0.008253 | 0.002501 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_s17 | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.008250 | -0.018619 | 0.001983 | -0.001252 | -0.004909 |
| fixed reference recent_fusion | bagged_balanced_s17 | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.006466 | -0.019031 | 0.005105 | -0.001728 | -0.005582 |
| fixed reference fusion | bagged_balanced_s17 | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.006363 | 0.001610 | -0.006249 | 0.002165 | -0.003842 |
| fixed reference balanced | bagged_balanced_s17 | pooled | baseline_balanced | pooled | -0.005258 | 0.005049 | -0.006051 | 0.002594 | 0.001624 |
| fixed reference unweighted | bagged_balanced_s17 | pooled | baseline_lgbm | pooled | -0.000110 | -0.028129 | 0.015012 | -0.005442 | 0.002770 |
| fixed reference latest_fusion | bagged_balanced_s17 | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.007305 | -0.019998 | 0.003322 | -0.001829 | -0.004909 |
| fixed reference recent_fusion | bagged_balanced_s17 | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.005522 | -0.020410 | 0.006444 | -0.002306 | -0.005582 |
| fixed reference fusion | bagged_balanced_s17 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.005418 | 0.000230 | -0.004910 | 0.001587 | -0.003842 |
| fixed reference balanced | bagged_balanced_s17 | latest | baseline_balanced | pooled | -0.004313 | 0.003669 | -0.004712 | 0.002016 | 0.001624 |
| fixed reference unweighted | bagged_balanced_s17 | latest | baseline_lgbm | pooled | 0.000835 | -0.029509 | 0.016351 | -0.006020 | 0.002770 |
| policy minus own pooled | bagged_balanced_s17 | latest | bagged_balanced_s17 | pooled | 0.000945 | -0.001379 | 0.001339 | -0.000578 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_s17 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.005099 | 0.007966 | -0.011068 | 0.002382 | -0.004909 |
| fixed reference recent_fusion | bagged_balanced_s17 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.003316 | 0.007554 | -0.007946 | 0.001906 | -0.005582 |
| fixed reference fusion | bagged_balanced_s17 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.003213 | 0.028194 | -0.019300 | 0.005799 | -0.003842 |
| fixed reference balanced | bagged_balanced_s17 | rank_fraction | baseline_balanced | pooled | -0.002108 | 0.031633 | -0.019102 | 0.006228 | 0.001624 |
| fixed reference unweighted | bagged_balanced_s17 | rank_fraction | baseline_lgbm | pooled | 0.003040 | -0.001545 | 0.001961 | -0.001808 | 0.002770 |
| policy minus own pooled | bagged_balanced_s17 | rank_fraction | bagged_balanced_s17 | pooled | 0.003150 | 0.026585 | -0.013051 | 0.003634 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_s29 | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.004812 | -0.018438 | 0.002666 | -0.002377 | -0.005194 |
| fixed reference recent_fusion | bagged_balanced_s29 | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.003029 | -0.018849 | 0.005788 | -0.002853 | -0.005867 |
| fixed reference fusion | bagged_balanced_s29 | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.002926 | 0.001791 | -0.005566 | 0.001040 | -0.004127 |
| fixed reference balanced | bagged_balanced_s29 | pooled | baseline_balanced | pooled | -0.001821 | 0.005230 | -0.005368 | 0.001468 | 0.001338 |
| fixed reference unweighted | bagged_balanced_s29 | pooled | baseline_lgbm | pooled | 0.003327 | -0.027948 | 0.015695 | -0.006568 | 0.002485 |
| fixed reference latest_fusion | bagged_balanced_s29 | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.006742 | -0.015334 | -0.000424 | -0.001166 | -0.005194 |
| fixed reference recent_fusion | bagged_balanced_s29 | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.004959 | -0.015746 | 0.002698 | -0.001642 | -0.005867 |
| fixed reference fusion | bagged_balanced_s29 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.004855 | 0.004894 | -0.008656 | 0.002251 | -0.004127 |
| fixed reference balanced | bagged_balanced_s29 | latest | baseline_balanced | pooled | -0.003750 | 0.008333 | -0.008458 | 0.002679 | 0.001338 |
| fixed reference unweighted | bagged_balanced_s29 | latest | baseline_lgbm | pooled | 0.001397 | -0.024845 | 0.012605 | -0.005357 | 0.002485 |
| policy minus own pooled | bagged_balanced_s29 | latest | bagged_balanced_s29 | pooled | -0.001930 | 0.003103 | -0.003090 | 0.001211 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_s29 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.007714 | 0.008530 | -0.012512 | 0.003078 | -0.005194 |
| fixed reference recent_fusion | bagged_balanced_s29 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.005930 | 0.008118 | -0.009390 | 0.002602 | -0.005867 |
| fixed reference fusion | bagged_balanced_s29 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.005827 | 0.028759 | -0.020744 | 0.006495 | -0.004127 |
| fixed reference balanced | bagged_balanced_s29 | rank_fraction | baseline_balanced | pooled | -0.004722 | 0.032197 | -0.020546 | 0.006924 | 0.001338 |
| fixed reference unweighted | bagged_balanced_s29 | rank_fraction | baseline_lgbm | pooled | 0.000426 | -0.000981 | 0.000517 | -0.001112 | 0.002485 |
| policy minus own pooled | bagged_balanced_s29 | rank_fraction | bagged_balanced_s29 | pooled | -0.002901 | 0.026968 | -0.015178 | 0.005455 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_s43 | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.005202 | -0.028609 | 0.008618 | -0.004224 | -0.005234 |
| fixed reference recent_fusion | bagged_balanced_s43 | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.003418 | -0.029021 | 0.011740 | -0.004700 | -0.005908 |
| fixed reference fusion | bagged_balanced_s43 | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.003315 | -0.008381 | 0.000386 | -0.000807 | -0.004167 |
| fixed reference balanced | bagged_balanced_s43 | pooled | baseline_balanced | pooled | -0.002210 | -0.004942 | 0.000584 | -0.000378 | 0.001298 |
| fixed reference unweighted | bagged_balanced_s43 | pooled | baseline_lgbm | pooled | 0.002938 | -0.038120 | 0.021647 | -0.008415 | 0.002445 |
| fixed reference latest_fusion | bagged_balanced_s43 | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.007516 | -0.013782 | 0.000989 | -0.000480 | -0.005234 |
| fixed reference recent_fusion | bagged_balanced_s43 | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.005733 | -0.014193 | 0.004111 | -0.000956 | -0.005908 |
| fixed reference fusion | bagged_balanced_s43 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.005629 | 0.006447 | -0.007244 | 0.002937 | -0.004167 |
| fixed reference balanced | bagged_balanced_s43 | latest | baseline_balanced | pooled | -0.004524 | 0.009886 | -0.007045 | 0.003366 | 0.001298 |
| fixed reference unweighted | bagged_balanced_s43 | latest | baseline_lgbm | pooled | 0.000623 | -0.023292 | 0.014018 | -0.004670 | 0.002445 |
| policy minus own pooled | bagged_balanced_s43 | latest | bagged_balanced_s43 | pooled | -0.002314 | 0.014828 | -0.007630 | 0.003744 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_s43 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.011252 | -0.013097 | -0.007029 | -0.000229 | -0.005234 |
| fixed reference recent_fusion | bagged_balanced_s43 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.009469 | -0.013509 | -0.003907 | -0.000705 | -0.005908 |
| fixed reference fusion | bagged_balanced_s43 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.009366 | 0.007131 | -0.015261 | 0.003188 | -0.004167 |
| fixed reference balanced | bagged_balanced_s43 | rank_fraction | baseline_balanced | pooled | -0.008261 | 0.010570 | -0.015063 | 0.003617 | 0.001298 |
| fixed reference unweighted | bagged_balanced_s43 | rank_fraction | baseline_lgbm | pooled | -0.003113 | -0.022608 | 0.006000 | -0.004420 | 0.002445 |
| policy minus own pooled | bagged_balanced_s43 | rank_fraction | bagged_balanced_s43 | pooled | -0.006051 | 0.015512 | -0.015647 | 0.003995 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_s17 | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.014137 | -0.004969 | -0.005733 | 0.003391 | -0.007406 |
| fixed reference recent_fusion | bagged_unweighted_s17 | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.012354 | -0.005380 | -0.002611 | 0.002915 | -0.008080 |
| fixed reference fusion | bagged_unweighted_s17 | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.012251 | 0.015260 | -0.013965 | 0.006808 | -0.006339 |
| fixed reference balanced | bagged_unweighted_s17 | pooled | baseline_balanced | pooled | -0.011146 | 0.018699 | -0.013767 | 0.007236 | -0.000874 |
| fixed reference unweighted | bagged_unweighted_s17 | pooled | baseline_lgbm | pooled | -0.005998 | -0.014479 | 0.007296 | -0.000800 | 0.000273 |
| fixed reference latest_fusion | bagged_unweighted_s17 | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.012066 | -0.012900 | -0.002181 | 0.001019 | -0.007406 |
| fixed reference recent_fusion | bagged_unweighted_s17 | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.010283 | -0.013311 | 0.000940 | 0.000543 | -0.008080 |
| fixed reference fusion | bagged_unweighted_s17 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.010179 | 0.007329 | -0.010414 | 0.004436 | -0.006339 |
| fixed reference balanced | bagged_unweighted_s17 | latest | baseline_balanced | pooled | -0.009074 | 0.010768 | -0.010215 | 0.004865 | -0.000874 |
| fixed reference unweighted | bagged_unweighted_s17 | latest | baseline_lgbm | pooled | -0.003927 | -0.022410 | 0.010848 | -0.003171 | 0.000273 |
| policy minus own pooled | bagged_unweighted_s17 | latest | bagged_unweighted_s17 | pooled | 0.002071 | -0.007931 | 0.003552 | -0.002371 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_s17 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.008772 | 0.012008 | -0.016799 | 0.004389 | -0.007406 |
| fixed reference recent_fusion | bagged_unweighted_s17 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.006989 | 0.011597 | -0.013677 | 0.003913 | -0.008080 |
| fixed reference fusion | bagged_unweighted_s17 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.006885 | 0.032237 | -0.025031 | 0.007806 | -0.006339 |
| fixed reference balanced | bagged_unweighted_s17 | rank_fraction | baseline_balanced | pooled | -0.005780 | 0.035676 | -0.024833 | 0.008235 | -0.000874 |
| fixed reference unweighted | bagged_unweighted_s17 | rank_fraction | baseline_lgbm | pooled | -0.000632 | 0.002498 | -0.003770 | 0.000199 | 0.000273 |
| policy minus own pooled | bagged_unweighted_s17 | rank_fraction | bagged_unweighted_s17 | pooled | 0.005366 | 0.016977 | -0.011066 | 0.000999 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_s29 | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.007173 | -0.034203 | 0.009982 | -0.004760 | -0.004705 |
| fixed reference recent_fusion | bagged_unweighted_s29 | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.005389 | -0.034614 | 0.013104 | -0.005236 | -0.005378 |
| fixed reference fusion | bagged_unweighted_s29 | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.005286 | -0.013974 | 0.001750 | -0.001343 | -0.003637 |
| fixed reference balanced | bagged_unweighted_s29 | pooled | baseline_balanced | pooled | -0.004181 | -0.010535 | 0.001948 | -0.000914 | 0.001828 |
| fixed reference unweighted | bagged_unweighted_s29 | pooled | baseline_lgbm | pooled | 0.000967 | -0.043713 | 0.023011 | -0.008951 | 0.002974 |
| fixed reference latest_fusion | bagged_unweighted_s29 | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.010331 | 0.003728 | -0.004239 | 0.004147 | -0.004705 |
| fixed reference recent_fusion | bagged_unweighted_s29 | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.008548 | 0.003317 | -0.001117 | 0.003671 | -0.005378 |
| fixed reference fusion | bagged_unweighted_s29 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.008445 | 0.023957 | -0.012471 | 0.007564 | -0.003637 |
| fixed reference balanced | bagged_unweighted_s29 | latest | baseline_balanced | pooled | -0.007340 | 0.027396 | -0.012273 | 0.007993 | 0.001828 |
| fixed reference unweighted | bagged_unweighted_s29 | latest | baseline_lgbm | pooled | -0.002192 | -0.005782 | 0.008790 | -0.000043 | 0.002974 |
| policy minus own pooled | bagged_unweighted_s29 | latest | bagged_unweighted_s29 | pooled | -0.003159 | 0.037931 | -0.014221 | 0.008907 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_s29 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.011150 | -0.023183 | -0.002404 | -0.002293 | -0.004705 |
| fixed reference recent_fusion | bagged_unweighted_s29 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.009367 | -0.023595 | 0.000718 | -0.002769 | -0.005378 |
| fixed reference fusion | bagged_unweighted_s29 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.009264 | -0.002954 | -0.010636 | 0.001124 | -0.003637 |
| fixed reference balanced | bagged_unweighted_s29 | rank_fraction | baseline_balanced | pooled | -0.008159 | 0.000484 | -0.010438 | 0.001553 | 0.001828 |
| fixed reference unweighted | bagged_unweighted_s29 | rank_fraction | baseline_lgbm | pooled | -0.003011 | -0.032694 | 0.010625 | -0.006483 | 0.002974 |
| policy minus own pooled | bagged_unweighted_s29 | rank_fraction | bagged_unweighted_s29 | pooled | -0.003978 | 0.011020 | -0.012386 | 0.002467 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_s43 | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.012718 | -0.014912 | -0.003859 | 0.000711 | -0.004787 |
| fixed reference recent_fusion | bagged_unweighted_s43 | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.010935 | -0.015323 | -0.000737 | 0.000235 | -0.005460 |
| fixed reference fusion | bagged_unweighted_s43 | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.010831 | 0.005317 | -0.012091 | 0.004128 | -0.003719 |
| fixed reference balanced | bagged_unweighted_s43 | pooled | baseline_balanced | pooled | -0.009726 | 0.008756 | -0.011893 | 0.004556 | 0.001746 |
| fixed reference unweighted | bagged_unweighted_s43 | pooled | baseline_lgbm | pooled | -0.004579 | -0.024422 | 0.009170 | -0.003480 | 0.002893 |
| fixed reference latest_fusion | bagged_unweighted_s43 | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.016007 | 0.003364 | -0.010719 | 0.005703 | -0.004787 |
| fixed reference recent_fusion | bagged_unweighted_s43 | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.014224 | 0.002953 | -0.007597 | 0.005227 | -0.005460 |
| fixed reference fusion | bagged_unweighted_s43 | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.014121 | 0.023593 | -0.018951 | 0.009120 | -0.003719 |
| fixed reference balanced | bagged_unweighted_s43 | latest | baseline_balanced | pooled | -0.013016 | 0.027032 | -0.018753 | 0.009549 | 0.001746 |
| fixed reference unweighted | bagged_unweighted_s43 | latest | baseline_lgbm | pooled | -0.007868 | -0.006146 | 0.002310 | 0.001512 | 0.002893 |
| policy minus own pooled | bagged_unweighted_s43 | latest | bagged_unweighted_s43 | pooled | -0.003289 | 0.018276 | -0.006860 | 0.004992 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_s43 | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.002908 | 0.010753 | -0.010217 | 0.002338 | -0.004787 |
| fixed reference recent_fusion | bagged_unweighted_s43 | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.001125 | 0.010341 | -0.007095 | 0.001862 | -0.005460 |
| fixed reference fusion | bagged_unweighted_s43 | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.001022 | 0.030982 | -0.018449 | 0.005755 | -0.003719 |
| fixed reference balanced | bagged_unweighted_s43 | rank_fraction | baseline_balanced | pooled | 0.000083 | 0.034420 | -0.018251 | 0.006183 | 0.001746 |
| fixed reference unweighted | bagged_unweighted_s43 | rank_fraction | baseline_lgbm | pooled | 0.005231 | 0.001242 | 0.002812 | -0.001853 | 0.002893 |
| policy minus own pooled | bagged_unweighted_s43 | rank_fraction | bagged_unweighted_s43 | pooled | 0.009810 | 0.025665 | -0.006358 | 0.001627 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_prob_mean | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.004237 | -0.021206 | 0.005115 | -0.003017 | -0.004116 |
| fixed reference recent_fusion | bagged_balanced_prob_mean | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.002454 | -0.021618 | 0.008236 | -0.003493 | -0.004789 |
| fixed reference fusion | bagged_balanced_prob_mean | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.002351 | -0.000977 | -0.003118 | 0.000400 | -0.003048 |
| fixed reference balanced | bagged_balanced_prob_mean | pooled | baseline_balanced | pooled | -0.001246 | 0.002461 | -0.002920 | 0.000829 | 0.002417 |
| fixed reference unweighted | bagged_balanced_prob_mean | pooled | baseline_lgbm | pooled | 0.003902 | -0.030717 | 0.018144 | -0.007207 | 0.003564 |
| fixed reference latest_fusion | bagged_balanced_prob_mean | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.003962 | -0.011551 | 0.001768 | -0.001121 | -0.004116 |
| fixed reference recent_fusion | bagged_balanced_prob_mean | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.002178 | -0.011962 | 0.004890 | -0.001598 | -0.004789 |
| fixed reference fusion | bagged_balanced_prob_mean | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.002075 | 0.008678 | -0.006464 | 0.002295 | -0.003048 |
| fixed reference balanced | bagged_balanced_prob_mean | latest | baseline_balanced | pooled | -0.000970 | 0.012117 | -0.006266 | 0.002724 | 0.002417 |
| fixed reference unweighted | bagged_balanced_prob_mean | latest | baseline_lgbm | pooled | 0.004178 | -0.021061 | 0.014797 | -0.005312 | 0.003564 |
| policy minus own pooled | bagged_balanced_prob_mean | latest | bagged_balanced_prob_mean | pooled | 0.000276 | 0.009655 | -0.003346 | 0.001895 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_prob_mean | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.005184 | 0.002936 | -0.007559 | 0.001285 | -0.004116 |
| fixed reference recent_fusion | bagged_balanced_prob_mean | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.003401 | 0.002525 | -0.004437 | 0.000809 | -0.004789 |
| fixed reference fusion | bagged_balanced_prob_mean | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.003297 | 0.023165 | -0.015791 | 0.004702 | -0.003048 |
| fixed reference balanced | bagged_balanced_prob_mean | rank_fraction | baseline_balanced | pooled | -0.002192 | 0.026604 | -0.015593 | 0.005131 | 0.002417 |
| fixed reference unweighted | bagged_balanced_prob_mean | rank_fraction | baseline_lgbm | pooled | 0.002956 | -0.006574 | 0.005470 | -0.002905 | 0.003564 |
| policy minus own pooled | bagged_balanced_prob_mean | rank_fraction | bagged_balanced_prob_mean | pooled | -0.000946 | 0.024142 | -0.012674 | 0.004302 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_logit_mean | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.004051 | -0.022269 | 0.005999 | -0.003280 | -0.004167 |
| fixed reference recent_fusion | bagged_balanced_logit_mean | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.002268 | -0.022681 | 0.009121 | -0.003756 | -0.004840 |
| fixed reference fusion | bagged_balanced_logit_mean | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.002164 | -0.002040 | -0.002233 | 0.000137 | -0.003100 |
| fixed reference balanced | bagged_balanced_logit_mean | pooled | baseline_balanced | pooled | -0.001059 | 0.001398 | -0.002035 | 0.000566 | 0.002366 |
| fixed reference unweighted | bagged_balanced_logit_mean | pooled | baseline_lgbm | pooled | 0.004089 | -0.031780 | 0.019028 | -0.007471 | 0.003512 |
| fixed reference latest_fusion | bagged_balanced_logit_mean | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.004593 | -0.013304 | 0.002074 | -0.001283 | -0.004167 |
| fixed reference recent_fusion | bagged_balanced_logit_mean | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.002810 | -0.013715 | 0.005196 | -0.001759 | -0.004840 |
| fixed reference fusion | bagged_balanced_logit_mean | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.002707 | 0.006925 | -0.006158 | 0.002134 | -0.003100 |
| fixed reference balanced | bagged_balanced_logit_mean | latest | baseline_balanced | pooled | -0.001602 | 0.010364 | -0.005960 | 0.002562 | 0.002366 |
| fixed reference unweighted | bagged_balanced_logit_mean | latest | baseline_lgbm | pooled | 0.003546 | -0.022814 | 0.015103 | -0.005474 | 0.003512 |
| policy minus own pooled | bagged_balanced_logit_mean | latest | bagged_balanced_logit_mean | pooled | -0.000543 | 0.008966 | -0.003925 | 0.001997 | 0.000000 |
| fixed reference latest_fusion | bagged_balanced_logit_mean | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.004858 | 0.002237 | -0.006937 | 0.001059 | -0.004167 |
| fixed reference recent_fusion | bagged_balanced_logit_mean | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.003075 | 0.001826 | -0.003816 | 0.000583 | -0.004840 |
| fixed reference fusion | bagged_balanced_logit_mean | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.002971 | 0.022466 | -0.015170 | 0.004476 | -0.003100 |
| fixed reference balanced | bagged_balanced_logit_mean | rank_fraction | baseline_balanced | pooled | -0.001866 | 0.025905 | -0.014972 | 0.004905 | 0.002366 |
| fixed reference unweighted | bagged_balanced_logit_mean | rank_fraction | baseline_lgbm | pooled | 0.003282 | -0.007273 | 0.006091 | -0.003131 | 0.003512 |
| policy minus own pooled | bagged_balanced_logit_mean | rank_fraction | bagged_balanced_logit_mean | pooled | -0.000807 | 0.024506 | -0.012936 | 0.004340 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_prob_mean | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.011851 | -0.024001 | 0.001557 | -0.001440 | -0.003893 |
| fixed reference recent_fusion | bagged_unweighted_prob_mean | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.010068 | -0.024413 | 0.004679 | -0.001916 | -0.004566 |
| fixed reference fusion | bagged_unweighted_prob_mean | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.009964 | -0.003772 | -0.006675 | 0.001977 | -0.002826 |
| fixed reference balanced | bagged_unweighted_prob_mean | pooled | baseline_balanced | pooled | -0.008859 | -0.000334 | -0.006477 | 0.002406 | 0.002640 |
| fixed reference unweighted | bagged_unweighted_prob_mean | pooled | baseline_lgbm | pooled | -0.003712 | -0.033512 | 0.014586 | -0.005630 | 0.003786 |
| fixed reference latest_fusion | bagged_unweighted_prob_mean | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.016485 | 0.000137 | -0.008163 | 0.005189 | -0.003893 |
| fixed reference recent_fusion | bagged_unweighted_prob_mean | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.014701 | -0.000275 | -0.005041 | 0.004713 | -0.004566 |
| fixed reference fusion | bagged_unweighted_prob_mean | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.014598 | 0.020366 | -0.016395 | 0.008606 | -0.002826 |
| fixed reference balanced | bagged_unweighted_prob_mean | latest | baseline_balanced | pooled | -0.013493 | 0.023804 | -0.016197 | 0.009034 | 0.002640 |
| fixed reference unweighted | bagged_unweighted_prob_mean | latest | baseline_lgbm | pooled | -0.008345 | -0.009374 | 0.004866 | 0.000998 | 0.003786 |
| policy minus own pooled | bagged_unweighted_prob_mean | latest | bagged_unweighted_prob_mean | pooled | -0.004634 | 0.024138 | -0.009720 | 0.006629 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_prob_mean | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.005649 | -0.003135 | -0.007136 | 0.000355 | -0.003893 |
| fixed reference recent_fusion | bagged_unweighted_prob_mean | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.003866 | -0.003547 | -0.004014 | -0.000121 | -0.004566 |
| fixed reference fusion | bagged_unweighted_prob_mean | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.003762 | 0.017093 | -0.015368 | 0.003772 | -0.002826 |
| fixed reference balanced | bagged_unweighted_prob_mean | rank_fraction | baseline_balanced | pooled | -0.002657 | 0.020532 | -0.015170 | 0.004200 | 0.002640 |
| fixed reference unweighted | bagged_unweighted_prob_mean | rank_fraction | baseline_lgbm | pooled | 0.002490 | -0.012646 | 0.005893 | -0.003836 | 0.003786 |
| policy minus own pooled | bagged_unweighted_prob_mean | rank_fraction | bagged_unweighted_prob_mean | pooled | 0.006202 | 0.020866 | -0.008693 | 0.001794 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_logit_mean | pooled | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.009678 | -0.013216 | -0.000507 | 0.000183 | -0.003786 |
| fixed reference recent_fusion | bagged_unweighted_logit_mean | pooled | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.007894 | -0.013628 | 0.002614 | -0.000293 | -0.004459 |
| fixed reference fusion | bagged_unweighted_logit_mean | pooled | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.007791 | 0.007013 | -0.008740 | 0.003600 | -0.002719 |
| fixed reference balanced | bagged_unweighted_logit_mean | pooled | baseline_balanced | pooled | -0.006686 | 0.010451 | -0.008542 | 0.004028 | 0.002747 |
| fixed reference unweighted | bagged_unweighted_logit_mean | pooled | baseline_lgbm | pooled | -0.001538 | -0.022727 | 0.012522 | -0.004008 | 0.003893 |
| fixed reference latest_fusion | bagged_unweighted_logit_mean | latest | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.015783 | 0.004370 | -0.009293 | 0.005979 | -0.003786 |
| fixed reference recent_fusion | bagged_unweighted_logit_mean | latest | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.014000 | 0.003959 | -0.006171 | 0.005503 | -0.004459 |
| fixed reference fusion | bagged_unweighted_logit_mean | latest | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.013896 | 0.024599 | -0.017525 | 0.009396 | -0.002719 |
| fixed reference balanced | bagged_unweighted_logit_mean | latest | baseline_balanced | pooled | -0.012791 | 0.028038 | -0.017327 | 0.009825 | 0.002747 |
| fixed reference unweighted | bagged_unweighted_logit_mean | latest | baseline_lgbm | pooled | -0.007643 | -0.005140 | 0.003736 | 0.001789 | 0.003893 |
| policy minus own pooled | bagged_unweighted_logit_mean | latest | bagged_unweighted_logit_mean | pooled | -0.006105 | 0.017586 | -0.008785 | 0.005797 | 0.000000 |
| fixed reference latest_fusion | bagged_unweighted_logit_mean | rank_fraction | interaction_blend_spline_c1p0_balanced_a10 | latest | -0.004205 | 0.005897 | -0.009670 | 0.001760 | -0.003786 |
| fixed reference recent_fusion | bagged_unweighted_logit_mean | rank_fraction | margin_blend_spline_c0p1_unweighted_a25 | latest | -0.002422 | 0.005485 | -0.006548 | 0.001284 | -0.004459 |
| fixed reference fusion | bagged_unweighted_logit_mean | rank_fraction | logit_mean_xgb_d4_w1_baseline_balanced_a50 | pooled | -0.002318 | 0.026126 | -0.017902 | 0.005177 | -0.002719 |
| fixed reference balanced | bagged_unweighted_logit_mean | rank_fraction | baseline_balanced | pooled | -0.001213 | 0.029564 | -0.017704 | 0.005606 | 0.002747 |
| fixed reference unweighted | bagged_unweighted_logit_mean | rank_fraction | baseline_lgbm | pooled | 0.003935 | -0.003614 | 0.003359 | -0.002430 | 0.003893 |
| policy minus own pooled | bagged_unweighted_logit_mean | rank_fraction | bagged_unweighted_logit_mean | pooled | 0.005473 | 0.019113 | -0.009163 | 0.001578 | 0.000000 |
| randomized minus unbagged | bagged_balanced_s17 | pooled | baseline_balanced | pooled | -0.005258 | 0.005049 | -0.006051 | 0.002594 | 0.001624 |
| randomized minus unbagged | bagged_balanced_s17 | latest | baseline_balanced | latest | -0.003681 | 0.005738 | -0.004933 | 0.002257 | 0.001624 |
| randomized minus unbagged | bagged_balanced_s17 | rank_fraction | baseline_balanced | rank_fraction | 0.006923 | 0.017260 | -0.001465 | 0.001603 | 0.001624 |
| randomized minus unbagged | bagged_balanced_s29 | pooled | baseline_balanced | pooled | -0.001821 | 0.005230 | -0.005368 | 0.001468 | 0.001338 |
| randomized minus unbagged | bagged_balanced_s29 | latest | baseline_balanced | latest | -0.003118 | 0.010402 | -0.008679 | 0.002920 | 0.001338 |
| randomized minus unbagged | bagged_balanced_s29 | rank_fraction | baseline_balanced | rank_fraction | 0.004309 | 0.017824 | -0.002909 | 0.002299 | 0.001338 |
| randomized minus unbagged | bagged_balanced_s43 | pooled | baseline_balanced | pooled | -0.002210 | -0.004942 | 0.000584 | -0.000378 | 0.001298 |
| randomized minus unbagged | bagged_balanced_s43 | latest | baseline_balanced | latest | -0.003892 | 0.011955 | -0.007266 | 0.003606 | 0.001298 |
| randomized minus unbagged | bagged_balanced_s43 | rank_fraction | baseline_balanced | rank_fraction | 0.000770 | -0.003803 | 0.002574 | -0.001008 | 0.001298 |
| randomized minus unbagged | bagged_unweighted_s17 | pooled | baseline_lgbm | pooled | -0.005998 | -0.014479 | 0.007296 | -0.000800 | 0.000273 |
| randomized minus unbagged | bagged_unweighted_s17 | latest | baseline_lgbm | latest | -0.002697 | -0.022410 | 0.011869 | -0.003578 | 0.000273 |
| randomized minus unbagged | bagged_unweighted_s17 | rank_fraction | baseline_lgbm | rank_fraction | 0.003292 | -0.008819 | 0.004484 | -0.002302 | 0.000273 |
| randomized minus unbagged | bagged_unweighted_s29 | pooled | baseline_lgbm | pooled | 0.000967 | -0.043713 | 0.023011 | -0.008951 | 0.002974 |
| randomized minus unbagged | bagged_unweighted_s29 | latest | baseline_lgbm | latest | -0.000962 | -0.005782 | 0.009812 | -0.000450 | 0.002974 |
| randomized minus unbagged | bagged_unweighted_s29 | rank_fraction | baseline_lgbm | rank_fraction | 0.000914 | -0.044011 | 0.018879 | -0.008984 | 0.002974 |
| randomized minus unbagged | bagged_unweighted_s43 | pooled | baseline_lgbm | pooled | -0.004579 | -0.024422 | 0.009170 | -0.003480 | 0.002893 |
| randomized minus unbagged | bagged_unweighted_s43 | latest | baseline_lgbm | latest | -0.006638 | -0.006146 | 0.003332 | 0.001106 | 0.002893 |
| randomized minus unbagged | bagged_unweighted_s43 | rank_fraction | baseline_lgbm | rank_fraction | 0.009156 | -0.010075 | 0.011066 | -0.004354 | 0.002893 |
| randomized minus unbagged | bagged_balanced_prob_mean | pooled | baseline_balanced | pooled | -0.001246 | 0.002461 | -0.002920 | 0.000829 | 0.002417 |
| aggregate minus member | bagged_balanced_prob_mean | pooled | bagged_balanced_s17 | pooled | 0.004012 | -0.002587 | 0.003132 | -0.001765 | 0.000793 |
| aggregate minus member | bagged_balanced_prob_mean | pooled | bagged_balanced_s29 | pooled | 0.000575 | -0.002768 | 0.002448 | -0.000639 | 0.001079 |
| aggregate minus member | bagged_balanced_prob_mean | pooled | bagged_balanced_s43 | pooled | 0.000964 | 0.007403 | -0.003504 | 0.001207 | 0.001119 |
| randomized minus unbagged | bagged_balanced_prob_mean | latest | baseline_balanced | latest | -0.000338 | 0.014186 | -0.006487 | 0.002965 | 0.002417 |
| aggregate minus member | bagged_balanced_prob_mean | latest | bagged_balanced_s17 | latest | 0.003343 | 0.008447 | -0.001554 | 0.000708 | 0.000793 |
| aggregate minus member | bagged_balanced_prob_mean | latest | bagged_balanced_s29 | latest | 0.002780 | 0.003784 | 0.002192 | 0.000045 | 0.001079 |
| aggregate minus member | bagged_balanced_prob_mean | latest | bagged_balanced_s43 | latest | 0.003554 | 0.002231 | 0.000780 | -0.000642 | 0.001119 |
| randomized minus unbagged | bagged_balanced_prob_mean | rank_fraction | baseline_balanced | rank_fraction | 0.006839 | 0.012231 | 0.002044 | 0.000507 | 0.002417 |
| aggregate minus member | bagged_balanced_prob_mean | rank_fraction | bagged_balanced_s17 | rank_fraction | -0.000085 | -0.005029 | 0.003509 | -0.001097 | 0.000793 |
| aggregate minus member | bagged_balanced_prob_mean | rank_fraction | bagged_balanced_s29 | rank_fraction | 0.002530 | -0.005593 | 0.004953 | -0.001793 | 0.001079 |
| aggregate minus member | bagged_balanced_prob_mean | rank_fraction | bagged_balanced_s43 | rank_fraction | 0.006068 | 0.016034 | -0.000530 | 0.001514 | 0.001119 |
| randomized minus unbagged | bagged_balanced_logit_mean | pooled | baseline_balanced | pooled | -0.001059 | 0.001398 | -0.002035 | 0.000566 | 0.002366 |
| aggregate minus member | bagged_balanced_logit_mean | pooled | bagged_balanced_s17 | pooled | 0.004199 | -0.003650 | 0.004016 | -0.002028 | 0.000742 |
| aggregate minus member | bagged_balanced_logit_mean | pooled | bagged_balanced_s29 | pooled | 0.000761 | -0.003831 | 0.003333 | -0.000903 | 0.001027 |
| aggregate minus member | bagged_balanced_logit_mean | pooled | bagged_balanced_s43 | pooled | 0.001151 | 0.006340 | -0.002619 | 0.000944 | 0.001067 |
| randomized minus unbagged | bagged_balanced_logit_mean | latest | baseline_balanced | latest | -0.000970 | 0.012433 | -0.006181 | 0.002803 | 0.002366 |
| aggregate minus member | bagged_balanced_logit_mean | latest | bagged_balanced_s17 | latest | 0.002711 | 0.006695 | -0.001248 | 0.000546 | 0.000742 |
| aggregate minus member | bagged_balanced_logit_mean | latest | bagged_balanced_s29 | latest | 0.002149 | 0.002031 | 0.002498 | -0.000117 | 0.001027 |
| aggregate minus member | bagged_balanced_logit_mean | latest | bagged_balanced_s43 | latest | 0.002923 | 0.000478 | 0.001085 | -0.000803 | 0.001067 |
| randomized minus unbagged | bagged_balanced_logit_mean | rank_fraction | baseline_balanced | rank_fraction | 0.007165 | 0.011532 | 0.002665 | 0.000281 | 0.002366 |
| aggregate minus member | bagged_balanced_logit_mean | rank_fraction | bagged_balanced_s17 | rank_fraction | 0.000241 | -0.005729 | 0.004131 | -0.001323 | 0.000742 |
| aggregate minus member | bagged_balanced_logit_mean | rank_fraction | bagged_balanced_s29 | rank_fraction | 0.002856 | -0.006293 | 0.005574 | -0.002019 | 0.001027 |
| aggregate minus member | bagged_balanced_logit_mean | rank_fraction | bagged_balanced_s43 | rank_fraction | 0.006394 | 0.015334 | 0.000091 | 0.001289 | 0.001067 |
| randomized minus unbagged | bagged_unweighted_prob_mean | pooled | baseline_lgbm | pooled | -0.003712 | -0.033512 | 0.014586 | -0.005630 | 0.003786 |
| aggregate minus member | bagged_unweighted_prob_mean | pooled | bagged_unweighted_s17 | pooled | 0.002286 | -0.019032 | 0.007290 | -0.004830 | 0.003513 |
| aggregate minus member | bagged_unweighted_prob_mean | pooled | bagged_unweighted_s29 | pooled | -0.004678 | 0.010202 | -0.008425 | 0.003320 | 0.000812 |
| aggregate minus member | bagged_unweighted_prob_mean | pooled | bagged_unweighted_s43 | pooled | 0.000867 | -0.009089 | 0.005416 | -0.002150 | 0.000894 |
| randomized minus unbagged | bagged_unweighted_prob_mean | latest | baseline_lgbm | latest | -0.007115 | -0.009374 | 0.005888 | 0.000591 | 0.003786 |
| aggregate minus member | bagged_unweighted_prob_mean | latest | bagged_unweighted_s17 | latest | -0.004419 | 0.013037 | -0.005982 | 0.004169 | 0.003513 |
| aggregate minus member | bagged_unweighted_prob_mean | latest | bagged_unweighted_s29 | latest | -0.006153 | -0.003591 | -0.003924 | 0.001041 | 0.000812 |
| aggregate minus member | bagged_unweighted_prob_mean | latest | bagged_unweighted_s43 | latest | -0.000477 | -0.003227 | 0.002556 | -0.000514 | 0.000894 |
| randomized minus unbagged | bagged_unweighted_prob_mean | rank_fraction | baseline_lgbm | rank_fraction | 0.006415 | -0.023963 | 0.014146 | -0.006337 | 0.003786 |
| aggregate minus member | bagged_unweighted_prob_mean | rank_fraction | bagged_unweighted_s17 | rank_fraction | 0.003123 | -0.015144 | 0.009663 | -0.004035 | 0.003513 |
| aggregate minus member | bagged_unweighted_prob_mean | rank_fraction | bagged_unweighted_s29 | rank_fraction | 0.005501 | 0.020048 | -0.004732 | 0.002647 | 0.000812 |
| aggregate minus member | bagged_unweighted_prob_mean | rank_fraction | bagged_unweighted_s43 | rank_fraction | -0.002741 | -0.013888 | 0.003080 | -0.001983 | 0.000894 |
| randomized minus unbagged | bagged_unweighted_logit_mean | pooled | baseline_lgbm | pooled | -0.001538 | -0.022727 | 0.012522 | -0.004008 | 0.003893 |
| aggregate minus member | bagged_unweighted_logit_mean | pooled | bagged_unweighted_s17 | pooled | 0.004460 | -0.008247 | 0.005226 | -0.003208 | 0.003621 |
| aggregate minus member | bagged_unweighted_logit_mean | pooled | bagged_unweighted_s29 | pooled | -0.002505 | 0.020987 | -0.010489 | 0.004943 | 0.000919 |
| aggregate minus member | bagged_unweighted_logit_mean | pooled | bagged_unweighted_s43 | pooled | 0.003040 | 0.001696 | 0.003351 | -0.000528 | 0.001001 |
| randomized minus unbagged | bagged_unweighted_logit_mean | latest | baseline_lgbm | latest | -0.006413 | -0.005140 | 0.004758 | 0.001382 | 0.003893 |
| aggregate minus member | bagged_unweighted_logit_mean | latest | bagged_unweighted_s17 | latest | -0.003717 | 0.017270 | -0.007111 | 0.004960 | 0.003621 |
| aggregate minus member | bagged_unweighted_logit_mean | latest | bagged_unweighted_s29 | latest | -0.005452 | 0.000642 | -0.005053 | 0.001832 | 0.000919 |
| aggregate minus member | bagged_unweighted_logit_mean | latest | bagged_unweighted_s43 | latest | 0.000224 | 0.001006 | 0.001426 | 0.000276 | 0.001001 |
| randomized minus unbagged | bagged_unweighted_logit_mean | rank_fraction | baseline_lgbm | rank_fraction | 0.007859 | -0.014931 | 0.011612 | -0.004931 | 0.003893 |
| aggregate minus member | bagged_unweighted_logit_mean | rank_fraction | bagged_unweighted_s17 | rank_fraction | 0.004567 | -0.006112 | 0.007129 | -0.002629 | 0.003621 |
| aggregate minus member | bagged_unweighted_logit_mean | rank_fraction | bagged_unweighted_s29 | rank_fraction | 0.006945 | 0.029080 | -0.007266 | 0.004053 | 0.000919 |
| aggregate minus member | bagged_unweighted_logit_mean | rank_fraction | bagged_unweighted_s43 | rank_fraction | -0.001297 | -0.004856 | 0.000547 | -0.000577 | 0.001001 |
| logit mean minus probability mean | bagged_balanced_logit_mean | pooled | bagged_balanced_prob_mean | pooled | 0.000187 | -0.001063 | 0.000884 | -0.000264 | -0.000052 |
| logit mean minus probability mean | bagged_balanced_logit_mean | latest | bagged_balanced_prob_mean | latest | -0.000632 | -0.001753 | 0.000306 | -0.000162 | -0.000052 |
| logit mean minus probability mean | bagged_balanced_logit_mean | rank_fraction | bagged_balanced_prob_mean | rank_fraction | 0.000326 | -0.000699 | 0.000621 | -0.000226 | -0.000052 |
| logit mean minus probability mean | bagged_unweighted_logit_mean | pooled | bagged_unweighted_prob_mean | pooled | 0.002173 | 0.010785 | -0.002065 | 0.001623 | 0.000107 |
| logit mean minus probability mean | bagged_unweighted_logit_mean | latest | bagged_unweighted_prob_mean | latest | 0.000702 | 0.004233 | -0.001129 | 0.000791 | 0.000107 |
| logit mean minus probability mean | bagged_unweighted_logit_mean | rank_fraction | bagged_unweighted_prob_mean | rank_fraction | 0.001444 | 0.009032 | -0.002534 | 0.001406 | 0.000107 |
| balanced minus unweighted | bagged_balanced_s17 | pooled | bagged_unweighted_s17 | pooled | 0.005888 | -0.013650 | 0.007716 | -0.004642 | 0.002497 |
| balanced minus unweighted | bagged_balanced_s17 | latest | bagged_unweighted_s17 | latest | 0.004761 | -0.007098 | 0.005503 | -0.002849 | 0.002497 |
| balanced minus unweighted | bagged_balanced_s17 | rank_fraction | bagged_unweighted_s17 | rank_fraction | 0.003673 | -0.004043 | 0.005731 | -0.002007 | 0.002497 |
| balanced minus unweighted | bagged_balanced_s29 | pooled | bagged_unweighted_s29 | pooled | 0.002360 | 0.015765 | -0.007316 | 0.002383 | -0.000489 |
| balanced minus unweighted | bagged_balanced_s29 | latest | bagged_unweighted_s29 | latest | 0.003589 | -0.019063 | 0.003815 | -0.005313 | -0.000489 |
| balanced minus unweighted | bagged_balanced_s29 | rank_fraction | bagged_unweighted_s29 | rank_fraction | 0.003437 | 0.031713 | -0.010108 | 0.005371 | -0.000489 |
| balanced minus unweighted | bagged_balanced_s43 | pooled | bagged_unweighted_s43 | pooled | 0.007516 | -0.013698 | 0.012477 | -0.004935 | -0.000448 |
| balanced minus unweighted | bagged_balanced_s43 | latest | bagged_unweighted_s43 | latest | 0.008491 | -0.017146 | 0.011708 | -0.006183 | -0.000448 |
| balanced minus unweighted | bagged_balanced_s43 | rank_fraction | bagged_unweighted_s43 | rank_fraction | -0.008344 | -0.023850 | 0.003188 | -0.002567 | -0.000448 |
| balanced minus unweighted | bagged_balanced_prob_mean | pooled | bagged_unweighted_prob_mean | pooled | 0.007614 | 0.002795 | 0.003557 | -0.001577 | -0.000223 |
| balanced minus unweighted | bagged_balanced_prob_mean | latest | bagged_unweighted_prob_mean | latest | 0.012523 | -0.011688 | 0.009931 | -0.006310 | -0.000223 |
| balanced minus unweighted | bagged_balanced_prob_mean | rank_fraction | bagged_unweighted_prob_mean | rank_fraction | 0.000465 | 0.006072 | -0.000423 | 0.000931 | -0.000223 |
| balanced minus unweighted | bagged_balanced_logit_mean | pooled | bagged_unweighted_logit_mean | pooled | 0.005627 | -0.009053 | 0.006506 | -0.003463 | -0.000381 |
| balanced minus unweighted | bagged_balanced_logit_mean | latest | bagged_unweighted_logit_mean | latest | 0.011189 | -0.017674 | 0.011366 | -0.007262 | -0.000381 |
| balanced minus unweighted | bagged_balanced_logit_mean | rank_fraction | bagged_unweighted_logit_mean | rank_fraction | -0.000653 | -0.003660 | 0.002733 | -0.000701 | -0.000381 |
