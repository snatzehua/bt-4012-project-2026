# Submission branch audit

Audit date: 18 November 2026. This is a read-only scientific comparison of saved source and results, not a new model run. Codex assisted with the audit. No historical model, score, selection or frozen source was changed.

## Pinned scope

Repository: `https://github.com/snatzehua/bt-4012-project-2026`.

| Workstream | Exact commit | Role |
|---|---|---|
| Local checkpoint | `43fb021258e342f3ae97e5a34ab74cc1171c8371` | Canonical chronological/matched-K study evidence |
| `feat--zs-exploration-00` | `d466ff34df6f62abe935b993ef75c8ca0dd3b1ed` | Notebook experiments and auxiliary operational evaluation |
| `lecture-aligned-pipeline` | `1f4f249e6b055da28995290d1a43009b8cce98c6` | Auxiliary explanations, fairness, costs and threat-model analysis |
| `Travis` | `248c6e8113aad8ee375a1f014d92d023e4e376ab` | No modelling code/results in the inspected tree |
| Remote `main` | `34bac13798dfa093c5774c81ba5cebf69bd80c3b` | Proposal and feedback |

The earlier description of ZS as validation-only is **superseded**. Its latest tree includes executed notebooks 06–11: temporal stacking, domain adaptation, averaging, development monthly updates, months-6/7 evaluation and results synthesis. Its updated README explicitly discloses prior team exposure to months 6–7.

For source retrieval, replace `<commit>` and `<path>` in `https://github.com/snatzehua/bt-4012-project-2026/blob/<commit>/<path>` with a pin and exact path below. The GitHub REST tree endpoint is `/repos/snatzehua/bt-4012-project-2026/git/trees/<commit>?recursive=1`.

## Recommended consolidation

Use the local frozen studies as the scientific core, especially exact size/class-matched historical controls, earlier-only preprocessing and threshold learning, preserved selections and reproducibility evidence. Add submission export/checking/documentation around those studies; do not silently alter frozen recipes.

Use ZS as a separately labelled auxiliary study for operational updating. Use the lecture branch selectively for explanation, fairness, cost sensitivity and the proposed review workflow. Do not create a unified model leaderboard: training periods, evaluation populations, feature representations, imbalance treatments, objectives and threshold policies differ.

No new full fits are required merely to compile the report. Completing an unfinished local experiment is different from packaging completed evidence; exclude incomplete context-feature-ablation results from performance claims.

## Updated ZS evidence

Source: `notebooks/10_temporal_holdout_evaluation.ipynb` at the ZS pin, executed outputs in cells 18, 20 and 22 (zero-based notebook cell indices). `notebooks/11_results_synthesis_and_conclusions.ipynb` supplies the selected source-run inventory.

**Evidence qualification:** the values below were read from embedded executed notebook outputs, not independently regenerated here. CSV/JSON run exports and row-level predictions are referenced in Google Drive and are not committed in this tree. Independent inference/metric reproduction is therefore not claimed. Notebook source and embedded-output provenance must accompany any adopted table.

Static model training uses months 0–4, calibration month 5 and evaluation months 6–7. The CatBoost recipe uses all eligible history before 10:1 legitimate/fraud undersampling, depth 4, 600 iterations, learning rate 0.05 and equal averaging across seeds 42/43/44. Scores are not established as calibrated probabilities.

| Metric | Full-data LightGBM seed 42 | RUS CatBoost seed 42 | RUS CatBoost mean of 3 |
|---|---:|---:|---:|
| Month-6 AP | 0.178885 | 0.179962 | 0.183983 |
| Month-7 AP | 0.210860 | 0.220854 | 0.225427 |
| Pooled months-6/7 AP | 0.193286 | 0.198463 | 0.202266 |
| Monthly exact 1% reviews, summed | 2049 | 2049 | 2049 |
| Fraud caught at that budget | 607 | 624 | 633 |
| Precision at that budget | 0.296242 | 0.304539 | 0.308931 |
| Recall at that budget | 0.210910 | 0.216817 | 0.219944 |

Exact-budget counts use `floor(q*N)` independently per month; local first-study capacity uses `ceil(q*N)`. The summed budget is not a single pooled ranking over both months.

The refresh policy trains through month 5 and calibrates on month 6 before predicting month 7. Frozen and threshold-only policies share scores.

| Month-7 policy | AP | Deployed recall | Deployed FPR | TP | FP |
|---|---:|---:|---:|---:|---:|
| Frozen | 0.225427 | 0.593838 | 0.053786 | 848 | 5132 |
| Threshold only | 0.225427 | 0.541317 | 0.041304 | 773 | 3941 |
| Retrain + threshold | 0.226676 | 0.555322 | 0.043746 | 793 | 4174 |

Across months 6–7, refresh catches 635 versus 633 frauds at the same summed monthly 1% budget. This is a small descriptive difference, not a demonstrated general retraining benefit. Threshold-only decisions change workload, not ranking/AP or exact-capacity capture. Assumed preceding-month label availability, repeated month-5 selection, prior later-period exposure, one new policy update and lack of formal gain uncertainty limit interpretation.

Notebooks 09/10 clearly separate fit, calibration and prediction periods. Notebook 09 explicitly acknowledges retrospective recipe selection using month 5, so its earlier-month backtest is not an unbiased historical deployment estimate. Notebook 10 records month-7 cutoffs before loading/scoring month 7. These are strengths worth retaining in the operational explanation.

## Lecture branch issues requiring correction before adoption

| Source | Finding | Submission treatment |
|---|---|---|
| `src/data.py`, `clean()` | Constant exclusion and categorical vocabulary are derived across the entire input before temporal splitting. Absence of label use does not imply absence of future covariate information. | Do not import this into canonical earlier-only preprocessing; disclose the auxiliary scope. |
| `src/metrics.py`, `recall_at_fpr()` | Evaluation labels locate the evaluation-period FPR cutoff. | Label retrospective benchmark separately from deployed cutoff decisions. |
| `experiments/e5_operations.py` and `results/e5_capacity.csv` | Nominal 1% capacity is a transferred validation quantile; actual review share is 1.27798%. | Describe as a past-selected score cutoff, not an exact 1% review budget. |
| `experiments/e7_feature_ablation.py` | Displayed raw/engineered values are seed means; `gain_*` and paired intervals are seed-0 differences. | Explain the different estimands; the nonreconciling columns are not sufficient evidence of a calculation failure. |
| `experiments/e3b_recency_fixed_k.py` | t-tests pool month/seed differences despite overlapping evaluation cases and training samples. | Avoid borrowing significance language as independent evidence of a small-K recency effect. |
| `experiments/e1_model_comparison.py` | Dummy operating metrics are hardcoded randomised expectations; bootstrap intervals apply to seed 0 beside seed-mean results. | Label expected random reference and interval scope explicitly. |
| `experiments/e1_model_comparison.py` | Model hyperparameters and imbalance treatment are jointly selected. | Call this a selected-pipeline comparison, not an isolated architecture effect. |
| `experiments/e11_adversarial.py` | Robust-feature evasion is zero by construction after declared controllable fields are removed. | Limit the claim to that threat model; do not infer general adversarial robustness. |

Dependencies in the lecture branch are minimum-version constraints rather than a strict lockfile. The local pinned environment and historical clean-copy receipts provide stronger reproduction evidence for the canonical studies. The lecture structure and lecture/rubric mapping remain useful for a concise report, but breadth of experiments alone does not establish better detection.

## Reporting decisions

1. Present the local controlled recency study and selected completed improvement studies as the core narrative.
2. Distinguish fixed-capacity results from F2 threshold policies; the latter have no imposed 1% review quota.
3. Use remote operational evidence only with its pinned source, split and embedded-output qualification.
4. State that month 7 is already consumed and that adaptive follow-up results are exploratory.
5. Preserve negative results and the stronger unweighted consumed-month reference rather than choosing a winner retrospectively from month 7.
6. Acknowledge generative-AI assistance and attribute the remote workstreams accurately; contributors must review and explain adopted work.
