# Robust Bank Account Fraud Detection under Temporal Drift and Class Imbalance

BT4012 Fraud Analytics, AY2026/27 Semester 1 · Group XX
Members: _add names_

We study how **training-window design**, **class-imbalance handling** and **model choice** affect
**out-of-time** fraud detection on the Bank Account Fraud (BAF) dataset suite (Jesus et al., NeurIPS 2022),
and how the resulting scores would be used in a bank's application-review process.

* Results in plain language: [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md)
* Which lecture method and rubric item each part covers: [`docs/lecture_rubric_checklist.md`](docs/lecture_rubric_checklist.md)

## Research questions → experiments

| Question | Experiment | Lecture | Main outputs |
|---|---|---|---|
| What does the data look like, and does it drift over time? | **E0** `e0_eda_drift.py` | L2 (+ PSI, beyond) | `e0_variable_summary.csv`, `e0_monthly.csv`, `e0_covariate_drift.csv`, `e0_concept_drift.csv` |
| Which variables relate to fraud? Outliers, robust statistics, Benford's law | **E8** `e8_statistical_eda.py` | L2 | `e8_univariate.csv`, `e8_bivariate_tests.csv`, `e8_outliers.csv`, `e8_benford.csv` |
| Do tree models beat an interpretable logistic-regression baseline (and a no-skill dummy)? | **E1** `e1_model_comparison.py` | L3 | `e1_summary.csv`, `e1_per_month.csv`, `e1_pr_curves.png` |
| How strong should the penalty be? L1 vs L2, under- vs overfitting | **E9** `e9_regularisation.py` | L3 | `e9_regularisation_path.csv`, `e9_regularisation.png`, `e9_lasso_path.png` |
| Do neural networks help? Supervised MLP vs unsupervised autoencoder | **E10** `e10_neural_networks.py` | L4 | `e10_summary.csv`, `e10_roc.png`, `e10_autoencoder_errors.png` |
| How optimistic is a random train/test split? | **E2** `e2_random_vs_temporal.py` | L3 (data splits) | `e2_summary.csv`, `e2_split_gap.png` |
| Does a recent rolling window beat expanding history? | **E3** `e3_training_window.py` | proposal RQ1 | `e3_summary.csv`, `e3_windows.png` |
| At the SAME training size K, does recent data beat old data? | **E3b** `e3b_recency_fixed_k.py` | TA feedback | `e3b_summary.csv`, `e3b_learning_curve.png`, `e3b_recency_effect.png` |
| Which imbalance treatment is best? | **E4** `e4_imbalance.py` | L3 (accuracy fails on imbalance) | `e4_summary.csv`, `e4_imbalance.png` |
| How should the threshold be chosen (capacity, cost, fairness)? | **E5** `e5_operations.py` | L3 (confusion matrix, cost table) | `e5_capacity.csv`, `e5_costs.csv`, `e5_lecture_cost_table.csv`, `e5_fairness.csv` |
| What drives predictions, and where do models fail? | **E6** `e6_interpretability.py` | L3 (coefficients) + SHAP | `e6_logreg_coefficients.csv`, `e6_shap_importance.csv`, `e6_error_profiles.csv` |
| Does feature engineering help, and for which model? | **E7** `e7_feature_ablation.py` | TA feedback | `e7_summary.csv`, `e7_feature_ablation.png` |
| Can a fraudster dodge the model by changing form fields? Which defence works? | **E11** `e11_adversarial.py` | L7 | `e11_evasion.csv`, `e11_defences.csv`, `e11_evasion.png` |
| How would the score be used by investigators? | **E12** `e12_case_triage.py` | L5, L9, L10, L11 | `e12_queue_summary.csv`, `e12_case_briefs_sample.jsonl`, `e12_case_brief_schema.json` |

All tables are written to `results/` and all figures to `figures/`. `full_run.log` is the log of
the run whose numbers are reported.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Download the BAF suite from Kaggle and put **`Base.csv`** in `data/`:
<https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022>
(official repository: <https://github.com/feedzai/bank-account-fraud>).
The data is not committed to this repository.

## How to reproduce

```bash
python run_all.py --data data/Base.csv                  # full run, all experiments (about 1.5-2 h on 20 threads)
python run_all.py --data data/Base.csv --quick          # 10% sample, 1 seed, smaller models (minutes)
python run_all.py --data data/Base.csv --only e8 e9     # selected experiments (E1 must have run first, except E0/E8)
python run_all.py --data data/Base.csv --seeds 0        # one seed: faster, no seed variance
python run_all.py --data data/Base.csv --features raw   # E1-E6 on raw columns only (default: engineered)
python -m tests.test_nn                                 # gradient check for the hand-written neural network
```

`results/run_info.json` records the settings, data size, cleaning log and library versions;
`results/timings.json` records how long each experiment took. Every random step is seeded.

**Smoke test without the real data:** `python tests/make_fake_baf.py` writes a fake file with the BAF
schema, then `python run_all.py --data data/fake_baf.csv --quick --sample-frac 1` runs everything
in about a minute. Never report numbers from the fake file.

## Design decisions

* **Out-of-time evaluation.** Following the BAF protocol, months 6–7 are the test set. Month 5 is
  the validation month, used only to choose hyperparameters, the imbalance strategy and decision
  thresholds; the test months are touched once, at the end (Lecture 3, slide 38: tune on validation, never on test).
* **No leakage.** `month` is not a model feature. All preprocessing (median imputation with missing
  indicators, scaling, one-hot encoding, quantile bins) lives inside the model pipeline and is fitted on
  training rows only. Cleaning (negative sentinels → missing, dropping the constant `device_fraud_count`)
  does not use the label. Every label-based EDA statistic (E8) uses the training months only.
* **Metrics for rare events (Lecture 3).** Accuracy is never used to choose anything (flagging nobody is
  ~99% accurate). We report PR-AUC, ROC-AUC, the BAF benchmark **recall at 5% FPR**, precision and recall
  at a threshold fixed on validation, confusion matrices, cost tables, and precision/recall when analysts
  can review only the top 1/2/5% of applications. Uncertainty: mean ± std over 3 seeds and 95% bootstrap CIs.
* **Models.** A constant-score dummy and logistic regression (L2/L1, Lecture 3) are the simple benchmarks;
  a feedforward network and an autoencoder follow Lecture 4; random forest and LightGBM go beyond the
  lectures because gradient-boosted trees are the standard strong baseline for tabular fraud data and the
  BAF paper's reference model.
* **Imbalance (Lecture 3 + E4).** For each model, the strategy (none / class weighting / undersampling to
  10 legitimate per fraud) is chosen on the validation month together with the hyperparameters.
* **Feature engineering differs by model (`src/features.py`, tested in E7).** All models get row-wise
  features (velocity ratios, number of empty optional fields, previous-address flag, total months at both
  addresses, number of valid phones). Logistic regression also gets log1p of skewed counts, one-hot age and
  income levels, and quantile bins for `name_email_similarity` and `credit_risk_score`. Trees do not need
  monotone transforms.
* **Recency vs data size (E3, E3b).** A rolling window is both more recent *and* smaller than expanding
  history. E3b fixes the training size K and compares the K most recent applications with the K oldest
  ones and K random ones, as the TA suggested.
* **Costs are assumptions.** E5 reports several missed-fraud : false-alarm cost ratios (10, 25, 50, 100)
  as a sensitivity analysis.
* **Fairness.** BAF was built to study bias. E5 reports the false-positive rate for applicants under 50 and
  50+, and refits the main model without `customer_age` to see whether the gap remains.
* **Neural network in NumPy (`src/nn.py`).** Written by hand so the Lecture 4 steps (ReLU, mini-batches,
  backpropagation, dropout scaled by 1 − p at inference) are visible; `tests/test_nn.py` checks the
  gradients numerically.

## Teammates' notebooks

The branch `feat--zs-exploration-00` holds a standalone notebook workflow (EDA, data preparation and a
dummy / logistic-regression / LightGBM baseline). It uses the same split (train 0–4, validation 5,
test 6–7) and the same six missing-value sentinel columns as `src/config.py`, so its validation numbers
are directly comparable with E1.

## Project structure

```
run_all.py              entry point
src/config.py           all settings (months, seeds, operating points, model sizes)
src/data.py             loading, cleaning, splits, undersampling
src/features.py         engineered features (all models) and logistic-regression transforms
src/models.py           logistic regression, random forest, LightGBM, imbalance strategies
src/nn.py               feedforward network / autoencoder in NumPy (Lecture 4)
src/metrics.py          recall@FPR, capacity metrics, (paired) bootstrap CIs, costs, group fairness
src/utils.py            logging, plotting, json helpers
experiments/e0..e12     one file per experiment
tests/                  fake BAF data for smoke tests; neural-network gradient check
docs/                   results summary, lecture/rubric checklist, optional LLM prompt design
```

## Limitations

The BAF data is synthetic (generated from a real bank's data with privacy protection), so drift
patterns and feature effects may not match any particular bank. It covers only 8 months, which limits
how many test months the window study can use, and has no timestamps inside a month. There are no
links between applications (so no network analysis, Lecture 8), no document images (Lecture 7 image
checks) and no investigation outcomes. Cost ratios are assumptions. The adversarial study assumes which
fields a fraudster controls.

## Acknowledgements and AI declaration

Dataset: S. Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML
Evaluation*, NeurIPS 2022 Datasets and Benchmarks.

Generative AI (Anthropic Claude, via Claude Code) was used to help write and run this pipeline and to
draft documentation. Teammates' notebooks on `feat--zs-exploration-00` declare ChatGPT/Codex assistance.
All members reviewed the code and can explain every component.
