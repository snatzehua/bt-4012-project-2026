# Robust Bank Account Fraud Detection under Temporal Drift and Class Imbalance

BT4012 Fraud Analytics, AY2026/27 Semester 1 · Group 8
Members: Travis Tan Wei Xuan, Wang Jun Feng, Teh Ze Shi, Kieran Ho Cheng Hong

We study how **training-window design**, **class-imbalance handling** and **model choice** affect
**out-of-time** fraud detection on the Bank Account Fraud (BAF) dataset suite (Jesus et al., NeurIPS 2022).

## Research questions → experiments

| Research question | Experiment | Main outputs |
|---|---|---|
| What does the data look like, and does it drift over time? | **E0** `e0_eda_drift.py` | `e0_variable_summary.csv`, `e0_monthly.csv`, `e0_covariate_drift.csv`, `e0_concept_drift.csv` |
| Do tree models beat an interpretable logistic-regression baseline? | **E1** `e1_model_comparison.py` | `e1_summary.csv`, `e1_per_month.csv`, `e1_pr_curves.png` |
| How optimistic is a random train/test split? | **E2** `e2_random_vs_temporal.py` | `e2_summary.csv`, `e2_split_gap.png` |
| Does a recent rolling window beat expanding history? Is it recency or just data size? | **E3** `e3_training_window.py` | `e3_summary.csv`, `e3_windows.png`, `e3_recency_vs_size.png` |
| Which imbalance treatment gives the best recall / false-positive trade-off? | **E4** `e4_imbalance.py` | `e4_summary.csv`, `e4_imbalance.png` |
| How should the operating threshold be chosen in practice (capacity, cost, fairness)? | **E5** `e5_operations.py` | `e5_capacity.csv`, `e5_costs.csv`, `e5_fpr_stability.csv`, `e5_fairness.csv` |
| What drives predictions, and where do the models fail? | **E6** `e6_interpretability.py` | coefficients, gain importance, SHAP, `e6_error_profiles.csv` |

Tables are written to `results/<dataset>/` and figures to `figures/<dataset>/` (e.g. `results/Base/`), so runs on different BAF variants never overwrite each other.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

Download the BAF suite from Kaggle and put the csv files in `data/`:
<https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022>
(official repository: <https://github.com/feedzai/bank-account-fraud>). We use **`Base.csv`**.
The data is not committed to this repository.

## How to reproduce

```bash
python run_all.py --data data/Base.csv                 # full run, all experiments
python run_all.py --data data/Base.csv --quick         # 10% sample, 1 seed, smaller models (minutes)
python run_all.py --data data/Base.csv --only e3 e4    # selected experiments (E1 must have run first)
python run_all.py --data data/Base.csv --seeds 0       # one seed: much faster, no seed variance
python run_all.py --data "data/Variant I.csv" --only e1 e2   # optional robustness check on another variant
```

`results/run_info.json` records the settings, data size, cleaning log and library versions of each run;
`results/timings.json` records how long each experiment took. Every random step is seeded.

**Runtime.** The full run fits many models on ~1M rows. On a typical laptop expect roughly one to
a few hours, most of it in E3 (training-window refits) and the random forest in E1. Use `--seeds 0`
or `--sample-frac 0.5` if time is short, and say so in the report.

**Smoke test without the real data:** `python tests/make_fake_baf.py` writes a fake file with the BAF
schema, then `python run_all.py --data data/fake_baf.csv --quick --sample-frac 1` runs everything
in about a minute. Never report numbers from the fake file.

## Design decisions

* **Out-of-time evaluation.** Following the BAF protocol, months 6–7 are the test set. Month 5 is
  held out as a validation month, used only to tune a small hyperparameter grid and to choose decision
  thresholds; the test months are touched once, at the end.
* **No leakage.** `month` is not a model feature. All preprocessing (median imputation with missing
  indicators, scaling, one-hot encoding) lives inside the model pipeline and is fitted on training rows
  only. Cleaning (negative sentinels → missing, dropping constant columns such as `device_fraud_count`)
  does not use the label.
* **Metrics for rare events.** Accuracy is never used. We report PR-AUC, ROC-AUC, the BAF benchmark
  metric **recall at 5% FPR**, precision/recall at a threshold fixed on validation (what a bank could
  actually deploy), and precision/recall when analysts can review only the top 1/2/5% of applications.
  Uncertainty: mean ± std over seeds and 95% bootstrap confidence intervals on the test set.
* **Recency vs data size (E3).** A rolling window is both more recent *and* smaller than expanding
  history. We add a size-matched expanding baseline so the recency effect can be separated from
  the effect of having less data. Comparisons are paired on the same test month and seed.
* **Costs are assumptions.** E5 reports results for several missed-fraud : false-alarm cost ratios
  (10, 25, 50, 100) as a sensitivity analysis instead of a single made-up cost.
* **Fairness.** BAF was built to study bias. E5 reports the false-positive rate for applicants under
  50 and 50+, and the predictive-equality ratio (lower FPR / higher FPR; 1 = equal burden).

## Project structure

```
run_all.py              entry point
src/config.py           all settings (months, seeds, operating points, model sizes)
src/data.py             loading, cleaning, splits, undersampling
src/models.py           logistic regression, random forest, LightGBM, imbalance strategies
src/metrics.py          recall@FPR, capacity metrics, bootstrap CIs, costs, group fairness
src/utils.py            logging, plotting, json helpers
experiments/e0..e6      one file per experiment
tests/make_fake_baf.py  fake BAF-shaped data for smoke tests only
```

## Limitations

The BAF data is synthetic (generated from a real bank's data with privacy protection), so drift
patterns and feature effects may not match any particular bank. It covers only 8 months, which limits
how many test months the window study can use. Cost ratios are assumptions.

## Acknowledgements

Dataset: S. Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML
Evaluation*, NeurIPS 2022 Datasets and Benchmarks.
