# BT4012 · Bank Account Fraud under Temporal Drift and Class Imbalance

This document describes the notebook workflow on branch **`feat--zs-exploration-00`** of [`snatzehua/bt-4012-project-2026`](https://github.com/snatzehua/bt-4012-project-2026/tree/feat--zs-exploration-00).

The branch investigates how training-data composition, class-imbalance treatment, model family, stacking, temporal reweighting, prediction averaging and monthly updating affect bank account fraud detection. It contains exploratory analysis, shared data preparation, development experiments, later-period evaluation and results synthesis.

**One row is a bank account opening application, not a payment transaction.** The proposed business use is to rank applications for manual investigation before approval. The experiments measure how many labelled fraudulent applications would enter that review queue, alongside legitimate applications that create workload and customer friction. They do not establish actual fraud prevention, financial savings or the correctness of automatically rejecting an applicant.

The implementation is notebook-based. The notebooks do not require a `src/` package or a deployed service. They can run in Google Colab or a local Jupyter kernel, including through VS Code.

## Contents

1. [Current scope and notebook map](#1-current-scope-and-notebook-map)
2. [Dataset and evaluation protocol](#2-dataset-and-evaluation-protocol)
3. [Repository and data locations](#3-repository-and-data-locations)
4. [Google Colab setup](#4-google-colab-setup)
5. [Local setup](#5-local-setup)
6. [What to run and dependencies](#6-what-to-run-and-dependencies)
7. [Common preparation and model-specific preprocessing](#7-common-preparation-and-model-specific-preprocessing)
8. [Experimental design](#8-experimental-design)
9. [Metrics and operational interpretation](#9-metrics-and-operational-interpretation)
10. [Outputs and run selection](#10-outputs-and-run-selection)
11. [Findings and reporting boundaries](#11-findings-and-reporting-boundaries)
12. [Reproducibility and collaboration](#12-reproducibility-and-collaboration)
13. [Troubleshooting](#13-troubleshooting)
14. [References and acknowledgement](#14-references-and-acknowledgement)

## 1. Current scope and notebook map

The branch contains the following notebooks under `notebooks/`. The numbers describe the experimental narrative; they do not imply that every notebook loads the previous notebook's fitted models.

| Notebook | Question or purpose | Main input |
|---|---|---|
| [00_eda_zs.ipynb](notebooks/00_eda_zs.ipynb) | Inspect schema, missingness, class imbalance and monthly variation | `Base.csv` |
| [01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb) | Apply common preparation and export a reproducible feature contract | `Base.csv` |
| [02_baseline_model_comparison.ipynb](notebooks/02_baseline_model_comparison.ipynb) | Establish dummy, logistic regression and LightGBM references | Prepared data and manifest |
| [03_training_data_composition.ipynb](notebooks/03_training_data_composition.ipynb) | Compare all history, recent K and matched-size historical K | Prepared data and manifest |
| [04_class_imbalance_treatment.ipynb](notebooks/04_class_imbalance_treatment.ipynb) | Compare untreated training, class weights and undersampling | Prepared data and manifest |
| [05_model_family_comparison.ipynb](notebooks/05_model_family_comparison.ipynb) | Compare bounded model configurations and treatment transfer | Prepared data and manifest |
| [06_temporal_stacking.ipynb](notebooks/06_temporal_stacking.ipynb) | Test a meta-model trained using forward-in-time predictions | Prepared data and manifest |
| [07_temporal_domain_adaptation.ipynb](notebooks/07_temporal_domain_adaptation.ipynb) | Test whether historical rows should be reweighted toward a recent period | Prepared data and manifest |
| [08_ensemble_averaging_exploration.ipynb](notebooks/08_ensemble_averaging_exploration.ipynb) | Compare fixed averages across seeds and model families | Notebook 05's saved predictions and manifests |
| [09_monthly_retraining_exploration.ipynb](notebooks/09_monthly_retraining_exploration.ipynb) | Compare frozen, threshold-only and retraining policies on development months | Prepared data and manifest |
| [10_temporal_holdout_evaluation.ipynb](notebooks/10_temporal_holdout_evaluation.ipynb) | Evaluate selected models and policies on months 6–7 | Prepared data and manifest |
| [11_results_synthesis_and_conclusions.ipynb](notebooks/11_results_synthesis_and_conclusions.ipynb) | Assemble saved findings, figures and reporting limitations | Results CSVs and JSON manifests from earlier stages |

Start with notebook **11** to understand the completed experimental story, then inspect the relevant earlier notebook for implementation and evidence. Embedded notebook outputs are useful for review, but regenerating the synthesis requires the exported CSV/JSON files as well.

The branch also contains `BT4012_Proposal_BAF_Revised.pdf`, `feedback.md` and this guide. It does not implement a production application, online decision API or automatic retraining service.

## 2. Dataset and evaluation protocol

We use the **Base** variant of the Bank Account Fraud Dataset Suite. Obtain `Base.csv` from the [dataset download page](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022); consult the [official repository](https://github.com/feedzai/bank-account-fraud) for dataset documentation.

The inspected file contains **1,000,000 applications and 32 source columns**, including the target and month. Its predictors describe application, applicant, account, device and behavioural information. Fraud is rare, so accuracy alone is not an informative objective.

The principal development split is chronological:

| Role | Month indices | Applications | Fraud-labelled applications |
|---|---|---:|---:|
| Training | 0–4 | 675,666 | 6,740 |
| Validation | 5 | 119,323 | 1,411 |
| Later-period evaluation | 6–7 | 205,011 | 2,878 |

These counts describe the inspected Base file. Preparation records the counts for the actual input used. Month indices are not named calendar months, and CSV row order does not establish ordering within a month.

`fraud_bool` is the target, `month` is used to define temporal boundaries, and preparation adds `row_id` to align records and predictions. None is a model predictor. In the inspected data, the training-constant feature `device_fraud_count` is excluded from the predictor list, leaving 29 predictors. The manifest is authoritative for the actual ordered feature list.

Development notebooks 03–08 compare configurations on month 5. Notebook 09 uses earlier monthly origins within the development period. Notebook 10 has a separately specified static comparison and sequential updating schedule, described below.

**Prior exposure must be disclosed:** another team member has already examined months 6–7. The later evaluation is therefore a fixed-protocol evaluation on a reserved temporal period with prior team exposure, not a wholly unseen team-wide test. Repeated month-5 selection also makes development performance optimistic.

## 3. Repository and data locations

### Tracked files versus generated files

| Location | Role | Normally committed? |
|---|---|---|
| `notebooks/` | Notebook code, explanations and selected execution outputs | Yes |
| `ZS_BRANCH_README.md` | Branch setup and experiment guide | Yes |
| Proposal PDF and `feedback.md` | Project scope and feedback | Yes |
| `data/Base.csv` | Raw local dataset | No |
| `data/processed/` | Local prepared data and its manifest | No; preserve together outside Git |
| `results/<stage>/<run>/` | Summary tables, figures and provenance records | Selected small files can be committed |
| `artifacts/<stage>/<run>/` | Models, row-level predictions and sampled row IDs | No |

Generated directories need not appear on GitHub. Colab does not automatically copy its Drive outputs into the repository. There is currently no required `src/` package. The installation command below does not assume a `requirements.txt` is already committed.

### Google Drive

Use the following locations under `MyDrive/BT4012_grp_proj`:

| Item | Drive-relative path |
|---|---|
| Raw dataset | `BT4012_grp_proj/Base.csv` |
| Prepared data | `BT4012_grp_proj/processed/baf_clean.parquet` |
| Preparation manifest | `BT4012_grp_proj/processed/preparation_manifest.json` |
| Experiment summaries | `BT4012_grp_proj/results/<stage>/<run>/` |
| Model and prediction artifacts | `BT4012_grp_proj/artifacts/<stage>/<run>/` |

The modelling notebooks use:

```python
DRIVE_PROJECT_DIR = Path('/content/drive/MyDrive/BT4012_grp_proj')
```

The notebooks themselves may be stored elsewhere. The mounted project directory determines where inputs are read and outputs are saved. If using a shared folder, change this path in each notebook you run. Sharing a notebook does not automatically share its data or output folders.

### Local repository

Keep `Base.csv` at `<repo>/data/Base.csv`. Notebook 01 writes the prepared pair to `<repo>/data/processed/`. Results and artifacts are written directly under `<repo>/results/` and `<repo>/artifacts/`.

The modelling notebooks search the working directory and its parents for a directory containing both `notebooks/` and `data/`. If detection fails, set:

```python
LOCAL_REPO_ROOT = '/absolute/path/to/bt-4012-project-2026'
```

For Windows:

```python
LOCAL_REPO_ROOT = r'C:\Users\your_name\projects\bt-4012-project-2026'
```

Notebook 11 does not read raw data, but its shared local root-detection code still checks for a `data/` directory. Set `LOCAL_REPO_ROOT` explicitly when using a results-only copy without that directory.

## 4. Google Colab setup

1. Place the genuine Base dataset at `MyDrive/BT4012_grp_proj/Base.csv`.
2. Open the notebook from this branch or upload the downloaded `.ipynb` to Colab. Ensure you are opening the feature-branch version rather than the default branch.
3. Use a **CPU runtime**. The supplied model configurations do not use GPU acceleration.
4. Check the project path in the configuration cell and authorize Drive mounting.
5. Run notebook 01 to completion if the prepared Parquet/JSON pair does not already exist.
6. Open the experiment you need, check its configuration, then run from the top in a fresh session.
7. Keep the printed run directory and inspect warnings and final completion status.
8. Save the executed notebook. Copy selected small exports to the repository deliberately when preparing a commit.

The modelling notebooks install missing specialist dependencies in Colab. Their presence checks do not necessarily upgrade an older installed version. If an API error indicates an incompatible version, install compatible packages and restart the runtime before rerunning.

An already-open Colab notebook does not update when its GitHub copy changes. Reopen or replace it when adopting revised code. Avoid changing settings halfway through an execution and then continuing with stale variables.

## 5. Local setup

From the repository root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the packages used across the notebook workflow:

```bash
python -m pip install numpy pandas "scikit-learn>=1.3" matplotlib seaborn joblib "pyarrow>=14" "lightgbm>=4" "catboost>=1.2" "xgboost>=2" ipython ipykernel jupyterlab
```

This is a package-installation recipe, **not a tested exact environment lock**. Prefer the versions recorded by the run you are reproducing. Minimum versions alone cannot guarantee that every future dependency combination will behave identically.

Open the repository in VS Code, open a notebook and select the virtual environment as its kernel. Alternatively:

```bash
python -m jupyterlab
```

Run preparation before fitting models. Some earlier notebooks' missing-package messages mention `requirements.txt`; if that file is absent, use the command above. A reviewed requirements file can later capture the agreed environment.

VS Code and Colab are execution environments, not competing modelling approaches. The same notebooks can continue to be used locally. Extracting common functions into `.py` modules is useful for maintenance or automated runs, but is not required to reproduce this branch.

## 6. What to run and dependencies

### Full reproduction

Read or run EDA, run notebook 01, and then run 02–11 in numerical order. This reproduces the experimental narrative. Preserve the same prepared data/manifest pair throughout the experiments being compared.

### Partial reproduction

| Goal | Required work |
|---|---|
| Understand findings without training | Read executed 11 and the relevant earlier notebooks |
| Establish the baseline | Run 01, then 02 |
| Rerun one of 03–07 | Supply the prepared pair and run that notebook; earlier model files are not required |
| Rerun averaging in 08 | Supply the matching completed notebook-05 result folder and prediction artifact folder |
| Rerun monthly exploration in 09 | Supply the prepared pair; 09 fits its own models |
| Rerun later-period evaluation in 10 | Supply the prepared pair; 10 fits its own models |
| Rebuild tables and written synthesis in 11 | Supply completed results/manifests from 03–10; stage 02 is optional |

Notebook 08 performs no model fitting. It needs notebook 05's `ensemble_input_manifest.json`, the copied `preparation_manifest.json`, the common validation index and listed prediction files. Keep `SAVE_PREDICTIONS=True` in 05 when creating this input package.

Notebooks 09 and 10 implement configurations motivated by earlier experiments, but do not load earlier fitted model files. Notebook 09 must refit earlier training origins: a model fitted through month 4 cannot be used to simulate a month-3 deployment.

Notebook 11 reads summaries and manifests only. It does not load `Base.csv`, prepared Parquet, prediction Parquet or model binaries, and does not retrain or retune models. Copying just the executed `.ipynb` files is insufficient to rerun it.

### Default compute scope

| Stage | Default scope |
|---|---|
| 03 | 7 composition conditions × 3 seeds = 21 fits |
| 04 | 6 imbalance conditions × 3 seeds = 18 fits |
| 05 | 11 screening fits, 10 additional seed checks, 3 full-data transfer fits = 24 fits |
| 06 | One seed; 13 base/reference fits and one meta-model fit |
| 07 | 3 period-classifier folds and 9 fraud-model fits |
| 08 | Saved-prediction arithmetic and evaluation; no fits |
| 09 | 9 CatBoost fits across three origins and three seeds |
| 10 | 1 LightGBM, 3 initial CatBoost and 3 refreshed CatBoost fits |
| 11 | Summary processing and plotting; no fits |

These are configured fit counts, not runtime guarantees. CPU, memory, dependencies and output persistence affect runtime. The later notebooks generally use two CPU threads. Changes to grids, seeds or save options should be recorded as a different run.

## 7. Common preparation and model-specific preprocessing

### Common preparation in notebook 01

Preparation validates the input schema, reports duplicates, adds a positional `row_id`, decodes documented missing sentinels, identifies training-constant columns and writes the ordered predictor schema and split metadata.

Negative missing-value sentinels are decoded in these six columns:

- `prev_address_months_count`
- `current_address_months_count`
- `intended_balcon_amount`
- `bank_months_count`
- `session_length_in_minutes`
- `device_distinct_emails_8w`

Do not globally replace all negative values. Negative values in other features can be legitimate. For `intended_balcon_amount`, negative values represent missingness rather than only the exact value −1.

The output retains audit columns; models select inputs using the manifest's `features` list. Reordering the raw CSV changes the meaning of positional row IDs, so IDs must be interpreted alongside source fingerprints.

Preparation does **not** learn global imputation values, scales, category vocabularies or class balancing. These are fitted inside the relevant model pipeline using its own training rows. Validation and evaluation class distributions remain unchanged.

### Family-specific pipelines

| Model family | Numeric features | Categorical features |
|---|---|---|
| Logistic regression | Median imputation, missing indicators and standardisation | Missing-token imputation and training-fitted one-hot encoding |
| LightGBM | Native numeric missing values; no scaling | Training-fitted category codes explicitly declared categorical |
| CatBoost | Numeric missing values retained | Strings with a missing token, explicitly declared categorical |
| XGBoost in notebook 05/06 | Numeric missing values retained | Dense one-hot encoding |

The five categorical predictors are `payment_type`, `employment_status`, `housing_status`, `source` and `device_os`.

LightGBM codes are not treated as ordinary continuous quantities: categorical positions are supplied explicitly, and unseen categories map to −1. The XGBoost implementation deliberately uses dense one-hot features so that encoded zeros are not confused with absent sparse entries. CatBoost performs its own categorical processing during fitting; no external target encoding is precomputed across the full dataset.

Recent and historical training subsets fit separate preprocessors. A smaller subset must not borrow fitted transformations from all-history training. The same principle applies at each temporal retraining origin.

The shared feature schema was established using the main development training period. Notebook 09 retains that schema and previously selected model settings; its earlier-month backtest is therefore retrospective development analysis rather than a fully nested historical model-selection exercise.

## 8. Experimental design

### 02 · Baselines

The initial comparison uses a constant-score reference, logistic regression and LightGBM with unweighted training. The dummy predictor establishes a no-skill reference. Logistic regression provides a linear comparator; LightGBM captures nonlinear relationships and interactions.

For a baseline reproduction, inspect `RUN_TUNING`, `EVALUATE_TEST`, seed and save options. Keep `EVALUATE_TEST=False` when reproducing development-only results; notebook 10 is the explicit later-period evaluation stage. Enabling tuning changes the experiment and adds fits.

### 03 · Training-data composition

| Condition | Eligible training rows | Interpretation |
|---|---|---|
| All history | Every application from months 0–4 | Practical maximum-data reference |
| Recent K | All applications from the latest 1, 2 or 3 eligible months | Emphasis on recent observations |
| Historical K | Random sample from months 0–4 with the same row count as its Recent K counterpart | Matched-size reference |

For example, Recent 1 uses month 4 and Historical 1 draws the same number of rows from months 0–4. Historical sampling can include recent rows; it is not restricted to old months. Whole-month boundaries define recency because within-month timestamps are unavailable.

The experiment holds the LightGBM configuration and unweighted treatment fixed and repeats seeds 42, 43 and 44. Recent versus Historical K addresses the sample-size concern in the proposal feedback. Matching K does not guarantee equal fraud prevalence or make the comparison causal. All history remains a useful reference, but comparing it directly with Recent K changes both recency and sample size.

### 04 · Imbalance treatment

With all-history eligibility retained, compare no treatment, square-root ratio weighting, full ratio weighting and random undersampling at 20, 10 or 1 legitimate application per fraud. The model configuration remains fixed.

Undersampling retains frauds and samples legitimate training applications. A 10:1 ratio refers to legitimate:fraud training rows, not a 10% fraud probability or a review budget. No balancing is applied to validation or later evaluation data.

### 05 · Bounded model-family comparison

Keep full-data unweighted LightGBM as a reference. Under 10:1 undersampling, compare fixed LightGBM and logistic regression with bounded CatBoost and XGBoost grids. CatBoost varies depth {4, 6} and iterations {300, 600}; XGBoost varies maximum depth {3, 5} and trees {300, 600}.

Screen new-family configurations using seed 42 and month-5 AP, then check selected configurations on seeds 43/44. For each seed, undersampled families receive the same sampled rows. Finally, transfer the strongest challenger configuration to untreated full data without retuning it.

This compares concrete pipelines under a limited search budget. It does not show the best attainable performance of each algorithm. The full-data transfer check tests that fixed configuration, not an independently optimised full-data model.

### 06 · Temporal stacking

Generate base predictions using training months 0–1 → prediction month 2, months 0–2 → month 3, and months 0–3 → month 4. These forward predictions support training a logistic meta-model. Base models are subsequently fitted for the month-5 comparison.

The default run uses seed 42 only. Small gains from this experiment do not establish robustness across seeds. Forward prediction boundaries reduce leakage in the meta-training construction, but previously selected settings still make this a development experiment.

### 07 · Temporal domain adaptation

Train a period classifier to distinguish months 0–3 from reference month 4, then use its estimates to reweight historical fraud-model training rows. Compare unweighted, mild and stronger weighting under the fixed CatBoost setup. Month 5 is the fraud-model validation period, not the reference period used to learn these weights.

Inspect period-classifier performance, influential features, clipped weights and effective sample size alongside fraud metrics. A classifier that separates periods well does not imply that the resulting weighting helps fraud detection. This is temporal importance weighting, not pseudo-labelling of unknown fraud outcomes.

### 08 · Ensemble averaging

Use notebook 05's saved predictions to compare individual models, within-seed family averages, a three-seed CatBoost average and a nine-model average. Predictions must share the same row IDs, labels and provenance before averaging.

The arithmetic mean of three models' AP values differs from AP computed after averaging their predictions. The former describes seed-level model performance; the latter evaluates one ensemble. Notebook 08 distinguishes these quantities and stores ensemble recipes.

### 09 · Monthly updating on development months

Three policies use the fixed three-member CatBoost recipe:

- **Frozen:** retain the initial model and cutoff.
- **Threshold only:** retain the model but update its cutoff using the latest labelled month.
- **Retrain + threshold:** expand training history and select a cutoff on a separate recent calibration month.

| Evaluation month | Frozen fit / cutoff month | Threshold-only fit / cutoff month | Retraining fit / cutoff month |
|---|---|---|---|
| 3 | 0–1 / 2 | 0–1 / 2 | 0–1 / 2 |
| 4 | 0–1 / 2 | 0–1 / 3 | 0–2 / 3 |
| 5 | 0–1 / 2 | 0–1 / 4 | 0–3 / 4 |

All policies coincide initially. The latest labelled month is held aside for cutoff selection, so the retrained model does not fit that month. Expanding training history changes recency and sample size together; this measures the combined operational effect.

Model settings were chosen through earlier month-5 experiments. These results are retrospective development backtests, not unbiased historical deployment estimates.

### 10 · Later-period evaluation

The static comparison fits full-data LightGBM, single-seed CatBoost and the three-seed CatBoost average on months 0–4. Each selects its cutoff using month 5 and is evaluated on months 6–7.

| Policy | Month 6 deployment | Month 7 deployment |
|---|---|---|
| Frozen | Initial ensemble; cutoff from month 5 | Same ensemble and cutoff |
| Threshold only | Same initial deployment | Initial ensemble; cutoff from month 6 |
| Retrain + threshold | Same initial deployment | Refit on months 0–5; cutoff from month 6 |

The simulation assumes month-6 labels are available before month 7 is scored. BAF does not supply label-resolution timestamps, so this is an assumption rather than an observed operational fact. Month 6 legitimately becomes calibration data for the predeclared month-7 update; month-7 labels are not used to fit a model or choose a deployed cutoff.

There is only one later-period update opportunity. Results cannot establish a universally optimal retraining frequency.

## 9. Metrics and operational interpretation

### Ranking quality

**Average precision (AP)** is the primary development selection criterion. It uses sklearn's `average_precision_score`, not trapezoidal integration of the precision–recall curve. ROC-AUC is complementary. AP depends on prevalence, so compare alternatives on the same evaluation population before interpreting changes between months.

### Recall at 5% FPR

FPR = legitimate applications flagged / all legitimate applications.

Recall = fraudulent applications flagged / all fraudulent applications.

Thus, 5% FPR means flagging 5% of legitimate applications. It does not mean that only 5% of flagged applications are legitimate, and it does not mean reviewing exactly 5% of all applications.

**Retrospective recall@5%FPR** uses the evaluated split's labels to locate an empirical ROC operating point subject to the FPR ceiling. It is a useful comparison of ranking performance, but that evaluation-derived cutoff is not an available deployment rule for that same future period.

### Previously selected cutoffs

A deployed cutoff is chosen using an earlier calibration month and then applied unchanged to its designated future month. Future realised FPR can differ from 5%. Report realised FPR, recall, precision and workload together: higher recall obtained by flagging many more legitimate applicants is not a free performance improvement.

Here, “calibration month” means cutoff selection. It does not imply that the scores have undergone probability calibration. Scores from undersampled training should not be interpreted directly as population fraud probabilities.

### Exact review capacity

A top-K policy selects `floor(capacity × N)` applications, using the supplied label-independent tie handling. The standard capacities are 1%, 2% and 5% of all evaluated applications.

This controls review workload within the assumed batch, not FPR. It assumes the batch is available for ranking and is not a simulation of immediate online decisions. In notebook 10, monthly capacity totals sum separate monthly queues; they are not one pooled two-month ranking.

For example, a 2% review budget represents approximately 200 applications per 10,000. Report how many have fraud labels and how many are legitimate, rather than inventing loss amounts or investigation costs.

### Aggregation and uncertainty

For policy summaries, sum monthly confusion counts before deriving pooled recall, precision and FPR. An arithmetic mean of monthly AP is a macro average, not pooled AP. Changing-model policies can have changing score scales, so do not create a new pooled cutoff after observing their evaluation outcomes.

Seed standard deviations describe variation among the fitted runs; they are not confidence intervals or significance tests. Ensemble members are not independent repeated evaluations of a deployment policy. Small numerical gains should be described as modest observations unless stronger uncertainty analysis supports a firmer conclusion.

## 10. Outputs and run selection

Each experiment writes a new run subdirectory. Preserve complete runs rather than combining preferred metrics from different configurations.

| Stage | Results directory beneath project root | Representative exports |
|---|---|---|
| 02 | `results/baseline/<run>/` | `validation_metrics.csv`, `selected_configs.json`, `run_info.json` |
| 03 | `results/training_composition/<run>/` | `condition_summary.csv`, `paired_summary.csv`, `run_info.json` |
| 04 | `results/imbalance_treatment/<run>/` | `condition_summary.csv`, `run_info.json` |
| 05 | `results/model_comparison/<run>/` | `final_development_summary.csv`, `ensemble_input_manifest.json`, `run_info.json` |
| 06 | `results/temporal_stacking/<run>/` | `stacking_summary.csv`, `run_info.json` |
| 07 | `results/temporal_domain_adaptation/<run>/` | `validation_summary.csv`, `domain_feature_importance.csv`, `run_manifest.json` |
| 08 | `results/ensemble_averaging/<run>/` | `validation_metrics.csv`, `ensemble_summary.csv`, `ensemble_recipes.json` |
| 09 | `results/monthly_retraining/<run>/` | `monthly_policy_metrics.csv`, `policy_aggregate.csv`, `run_manifest.json` |
| 10 | `results/temporal_holdout/<run>/` | `static_model_metrics.csv`, `monthly_policy_metrics.csv`, `review_capacity_aggregate.csv`, `run_manifest.json` |
| 11 | `results/results_synthesis/<run>/` | `results_synthesis.md`, `source_inventory.csv`, `synthesis_manifest.json`, tables and figures |

Artifact directories for modelling stages use the corresponding stage name beneath `artifacts/`. The notebook-05 filename is `05_model_family_comparison.ipynb`, but its output directory remains **`model_comparison`**. Do not rename it merely to match the notebook filename.

Notebook 08 now uses **`ensemble_averaging`** and **`ensemble_summary.csv`**. The earlier `closing_ensemble` directories are legacy outputs. Only remove them after successfully regenerating 08 and the updated 11; never delete the notebook-05 inputs that 08 still requires.

### Selecting notebook 05 inputs for 08

If exactly one input manifest exists, discovery can select it. If multiple runs exist, set the intended file explicitly:

```python
INPUT_MANIFEST_PATH = Path(
    '/content/drive/MyDrive/BT4012_grp_proj/results/model_comparison/'
    '<actual_run_name>/ensemble_input_manifest.json'
)
```

Replace `<actual_run_name>` with a real completed run. `INPUT_ARTIFACTS_DIR` can be set separately if the matching artifacts were moved. Prefer retaining the corresponding `artifacts/model_comparison/<same_run_name>/` layout.

### Selecting results for 11

Notebook 11 requires stages 03–10. Stage 02 may be absent, in which case its omission is recorded. Auto-discovery accepts a unique eligible run; it does not silently choose the newest or best-scoring one.

When multiple eligible runs exist, set the relevant entry in the existing configuration dictionary:

```python
RUN_DIR_OVERRIDES['08'] = (
    '/content/drive/MyDrive/BT4012_grp_proj/results/ensemble_averaging/'
    '<actual_run_name>'
)
```

This value is a **run directory**, whereas 08's `INPUT_MANIFEST_PATH` is a **JSON file**. Keep the other stage entries in the dictionary. Set overrides after its initial definition and before discovery, so a later assignment does not reset them.

The synthesis checks prepared-data provenance and the relationship between selected 05 and 08 inputs. If those checks fail, restore consistent runs instead of disabling the checks.

## 11. Findings and reporting boundaries

The completed exploration supports the following restrained account. Consult the selected notebook-11 exports for exact numbers, run identities and metric definitions rather than copying rounded displayed values into code.

| Experiment | Observed direction and interpretation |
|---|---|
| Training composition | All history was retained. Recent-only subsets did not establish an advantage over matched-size historical samples under the tested configuration. |
| Imbalance treatment | 10:1 undersampling improved validation AP and reduced fitting cost relative to untreated LightGBM, with a small recall@5%FPR trade-off. |
| Model family | CatBoost under the tested undersampling setup was carried forward from the bounded comparison. This is a pipeline-specific finding. |
| Temporal stacking | Added complexity produced only a small development improvement; the default experiment used one seed. |
| Temporal reweighting | The tested weighting schemes did not improve the matched unweighted reference. This does not rule out all domain-adaptation approaches. |
| Averaging | The three-CatBoost average was carried forward based on development AP and lower model count than the nine-model blend. Gains were modest. |
| Monthly updating | Threshold maintenance changes workload even when ranking is unchanged. Model refresh must be assessed against threshold-only updating, not just a frozen cutoff. |
| Later-period evaluation | Report ranking, realised cutoff behaviour and fixed review-budget outcomes separately; they answer different operational questions. |

The selected recipe uses all-history eligibility, 10 legitimate training examples per fraud, CatBoost depth 4 and 600 iterations, and equal prediction averaging across seeds 42/43/44. “All history” describes the eligible time range before undersampling; it does not mean every eligible legitimate application is retained in each member's fit.

The business discussion should explain the proposed review workflow, available review capacity, label delay, cost of legitimate reviews and model-maintenance overhead. Monetary claims require additional evidence about losses, review costs and intervention effectiveness that this dataset does not provide.

Important limitations include synthetic data; only a small number of months; assumed label availability; repeated development selection; prior team exposure to later months; unequal search budgets across families; recency and sample-size changes together in expanding-history retraining; and no formal significance claim from point differences alone.

Further experiments can be recorded, but any change motivated by months 6–7 should be labelled test-informed exploration. Those same outcomes cannot then serve as independent confirmation of the revised choice.

## 12. Reproducibility and collaboration

Preserve the raw input identity, prepared Parquet/manifest pair, notebook configuration, dependency versions, selected run names and exported manifests. Predictions must remain aligned by row ID; do not average arrays copied from independently sorted tables.

Preparation overwrites its configured prepared pair. Most experiments create new run folders instead. If preparation changes, retain the previous pair separately when older runs must remain reproducible. Do not mix artifacts from different preparations merely because filenames match.

For the model-family experiments, saved `.joblib` pipelines include preprocessing. Native classifier exports alone do not necessarily include the sklearn transformations. Only load trusted model files and use compatible dependencies. Notebooks 09 and 10 refit their own models, so they do not depend on loading earlier joblib files.

Suggested additions to the existing `.gitignore` are:

```gitignore
/data/
/artifacts/
.venv/
__pycache__/
**/.ipynb_checkpoints/
```

Merge these with existing rules; do not replace unrelated team exclusions. Ignoring a path does not untrack files already committed. Do not commit raw/processed data or large row-level artifacts. Avoid blanket-ignoring all JSON or CSV files because small manifests and summaries are valuable evidence.

For a team handoff, share the executed notebooks, this guide, selected complete result folders, the source run identities and access instructions for larger artifacts. Notebook 11 can be reproduced using result folders alone; notebook 08 additionally needs the matching notebook-05 predictions.

Keep one main editor per notebook where possible. Review notebook diffs, preserve useful plots/tables, and remove accidental large dataframe dumps. Do not describe branch work as merged into the main project until that integration actually occurs.

## 13. Troubleshooting

| Symptom | Check or action |
|---|---|
| `Base.csv` not found | Confirm the mounted path and capitalisation; local input belongs in `data/`, Drive input directly in the project folder |
| Prepared Parquet or JSON missing | Run 01 or restore the matching prepared pair |
| Incorrect local root | Set `LOCAL_REPO_ROOT` explicitly |
| Missing-package message references absent requirements | Use the local installation command in this guide in the notebook's kernel environment |
| Package installed but imports still fail | Check the selected kernel; restart after installing/upgrading compiled dependencies |
| Checksum or provenance mismatch | Restore matching data and run artifacts; do not bypass the assertion to combine inconsistent results |
| 08 finds zero manifests | Restore a completed 05 results folder or run 05 with prediction saving enabled |
| 08 finds multiple manifests | Set `INPUT_MANIFEST_PATH` to the intended file |
| 08 cannot find predictions after moving folders | Restore matching artifact layout or set `INPUT_ARTIFACTS_DIR` |
| 11 finds multiple eligible runs | Set the affected `RUN_DIR_OVERRIDES` entries to actual run directories |
| 11 cannot find 08 summaries | Use the updated 08/11 pair and `results/ensemble_averaging/`, not legacy closing folders |
| 11 reports missing files | Copy complete summary/manifests from the source run; an embedded notebook table is not an exported CSV |
| Rerun creates a different folder | Expected: each execution gets a unique run name |
| Failure after several fits | Preserve partial output for diagnosis; do not assume automatic resume or mark the run complete manually |
| Logistic regression convergence warning | Inspect convergence before interpreting or reporting that fit |
| Dummy recall@FPR is blank | Expected for the constant-score deterministic reference used in the baseline |
| Deployed FPR exceeds 5% | A past cutoff does not guarantee the same FPR under temporal change; inspect the specified cutoff and evaluation period |
| Frozen and threshold-only AP are identical | Expected: the score ranking is unchanged; only the cutoff differs |
| Results exist in Drive but not GitHub | Expected: copy selected exports deliberately and commit them yourself |

## 14. References and acknowledgement

- [Official BAF repository](https://github.com/feedzai/bank-account-fraud)
- [Official BAF datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)
- [Base dataset download](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- Jesus et al. (2022), *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS Datasets and Benchmarks.
- Project scope and feedback: [proposal](BT4012_Proposal_BAF_Revised.pdf) and [feedback](feedback.md).

Notebook development and documentation used ChatGPT/Codex assistance. The group should accurately acknowledge other assistance, including any Claude-generated scaffolding actually used, and distinguish generated suggestions from code and results that were reviewed and executed. Each contributor should be able to explain the assumptions, implementation and interpretation of their submitted work.
