# BT4012: Bank Account Fraud under Temporal Drift and Class Imbalance

This branch establishes the data preparation and baseline modeling workflow for the group project. We use the **Base** variant of the Bank Account Fraud (BAF) Dataset Suite to predict fraud in bank account opening applications.

**One row is an account application, not a payment transaction.** Our proposed use case is to rank applications for additional investigation under a limited review budget. The dataset contains labels, not the outcome of an actual intervention, so a flagged fraudulent application is not automatically fraud prevented or money saved.

The first modeling step compares a constant-score reference, logistic regression and LightGBM. It does not yet implement monthly retraining, training-window comparisons, imbalance treatments or a production deployment.

## 1. What is included on this branch

```text
bt-4012-project-2026/
├── README.md
├── requirements.txt
├── .gitignore
├── BT4012_Proposal_BAF_Revised.pdf        # Existing project file
├── feedback.md                          # Existing lecturer feedback
├── notebooks/
│   ├── 00_data_preparation.ipynb         # New: shared preparation/export
│   ├── eda_zs.ipynb                      # Existing: no rename required
│   └── 02_baseline_model_comparison.ipynb # New: baseline pipelines/evaluation
├── data/                                # Local datasets; ignored by Git
│   ├── Base.csv
│   └── processed/
│       ├── baf_clean.parquet
│       └── preparation_manifest.json
├── results/                             # Small shareable tables/figures
│   └── baseline/
│       └── <run_name>/
└── artifacts/                           # Models/predictions; ignored by Git
    └── baseline/
        └── <run_name>/
```

The existing proposal, feedback and EDA notebook remain unchanged. The delivery archive contains only the new notebooks, this README, requirements and a suggested `.gitignore` fragment. Copy those additions into the repository; it contains no raw data or generated model results.

### Are the notebooks coupled to `src/`?

**These two notebooks are standalone.** They do not require a `src/` folder, repository cloning in Colab or private GitHub credentials. This keeps the first run easy for the group.

They share a **data contract** instead: the preparation notebook writes Parquet data and a JSON manifest; baseline modeling reads both. Model-building and metric helpers are documented in the baseline notebook. Once the experiments stabilize, those functions can be extracted into `src/` and imported by notebooks and scripts without changing the experiment design.

Do not maintain divergent copies of evaluation logic in several model notebooks. Agree on a canonical helper implementation when adding the next experiment.

## 2. Input data and fixed evaluation protocol

Download `Base.csv` from the [official Kaggle dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022). Do not use the fake-data generator from earlier scaffolding for reported results.

Our EDA run contained 1,000,000 applications and 32 source columns:

| Split | Month indices | Applications in the inspected Base.csv | Purpose |
|---|---|---:|---|
| Training | 0–4 | 675,666 | Fit preprocessing and model parameters |
| Validation | 5 | 119,323 | Compare configurations and choose thresholds |
| Test | 6–7 | 205,011 | Evaluate agreed final comparisons |

These counts document the earlier EDA, not results pre-populated into the new notebooks. Preparation derives and records the actual counts each time it runs. Month indices are not named calendar months.

`fraud_bool` is the binary target. `month` is a split key. `row_id` is a newly added alignment key. None of these is a predictor. `device_fraud_count` was constant in training in the inspected data; preparation identifies training constants again rather than assuming the result.

**Do not change the split or select final configurations after inspecting test results without documenting the resulting limitation.** All training conditions must eventually use the same future evaluation rows.

## 3. File locations in Google Drive

For the supplied Colab configuration, use this layout:

```text
MyDrive/
└── BT4012_grp_proj/
    ├── Base.csv
    ├── processed/                       # Created by notebook 00
    │   ├── baf_clean.parquet
    │   └── preparation_manifest.json
    ├── results/
    │   └── baseline/
    │       └── <run_name>/
    │           ├── validation_candidates.csv
    │           ├── validation_metrics.csv
    │           ├── validation_capacity.csv
    │           ├── selected_configs.json
    │           ├── run_info.json
    │           ├── preparation_manifest.json
    │           └── figures/
    ├── artifacts/
    │   └── baseline/
    │       └── <run_name>/
    │           ├── dummy.joblib
    │           ├── logreg.joblib
    │           ├── lightgbm.joblib
    │           └── validation_predictions_<model>.parquet
    └── eda_outputs/                     # Existing EDA output folder, if present
```

The Drive notebooks themselves may be stored elsewhere. The dataset/output paths are set explicitly in the first configuration cell:

```python
DRIVE_PROJECT_DIR = Path('/content/drive/MyDrive/BT4012_grp_proj')
```

If using a shared Drive or a shortcut, change this value in **both notebooks** to the actual mounted path. Access to one member's My Drive is not automatically granted to teammates. Each member can use their own copy or an explicitly shared project folder they can access. Do not embed Drive credentials in code.

## 4. Run in Google Colab

1. Open Colab and choose **File → Upload notebook** to upload `00_data_preparation.ipynb`.
2. Choose a CPU runtime. A GPU is not used by the supplied configurations.
3. Run the cells from the top and authorize Google Drive mounting.
4. Confirm that the printed input path points to the intended `Base.csv`.
5. Run preparation through the final export cell. It creates the Parquet/JSON pair in `processed/`.
6. Upload/open `02_baseline_model_comparison.ipynb`.
7. Confirm it uses the same Drive project path.
8. Leave `RUN_TUNING=False` and `EVALUATE_TEST=False` for the first run.
9. Run all cells to establish the validation baselines and save the outputs.
10. Inspect warnings, validation tables and figures before planning further experiments.

Colab usually provides the standard scientific Python packages. The notebooks install missing `pyarrow` or `lightgbm` where required. A newly installed compiled dependency can occasionally require restarting the runtime; rerun from the top afterward.

**Do not expect these artifacts to appear in GitHub automatically.** Colab writes results to Drive. Save the edited notebook and manually copy selected small result tables/figures into the repository when ready to commit. Open the actual downloaded `.ipynb` file to transfer code changes; changing the repository file does not update an already-open Colab copy.

## 5. Run locally with VS Code / Jupyter

The user's existing `<repo>/data/Base.csv` placement is correct. The local configuration uses it directly; there is no need to move it into another `raw/` folder.

Create an isolated environment from the repository root:

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

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

In VS Code, open the repository folder, install the Python/Jupyter extensions if needed, open the preparation notebook, and select this environment as the notebook kernel. Run preparation, then the baseline notebook.

Alternatively, launch JupyterLab:

```bash
python -m jupyterlab
```

The notebooks search the current directory and its parents for a folder containing both `notebooks/` and `data/`. If the project root is not detected correctly, explicitly set this value in both notebooks:

```python
LOCAL_REPO_ROOT = '/absolute/path/to/bt-4012-project-2026'
```

For Windows, a raw string can be used:

```python
LOCAL_REPO_ROOT = r'C:\Users\your_name\projects\bt-4012-project-2026'
```

The same notebook files work in Colab or a local kernel. **VS Code is an editor, not a different modeling method.** You can continue running notebooks there, or keep using Colab for its compute resources. Moving to `.py` scripts is optional and most useful when automating repeated experiments.

## 6. What preparation does—and does not do

### Common transformations

Notebook 00:

1. Reads and validates the expected Base.csv schema.
2. Reports exact duplicate rows without deleting applications automatically.
3. Adds `row_id` using the original CSV row position before filtering/sampling.
4. Converts documented negative sentinels to `NaN` in six specific columns.
5. Keeps category labels as unencoded strings.
6. Identifies constant columns using training rows only.
7. Defines ordered predictor lists and chronological split metadata.
8. Saves data and a manifest with checksums and versions.

The sentinel columns are:

- `prev_address_months_count`
- `current_address_months_count`
- `intended_balcon_amount`
- `bank_months_count`
- `session_length_in_minutes`
- `device_distinct_emails_8w`

Do not globally replace all negative numbers in the dataset. Negative values in other features may be valid. All negative `intended_balcon_amount` values represent missingness, not only the exact value −1.

The prepared file retains excluded constants, label, month and ID for auditing. The manifest's `features` list controls the actual model inputs. Retain the source checksum alongside row IDs: a reordered or replaced CSV changes what a positional ID represents. IDs are not within-month timestamps.

### Transformations deliberately deferred

Preparation does **not** fit medians, scale values, fit categorical vocabularies, create one-hot columns, balance the dataset or train a model. It saves neither a global fitted preprocessor nor balanced validation/test data.

This matters for the future recency experiment. Recent K and Historical K must each fit preprocessing on their own training rows; neither may borrow the All history imputer or encoder.

### Re-running preparation

Preparation replaces `baf_clean.parquet` and `preparation_manifest.json` in the configured processed folder. Preserve an older pair separately if you need to reproduce a run based on it. A preparation ID and fingerprints identify the source and saved output. Baseline verifies the output checksum by default and copies the manifest into its results folder.

The checksum is an integrity check, not an anonymization mechanism. The prepared file still contains application labels and features.

## 7. Baseline pipelines

| Model | Numeric treatment | Categorical treatment | Purpose |
|---|---|---|---|
| DummyClassifier | No learned feature transformation | Ignored by the constant predictor | No-skill reference emitting training prevalence |
| Logistic regression | Training-median imputation, missing indicators, scaling | Missing category imputation + training-fitted one-hot encoding | Interpretable linear baseline |
| LightGBM | Native numeric NaNs; no scaling | Training-fitted category codes, explicitly marked categorical | Nonlinear tree-based comparator |

LightGBM category codes are **not treated as continuous ordered numbers**: their transformed column indices are explicitly passed through `categorical_feature`. Unseen future categories become −1 and are handled as missing by LightGBM. Logistic regression ignores unseen categories in the one-hot representation.

All baseline models are **unweighted**, use the same rows, and receive the same underlying predictors. Do not introduce weighting or undersampling for only one model and then attribute every performance difference to model architecture.

Returned scores support ranking. We do not claim they are calibrated probabilities. Logistic coefficients also do not establish causal effects.

## 8. Configuration and runtime

Edit settings in Section 2 of the baseline notebook:

| Setting | Default | Meaning |
|---|---|---|
| `SEED` | `42` | Randomness for model fitting and capacity tie breaking |
| `N_JOBS` | `2` | LightGBM CPU threads; increase deliberately on larger machines |
| `RUN_TUNING` | `False` | One default candidate per family; True enables three LR and three LightGBM candidates |
| `EVALUATE_TEST` | `False` | Explicit guard for months 6–7 evaluation |
| `SAVE_MODELS` | `True` | Save fitted joblib estimators and their preprocessing |
| `SAVE_PREDICTIONS` | `True` | Save row-aligned validation scores and optional test scores |
| `VERIFY_CHECKSUM` | `True` | Verify prepared data matches its manifest |
| `FPR_TARGET` | `0.05` | False-positive-rate operating target |
| `REVIEW_CAPACITIES` | `[0.01, 0.02, 0.05]` | Exact top-ranking review budgets |
| `LGBM_TREES` | `300` | Number of boosting trees; no early stopping in this initial comparison |
| `LOGREG_MAX_ITER` | `2000` | Logistic-regression convergence limit |
| `RUN_LABEL` | `unweighted_baseline` | Human-readable part of the unique run folder name |

The optional LR search varies `C` across 0.1, 1 and 10. The optional LightGBM search varies `num_leaves` across 15, 31 and 63 while keeping `min_child_samples=100`. The notebook records the complete candidate grid and other settings.

A full-data run can take several minutes or longer, depending on CPU, memory and solver convergence. Tuning fits additional candidates. Do not interpret these as promised runtime bounds. The baseline uses the full training data; it does not silently subsample to speed up execution.

If memory is constrained, disable saving models/predictions, close unused notebooks, or use a larger runtime. If you intentionally train on less data, label the change and do not present it as the full-data baseline. Restarting the kernel releases notebook memory.

## 9. Metrics and how to read them

### Ranking metrics

- **Average precision (AP)** is the primary validation selection metric. This implementation uses `average_precision_score`; it is not trapezoidal PR-curve integration.
- **ROC-AUC** is a complementary ranking metric.
- **Recall at target FPR** is a split-specific retrospective benchmark: labels from that split determine a cutoff satisfying the empirical FPR ceiling.

For constant dummy scores, AP equals the evaluated prevalence and ROC-AUC is 0.5 when both classes exist. The notebook leaves the dummy recall-at-FPR benchmark blank because it has no nontrivial deterministic ranking. A seeded tie-broken dummy top-K queue is still provided as a reference; its realized recall need not equal the review fraction exactly.

### Validation-selected thresholds

`validation_threshold_*` metrics apply the cutoff chosen using month-5 labels. On validation this is the fitted operating point; on test it is the unchanged decision policy. The selected cutoff flags `score >= threshold`. Score ties can prevent attaining exactly 5% FPR, so the selected empirical validation FPR does not exceed the ceiling.

The ROC search allows an infinite cutoff corresponding to no alerts. JSON stores this value as null plus `fpr_threshold_is_positive_infinity=true`; reconstruct it as `float('inf')` when applying the saved threshold.

### Exact capacity versus unchanged thresholds

`validation_capacity.csv` and `test_exact_capacity.csv` select `floor(capacity × N)` highest-scoring applications in each evaluation frame. Scores tied at the boundary use a seeded, label-independent ordering shared across model comparisons on that frame. Pooled test ranking assumes a pooled review batch; monthly tables are the month-by-month budget scenario. These do not simulate instantaneous online decisions.

`test_fixed_validation_capacity_thresholds.csv` instead applies the validation-derived capacity cutoff unchanged. Its future alert rate may exceed or fall below the intended capacity. These are separate policies; do not mix them in the report.

Per-10,000 columns standardize the observed sample counts for interpretation. They quantify flagged fraud labels and reviewed legitimate applications, not financial savings, proven prevention or automatic rejection.

### Validation selection and uncertainty

Validation is reused for configuration selection and threshold setting. Its selected performance is a development estimate, not an untouched final test estimate. This is a single-seed initial baseline, not a comprehensive estimate of seed uncertainty, future drift or statistical significance.

## 10. Outputs and reproducibility

Each execution creates a timestamped run directory with a random suffix. Keep these files together instead of mixing the best-looking metrics from separate runs.

| File | Contents |
|---|---|
| `validation_candidates.csv` | Every tried configuration, fit time, warnings and ranking metrics |
| `validation_metrics.csv` | Selected candidate for each model family and validation operating metrics |
| `validation_capacity.csv` | Exact review-budget outcomes for each selected family |
| `selected_configs.json` | Chosen hyperparameters, validation thresholds and family selected by AP |
| `run_info.json` | Seed, settings, candidate grids, versions, treatment, test-evaluation flag and preparation identity |
| `preparation_manifest.json` | Snapshot of the prepared-data contract used by this run |
| `figures/` | Validation PR and capacity plots |
| `artifacts/.../*.joblib` | Optional fitted estimators including sklearn preprocessing |
| `artifacts/.../*predictions*.parquet` | Optional row ID, month, label, model and score |

If test evaluation is enabled, the notebook also saves `test_metrics.csv`, `test_exact_capacity.csv` and `test_fixed_validation_capacity_thresholds.csv`, plus optional test predictions.

Requirements specify compatible minimum versions, not a strict lockfile. Retain recorded versions. For the final selected environment, consider saving a reviewed environment snapshot with `python -m pip freeze` and validate the reproduction steps in a fresh environment. Different library versions or hardware can lead to small numerical differences.

Re-running configuration creates a new run name. To avoid mixing old variables with new settings, use **Restart kernel/session and Run All** rather than changing settings halfway through execution.

### Reusing artifacts in Python scripts

Future scripts can load the common inputs:

```python
import json
import pandas as pd

prepared = pd.read_parquet('data/processed/baf_clean.parquet')

with open('data/processed/preparation_manifest.json', encoding='utf-8') as handle:
    manifest = json.load(handle)

train = prepared[prepared[manifest['month_column']].isin(
    manifest['split_months']['train']
)]
X_train = train[manifest['features']]
y_train = train[manifest['target']]
```

Use the notebook's equivalent contract checks when adapting this abbreviated example. To score with a trusted fitted baseline:

```python
import joblib

model = joblib.load('artifacts/baseline/<run_name>/logreg.joblib')
scores = model.predict_proba(prepared[manifest['features']])[:, 1]
```

Replace `<run_name>` with the actual folder. This demonstrates loading, not an evaluation protocol: explicitly choose the evaluation rows in real experiments. Keep matching package versions, and never load untrusted joblib files. Saved models contain no notebook-defined estimator classes, so they can be loaded in scripts with the required libraries installed.

## 11. Collaboration and Git hygiene

- Keep each notebook under one main owner's control to reduce notebook merge conflicts.
- Commit code, agreed small tables/figures, dependency information and written interpretation.
- Keep raw/processed datasets and large models/predictions out of Git.
- When saving notebook outputs for review, avoid embedding entire datasets or enormous prediction tables.
- Include the source run name/preparation ID when sharing results.
- Do not commit secrets or user-specific credentials.
- Git ignores do not untrack files already committed. If Base.csv was previously tracked, remove it from Git tracking while retaining the local file, then inspect the staged change before committing. Do not delete the only dataset copy.
- Use `git status` to inspect changes before committing. No repository writes or pushes are performed by these notebook deliveries.

The supplied `gitignore_additions.txt` is a suggested fragment. Merge it into the existing `.gitignore`; do not replace unrelated team rules. Remove an old `README.md` ignore entry if it still exists. The current user-reported `/data/` exclusion is appropriate.

## 12. Next experiments and editor choice

The agreed main experiment compares:

1. **All history:** every eligible training application from months 0–4.
2. **Recent K:** the most recent complete training month(s), with K equal to their row count.
3. **Historical K:** random applications from all eligible months with the same K.

Keep validation/test rows fixed. Refit preprocessing separately for each subset, keep model settings and imbalance treatment fixed for the main comparison, record fraud counts/prevalence, and repeat historical sampling across seeds. Do not infer within-month recency from CSV row order.

Then compare untreated training with class weighting/undersampling in a separate controlled experiment. Test results are not a reason to repeatedly redesign the model. Test evaluation can be deferred until the planned configurations are selected on validation.

These experiments can remain notebooks in either Colab or VS Code. Use scripts when repeated automated runs become easier to manage that way. A practical next structure is `03_training_window_experiment.ipynb`, followed by `04_imbalance_and_operations.ipynb`. Extract canonical helpers into `src/` when it avoids duplication; do not spend the remaining project time rebuilding infrastructure.

## 13. Troubleshooting

| Symptom | Action |
|---|---|
| Raw CSV not found | Check the printed path, Drive folder name and `Base.csv` capitalization |
| Prepared data missing | Run preparation with the same project path first |
| Fingerprint mismatch | Check whether only one artifact was copied or an export was interrupted; rerun preparation or restore the matching pair |
| Unexpected `Unnamed: 0` column | Inspect the CSV source; do not silently use an exported index as a predictor |
| Local package import error | Install requirements in the exact Python environment selected as the notebook kernel |
| Logistic convergence warning | Inspect the warning and solver configuration; raise the iteration limit or investigate numerical behaviour before reporting the fit |
| LightGBM feature-name warning | A generic warning about transformed matrix column names may occur; check recorded warnings and actual outputs rather than treating it as convergence failure |
| NaN recall-at-FPR for dummy | Expected: no nontrivial deterministic ranking exists for equal scores |
| No test result files | Expected when `EVALUATE_TEST=False` |
| LightGBM unknown-category warnings/negative codes | Unseen categories use −1 and are handled as missing; investigate unexpected category changes |
| Results appear only in Drive | Expected for Colab; copy selected files into the repository deliberately |

## 14. References, limitations and acknowledgement

- [Official BAF repository](https://github.com/feedzai/bank-account-fraud)
- [Official BAF datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)
- [Dataset download](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- Jesus et al. (2022), *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS Datasets and Benchmarks.

Limitations include synthetic observations, coarse monthly timing, no actual review outcomes/costs, and limited future evaluation months. This code establishes an experiment pipeline; model performance and practical benefit must be supported by executed results.

These notebooks and documentation were prepared with ChatGPT/Codex assistance. The group should accurately declare Claude-generated scaffolding and other AI/external-code use, review all code, and be able to explain every submitted component.
