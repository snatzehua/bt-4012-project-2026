# Bank account fraud detection under temporal drift

BT4012 Fraud Analytics | Group 8 | NUS AY2026/2027 Semester 1

This project studies fraudulent bank-account applications using the Bank Account Fraud (BAF) Base dataset. We compare training-data recency, model families and imbalance treatments under chronological evaluation, then examine feature engineering, ensembles and review policies. The practical question is how to identify more fraud while limiting unnecessary reviews of legitimate applicants.

The completed controlled study responds to the [revised proposal](BT4012_Proposal_BAF_Revised.pdf) and [lecturer feedback](feedback.md). It compares recent applications with historical samples matched in both size and fraud/nonfraud counts, so that training volume does not explain an apparent recency effect. Logistic regression and tree models use separately justified preprocessing. The proposal describes the original research intent; the [experiment protocol](docs/experiment-protocol.md) and measured findings describe what was implemented.

## Main findings

- **Recency alone offered no consistent advantage.** Recent-only training did not reliably outperform size/class-matched historical samples in the tested configurations. The historical pool includes recent observations; it is not an old-only control. Conditional month-7 intervals for both models include zero. See the [controlled study](reports/baf-study-v1/findings.md).
- **Model and operating policy matter together.** At the same month-7 review budget, the tested unweighted tree models outperform unweighted logistic regression. Balanced class weighting lowers recall at that capacity, even though it can increase recall at a default score threshold. These results concern fixed presets rather than every possible model configuration.
- **Feature gains were model-specific.** Engineered features improve the tested logistic model more consistently than the tree models. A separate fixed-regularisation diagnostic supports that feature effect. The capacity-selected 25% LightGBM / 75% CatBoost blend catches 334 frauds versus 319 for unweighted LightGBM within the same 969 reviews. This follow-up month-7 result is exploratory; see the [improvement findings](reports/baf-improvements-v1/findings.md).
- **Later F2 searches did not establish a stronger future model.** The selected bagged-anchor fusion scores development macro F2 0.330881, but its month-7 F2 of 0.372240 trails the unweighted LightGBM reference at 0.376884. F2 policies have no imposed 1% review quota or precision floor. The [bagged-anchor findings](reports/baf-bagged-anchor-fusion-v1/findings.md) retain the negative comparison and uncertainty.

Month 7 was evaluated in the original study before the follow-up experiments. Those later comparisons use an already-consumed holdout and need confirmation on new data. A better observed month-7 reference does not retrospectively replace the model chosen on development data.

## Report, presentation and code

The [submission pack](submission/README.md) identifies the available report draft, editable sources, preliminary slides and remaining final assets. The tracked [PDF draft](submission/reportXX-draft.pdf) is built from [Markdown](submission/report.md); it does not reproduce the separately edited `report08.docx`. A final `report08.pdf` is not included in this checkout.

Use the [code and evidence guide](docs/submission-code-guide.md) to trace report tables to experiments and reproduce either saved tables or models. The canonical implementation is in `baf/`, with configurations in `configs/` and methods in `docs/`. The [pinned branch audit](docs/submission-branch-audit.md) explains the separate supplementary notebook workflow, its operational results and differences in training periods and capacity rounding. Its external Drive exports were not independently reproduced here, so it is not combined with the core studies into one leaderboard.

## Reproduce the report tables

With Python available, run these commands from the repository root:

```sh
python scripts/build_submission_evidence.py --check
python scripts/build_submission_evidence.py
```

The exporter uses the Python standard library and committed evidence; it does not need Base.csv or fitted models. It writes ten CSV tables and a provenance manifest to `submission/generated/`. This reproduces tables from saved results, rather than retraining models or reconstructing row-level predictions. `--check` validates source hashes and metric consistency without comparing or overwriting existing generated files. The [guide](docs/submission-code-guide.md#export-tables-from-committed-evidence) includes an isolated comparison procedure and focused tests.

Frozen hashes require the original file bytes. On Windows, Git's automatic LF-to-CRLF conversion can make a clean checkout fail evidence verification. Use the guide's checkout instructions with conversion disabled; do not change expected hashes to accept altered files.

## Run the experiments

Exact model reproduction uses Python 3.12.9 and pinned dependencies. `requirements.txt` covers the first study; `requirements-improvements.txt` includes it and adds CatBoost dependencies; `requirements-xgboost.txt` includes both and adds XGBoost. The latter is required for the full historical test suite. Some later runners use Unix facilities, so the documented model-replay route uses Linux or WSL.

Obtain BAF Base **Kaggle version 2**, place it at `data/raw/baf/Base.csv`, and verify its SHA-256 before running data-dependent commands. The [ordered model instructions](docs/submission-code-guide.md#reproduce-the-models) cover setup, download, scoped checks, EDA, validation, selection freeze, evaluation and reporting. [Later-study recipes](docs/study-reproduction.md) retain ancestor-selection restores, cache rules and gates; they are not a single command to rerun every search.

Run full replays in a disposable checkout: reporting stages can write tracked files as well as ignored outputs. Keep frozen configurations, selections and historical evidence intact. Resume only with unchanged inputs and software; further tuning requires a separate study. The incomplete context-feature-ablation study and stopped spline attempt remain available for inspection but contribute no completed performance claims.

## Repository layout

| Location | Contents |
|---|---|
| `baf/` | Sampling, preprocessing, models, evaluation and report modules |
| `configs/`, `docs/` | Study configurations, protocols and reproduction guides |
| `reports/baf-*/` | Findings, figures, compact results and verification records |
| `submission/`, `scripts/` | Report/presentation drafts and evidence/document builders |
| `tests/` | Methodology and evidence checks with study-specific prerequisites |
| `data/`, `outputs/` | Ignored local datasets, fitted models and prediction caches |
| `openspec/`, `.agents/` | Requirements, planning history and coding workflows |

## Limitations and acknowledgement

BAF is synthetic account-application data spanning eight months. Results do not measure fraud prevented, investigation outcomes or monetary savings. Label availability, future drift and fairness require additional study before deployment. Fixed-model bootstrap intervals describe conditional row uncertainty; they omit retraining, adaptive search and future-period uncertainty. Random-split diagnostics use different populations and do not isolate drift by themselves. Settings are analyst-selected rather than established deployment recommendations.

ChatGPT/Codex assisted with experimental design, implementation, checks, interpretation and writing. Study protocols and reports retain detailed assistance and software references; the branch audit attributes supplementary work separately. Group members remain responsible for reviewing and explaining submitted material. Data and method references are recorded in the [report draft](submission/report.md) and [protocol](docs/experiment-protocol.md).
