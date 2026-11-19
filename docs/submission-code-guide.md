# Submission code and evidence guide

This guide consolidates the completed local studies for the final report and slide skeleton. It does not change their training, preprocessing, selection, score or inference implementations. The original protocol and all frozen evidence remain authoritative. Settings are analyst-selected and open to group review.

## Quick route: reproduce the submission tables

From the repository root, run:

```sh
python3 scripts/build_submission_evidence.py --check
python3 scripts/build_submission_evidence.py
.venv/bin/python -m pytest -q tests/test_submission_evidence.py
```

The exporter uses only the Python standard library and committed small tables/selection records; raw Base.csv, ignored fitted models and prediction caches are not needed. Tests require pytest from the pinned environment. The exporter checks fixed input SHA-256 identities, original and exploratory confusion matrices at every exported review capacity, all five consumed F2 roles, and the selected candidate/policy/development F2 against its frozen selection. It rejects missing, changed or inconsistent evidence. Do not update hashes merely to make a failed export pass: investigate the discrepancy.

Outputs are deterministic CSVs plus `submission/generated/evidence_manifest.json`. Re-exporting overwrites only the generated presentation view; it never overwrites frozen reports. `--output /absolute/path/to/new-directory` selects another export destination. Output inside `reports/`, `configs/`, `baf/` or `openspec/` is rejected.

This is **table reproduction from saved evidence**, not model retraining, independent raw-score recomputation, a new scientific audit or a new statistical inference. No model fit is performed. Development macro F2 averages month-specific achieved F2 and must not be recomputed from averaged confusion counts.

## Files to use in the report

| Generated file | Evidence and permitted interpretation |
|---|---|
| `monthly.csv` | Development-month volume and fraud prevalence |
| `recency_development.csv` | Recent K minus exactly size/class-matched historical K; three forward months |
| `recency_intervals.csv` | Conditional fixed-model paired month-7 row-bootstrap intervals from the first study |
| `original_final_summary.csv` | First-study models/weighting at the identical 969-application month-7 review quota; historical rows are seed means |
| `logistic_fixed_c.csv` | Adaptive development-only fixed-C diagnostic separating regularisation and engineered-feature effects |
| `improvement_development.csv` | Development model/policy comparisons; F2 and fixed-capacity objectives remain separate |
| `improvement_month7.csv` | Consumed-month AP and exact 1% review counts for five selected/dependency score sources |
| `f2_comparison.csv` | Five named strategies, original policy identity, development macro F2 and consumed-month decisions |
| `f2_development_intervals.csv` | Original conditional development intervals; include selection/search limitations |
| `f2_consumed_intervals.csv` | Original conditional consumed-month intervals; do not imply untouched confirmation |
| `evidence_manifest.json` | Exact source and generated-file hashes, validation scope and limitations |

The five F2 roles are unweighted LightGBM, balanced LightGBM, XGBoost/LightGBM fusion, additive latest fusion and the selected bagged-anchor fusion. The exporter does not sweep the consumed month to choose a new winner. Unweighted LightGBM's stronger observed month-7 F2 is descriptive and does not replace the development-selected bagged-anchor strategy.

Only completed studies contribute measured claims. The stopped L-BFGS spline study and incomplete context-feature ablation are excluded from this submission view. Existing exploratory search history must remain disclosed; excluding a result from the concise report is not erasing it from the repository.

## Full model reproduction is a different route

Use Python 3.12.9 and the exact pins in `requirements.txt`, `requirements-improvements.txt` and, for XGBoost descendants, `requirements-xgboost.txt`. Obtain BAF Base Kaggle version 2 using the data-access and SHA-256 instructions in README; keep data under ignored `data/raw/baf/Base.csv`. A different dataset version is not the same experiment.

The historical full-training recipes and ordered commands are preserved in README under **Reproducing results**, **Exploratory improvement experiments**, **XGBoost score fusion**, and **Bagged-anchor fusion**. Later descendants need their exact ancestor-selection restore blocks, pinned environment and resource gates. The shallow-CatBoost complete reproduction guide is `reports/baf-shallow-catboost-v1/reproduction.md`.

For the original first study, README documents the historical fresh-run order:

```sh
.venv/bin/python -m baf.eda
.venv/bin/python -m baf.study validate
.venv/bin/python -m baf.study freeze
.venv/bin/python -m baf.study final
.venv/bin/python -m baf.study diagnostic
.venv/bin/python -m baf.report
```

These commands are not an instruction to rerun a frozen study in place. Existing guards reject overwritten selections and stale identities. Conduct any costly full reproduction in a separate checkout/output destination following the applicable reproduction guide, retain original selections and receipts, and record what was genuinely rerun. Changing output destinations requires the documented config procedure and can affect descendant-preservation identities. Do not simply edit a frozen config. A fresh directory does not make month 7 independent again.

The present consolidation adds no new model family or model search. Local original controlled comparisons provide the report's scientific core; external branch workflows are reviewed separately rather than silently merged into the frozen experiment code. In particular, a split-specific recall-at-FPR benchmark uses evaluation labels to define its cutoff, while a validation-derived threshold can exceed its nominal future budget. Neither is interchangeable with the exact 1% queue used in the original study.

## Operational interpretation and acknowledged assistance

Each row is a synthetic account application, not a payment or an investigation outcome. Alerts mean additional review, not automatic rejection or money saved. The report should show fraud detected, fraud missed, legitimate reviews and actual alert fraction alongside F2/AP. Costs, label delays, human capacity and retraining cadence need bank-specific evidence before deployment; no financial savings are measured by these studies.

Codex/ChatGPT assisted with this consolidation, exporter, tests and documentation. Existing study reports record their own assistance. Preserve those acknowledgements and any teammate/external-branch contributions in the final submission. All group members must be able to explain and justify the submitted code and interpretation.
