# Code and evidence guide

Use this guide to inspect report evidence or reproduce the model runs. The [submission pack](../submission/README.md) describes document sources; the [branch audit](submission-branch-audit.md) treats supplementary workflows separately. Run commands from the repository root. Shell blocks for full model reproduction use Bash on Linux or WSL; `python` in the evidence-only route means an available Python interpreter.

## Export tables from committed evidence

The exporter needs only the Python standard library. It checks committed source hashes, selection identities and confusion-count consistency, then exports ten CSVs and `evidence_manifest.json` to `submission/generated/`. It does not fit models, read Base.csv, reconstruct raw predictions or reproduce supplementary Drive artifacts.

### Preserve original checkout bytes

Frozen evidence uses SHA-256 of exact file bytes. On Windows, `core.autocrlf=true` changes LF files to CRLF during checkout; Git may report a clean tree while the exporter rejects those files. Create a separate checkout with conversion disabled:

```sh
git -c core.autocrlf=false clone https://github.com/snatzehua/bt-4012-project-2026.git bt4012-replay
cd bt4012-replay
git config core.autocrlf false
git switch --detach 7e9baab
```

Commit `7e9baab` contains the submission exporter and its evidence. If it is unavailable on the remote, clone from the local repository path instead; the scientific sources and evidence are unchanged by the documentation cleanup. Uncommitted edits are not included in a clone. Do not normalise frozen files in an existing research checkout or replace expected hashes to silence a mismatch.

### Export and compare

```sh
python scripts/build_submission_evidence.py --check
python scripts/build_submission_evidence.py
```

The first command checks source evidence without writing files. The second overwrites `submission/generated/`. Use the [table index](#report-table-index) to locate results, or continue to [model reproduction](#reproduce-the-models). Optional [export comparison and tests](#optional-export-verification) are below the main routes.

## Report table index

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

## Reproduce the models

### Environment and data

Exact scientific replay uses Python 3.12.9. In a separate checkout with original LF bytes, create an environment and install the existing pins:

```sh
python3.12 -c "import sys; assert sys.version_info[:3] == (3, 12, 9), sys.version"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

`requirements-improvements.txt` includes the base requirements and adds CatBoost, graphviz and plotly. `requirements-xgboost.txt` includes improvements and adds XGBoost; install it instead for later studies or their complete suites. Keep exact versions rather than upgrading to resolve an installation problem. Installation success in a new environment is not established by historical receipts. Some later modules import Unix `resource`, so use Linux or WSL for those replays.

The dataset is [BAF Base, Kaggle version 2](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/versions/2): 1,000,000 applications, 32 columns and eight months. See the [official data repository](https://github.com/feedzai/bank-account-fraud) and [dataset paper](https://arxiv.org/abs/2211.13358). Download and extract before running data-dependent checks:

```sh
mkdir -p data/raw/baf
curl --fail --location --retry 3 \
  'https://www.kaggle.com/api/v1/datasets/download/sgpjesus/bank-account-fraud-dataset-neurips-2022/Base.csv?datasetVersionNumber=2' \
  --output data/raw/baf/Base.csv.zip
unzip -n data/raw/baf/Base.csv.zip -d data/raw/baf
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import json
c = json.loads(Path('configs/study.json').read_text())
p = Path(c['data_path'])
assert p.stat().st_size == 213427735
assert hashlib.sha256(p.read_bytes()).hexdigest() == c['data_sha256']
print('BAF Base v2 checksum verified.')
PY
```

Expected SHA-256: `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. The URL is the historical version-specific download route; a login or network restriction may require downloading version 2 manually through Kaggle. Keep raw data and bulk outputs in ignored `data/` and `outputs/`.

### Original controlled study

The base-only methodology checks have a defined scope:

```sh
.venv/bin/python -m pytest -q tests/test_methodology.py
```

This file uses synthetic sampling, metrics and preprocessing fixtures and tests stale/missing selection rejection. It includes a small logistic fit and does not read Base.csv. It is not the complete repository suite. For example, `tests/test_improvements.py` reads Base.csv, and later tests require extra libraries, preserved ancestor selections and small model fits.

In an isolated checkout with no first-study selection, execute:

```sh
.venv/bin/python -m baf.eda
.venv/bin/python -m baf.study validate
.venv/bin/python -m baf.study freeze
.venv/bin/python -m baf.study final
.venv/bin/python -m baf.study diagnostic
.venv/bin/python -m baf.report
```

The [protocol](experiment-protocol.md) and [config](../configs/study.json) define training months 0-3 / validation 4, 0-4 / 5 and 0-5 / 6, with month 7 evaluated after selection freeze. Historical controls match recent K in exact fraud and nonfraud counts, using seeds 17, 29 and 43. Model seed is 2026; fits use at most six threads. The full recipe has 90 validation fits, 14 final fits and three random-split diagnostic fits.

Run artifacts go to `outputs/baf-study-v1/`; small summaries and figures go to `reports/baf-study-v1/`. **The report command writes tracked report paths**, so a different ignored output directory alone is not replay isolation. Keep the original evidence outside the disposable checkout and compare reproduced outputs before adopting any changes.

Interrupted validation can resume with unchanged source/config/data/software. Validation refuses to run after freeze, freeze refuses to overwrite choices, and final evaluation rejects stale or missing choices. Never delete a working study's selection to bypass these guards. A revised scientific recipe requires a new configuration and output identity; it cannot restore an untouched month 7.

### Feature and ensemble improvements

After environment/data setup, install the improvements superset and run its scoped checks before training:

```sh
.venv/bin/python -m pip install -r requirements-improvements.txt
.venv/bin/python -m pytest -q tests/test_improvements.py
.venv/bin/python -m baf.improvements validate
.venv/bin/python -m baf.logistic_ablation
.venv/bin/python -m baf.improvements freeze
.venv/bin/python -m baf.improvements exploratory
.venv/bin/python -m baf.improvement_report
```

These tests need Base.csv. The [improvement protocol](improvement-protocol.md) and [config](../configs/improvements.json) preserve the first study and define native categories, engineered features, weighting, undersampling and blends. The original exported selection is committed, so the first study need not be retrained first. Predictions/models use `outputs/baf-improvements-v1/`; report tables use `reports/baf-improvements-v1/`.

The separate `configs/logistic-ablation.json` diagnostic adds development-only raw-C=0.1 fits. It is excluded from main selection and month 7. Resume and freeze rules require unchanged identities. For any separately authorised revised config, keep the diagnostic's `main_config` and `output_dir` consistent and use the corresponding `--config` and report `--diagnostic-config` arguments.

### Later studies and supplementary notebooks

Use the [later-study recipes](study-reproduction.md) for ordered ancestor restores and stages, and the existing [shallow CatBoost guide](../reports/baf-shallow-catboost-v1/reproduction.md) for that study's full setup. They preserve cache-specific handling, resource gates and separate Newton fit/analysis histories. Several full-suite sequences are historical: the [receipt-handoff investigation](study-reproduction.md#full-suite-receipts-and-training-replay) identifies their unresolved current-replay transition. Do not substitute an unrestricted test run immediately after a base-only installation. Historical full-suite receipts describe their recorded environments; they are not tests performed by the table exporter.

The [supplementary notebook workflow at its audited commit](https://github.com/snatzehua/bt-4012-project-2026/tree/d466ff34df6f62abe935b993ef75c8ca0dd3b1ed) has its own notebook instructions and external Drive artifacts. The [branch audit](submission-branch-audit.md) specifies the training/calibration timeline, prior evaluation exposure and `floor(q*N)` monthly budget, versus the canonical study's `ceil(q*N)`. Embedded notebook results were inspected rather than independently rerun. A fresh clone of this branch does not contain its external model and prediction exports.

## Optional export verification

### Compare generated tables

To verify an existing export without overwriting it, generate into a temporary destination and compare:

```sh
python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys

with TemporaryDirectory() as directory:
    subprocess.run([sys.executable, 'scripts/build_submission_evidence.py',
                    '--output', directory], check=True)
    generated = Path('submission/generated')
    for fresh in sorted(Path(directory).iterdir()):
        saved = generated / fresh.name
        if fresh.suffix == '.json':
            # JSON formatting and platform newlines do not change its content.
            assert json.loads(fresh.read_text()) == json.loads(saved.read_text()), fresh.name
        else:
            assert fresh.read_bytes() == saved.read_bytes(), fresh.name
print('All generated tables and manifest match.')
PY
```

CSV comparison is byte-exact; JSON comparison checks parsed content. A checkout with converted CSV line endings will fail the byte comparison. Output inside `reports/`, `configs/`, `baf/` or `openspec/` is rejected by the exporter. A mismatch should be investigated before regenerating committed files.

### Focused evidence tests

The focused tests require pytest, without the modelling stack or raw data:

```sh
python -m pytest -q tests/test_submission_evidence.py
```

Pytest is pinned in `requirements.txt` for the scientific environment. For an evidence-only environment, installing pytest alone is sufficient.

### Temporary-directory troubleshooting

If a restricted environment denies access to the default temporary directory, pass `--basetemp tmp/submission-evidence-tests` with a fresh disposable path; pytest clears that directory.

## Interpretation and assistance

AP summarises ranking, while threshold F2 and exact-capacity review describe different decisions. Macro development F2 averages achieved month-specific F2; it is not F2 recomputed from averaged confusion counts. Learned thresholds can change future workload; an F2 policy has neither a 1% quota nor a precision floor.

Applications flagged mean additional review, not automatic rejection or measured money saved. Fixed-model intervals omit retraining, adaptive selection and future-period uncertainty. Month 7 is consumed; all follow-up comparisons are exploratory. Preserve unsuccessful and unfinished studies alongside positive findings.

ChatGPT/Codex assisted with design, code, checks, interpretation and documentation. Study reports retain detailed acknowledgements and software references; the branch audit identifies supplementary sources. Submitted work requires group review and explanation by its authors.
