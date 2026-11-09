

The separate [protocol](docs/bagged-lightgbm-seeds-protocol.md), [configuration](configs/bagged-lightgbm-seeds.json) and [OpenSpec tasks](openspec/changes/test-bagged-lightgbm-seed-ensembles-f2/tasks.md) define effective uniform row sampling in the original balanced/unweighted LightGBM pipeline. It uses fraction 0.8, frequency 1 and bagging seeds 17/29/43, retaining the original 200-round/15-leaf settings and training-only preprocessing. Each weighting contributes three individual models and equal probability/logit means. These settings are analyst-selected, not group-approved.

Five of fourteen tasks are verified: the prospective declaration, randomized pipeline, fixed aggregation, analytical tests and pre-execution gates. The full methodological suite passed 1070 tests (129 current-study cases); four tiny current-study cache-format fits across focused/full invocations are recorded separately from historical test-module fits. The separate nine-fit synthetic gate passed active/inactive sampling, repeatability, diversity and serialization checks. Actual-host predictor-only preflight covered 903157 eligible rows and 157 maximum required features, with estimated workspace 11,130,259,216 bytes below 70% of host RAM. The full development run is in progress; no new F2 result or winner has yet been established.

The finite declaration contains 18 new development fits, 15 individually prepared reference-base results, 54 physical/45 selectable scores, 90 forward operating rows, 342 matched contrasts, 18 seed-diversity rows and 36 decision-overlap rows. Five fixed operating references remain visible. Selection uses actual macro month-5/6 positive-class F2, beta=2, without a review quota or precision floor. Month 7 remains already consumed and cannot provide independent confirmation. The broader improvement goal remains active.

The 52-source/config/software/protocol identity and 634 mandatory historical paths are isolated from all fifteen completed studies and the stopped attempt. Fifteen original/exported selection pairs and 3522 available historical ignored files are preserved. Required references use current-parent-first cache precedence and only exact individual recipes when wholly absent; partial or changed caches fail. No unrelated historical grid is required. New caches/reports use `outputs/baf-bagged-lightgbm-seeds-v1/` and `reports/baf-bagged-lightgbm-seeds-v1/`.

For reproduction use Python 3.12.9 and unchanged `requirements.txt`, `requirements-improvements.txt` and `requirements-xgboost.txt` (27 dependency pins). Obtain Base v2 with the data-access instructions above, verifying SHA256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`. Model seed is 2026, threads six, bootstrap seed 1027 and 500 paired fixed-decision draws per month. Keep existing frozen evidence intact; run in a disposable copy excluding ignored outputs and move only that copy's current-study report export aside. The following links the already-verified environment and read-only dataset; a new environment instead uses the pinned installation above:

```bash
set -e
baf_original_root="$PWD"
baf_replay_root="$(mktemp -d "${TMPDIR:-/tmp}/baf-bagged-replay.XXXXXX")"
rsync -a --exclude=.git --exclude=.venv --exclude=data --exclude=outputs --exclude=.pytest_cache --exclude=__pycache__ ./ "$baf_replay_root/"
ln -s "$baf_original_root/.venv" "$baf_replay_root/.venv"
mkdir -p "$baf_replay_root/data/raw/baf" "$baf_replay_root/replay-evidence"
ln -s "$baf_original_root/data/raw/baf/Base.csv" "$baf_replay_root/data/raw/baf/Base.csv"
mv "$baf_replay_root/reports/baf-bagged-lightgbm-seeds-v1" "$baf_replay_root/replay-evidence/"
mkdir -p "$baf_replay_root/reports/baf-bagged-lightgbm-seeds-v1"
cd "$baf_replay_root"
```

Restore exactly fifteen ancestor selections; reject changed records before copying missing ones:

```bash
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import shutil
from baf.common import read_config, digest
c = read_config('configs/bagged-lightgbm-seeds.json')
pairs = [(Path(p), Path(p.replace('outputs/', 'reports/', 1)), h)
         for p, h in c['preserve_sha256'].items()
         if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(pairs) == 15
for original, exported, expected in pairs:
    assert exported.is_file() and digest(exported) == expected, str(exported)
    if original.exists():
        assert digest(original) == expected, str(original)
for original, exported, expected in pairs:
    if not original.exists():
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(exported, original)
PYTHON
```

Run stages sequentially. `validate` includes the complete grid-specific independent development audit before freeze; the explicit `audit_development` entry point is available before selection. The methodological suite verifies all 45 winner/policy cases and includes two genuine tiny new-format fits per full invocation. `preflight` checks preserved source-miniature ancestry without rerunning old gates; `miniature` runs nine new synthetic fits.

```bash
set -e
.venv/bin/python -m pytest -q tests --junitxml=reports/baf-bagged-lightgbm-seeds-v1/tests.xml
.venv/bin/python -m baf.bagged_boosting record_checks
.venv/bin/python -m baf.bagged_boosting preflight
.venv/bin/python -m baf.bagged_boosting miniature
.venv/bin/python -m baf.bagged_boosting prepare_development
.venv/bin/python -m baf.bagged_boosting validate
.venv/bin/python -m baf.bagged_boosting freeze
.venv/bin/python -m baf.bagged_boosting exploratory
.venv/bin/python -m baf.bagged_boosting_report report
```

Wholly absent references can require 15 development reproductions in addition to 18 new bagged fits. Frozen transfer permits seven reference-winner or fourteen new-winner roles, with at most eleven bases/seventeen scores/six new bagged fits and tighter cache/type-specific bounds. Missing final references can require two additive and three tree reproductions, counted separately. Numerical/resource failures retain evidence and stop completion. Conditional row intervals exclude training-seed/retraining, repeated search/selection, temporal/generative uncertainty and multiplicity correction. Further model or policy tuning requires a separate study.

