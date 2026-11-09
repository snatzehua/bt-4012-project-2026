# Reproducing the shallow CatBoost study

Run from the repository root in a separate checkout with the original study output directory absent. Existing selections and finished stages are immutable; never delete a selection to retune. Full data and outputs stay ignored. Python is 3.12.9; requirements-xgboost.txt includes all 27 exact package pins used here. Use the README's Base-v2 download instructions and verify SHA-256 7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809.

```sh
uv venv --python 3.12.9 .venv
uv pip install --python .venv/bin/python -r requirements-xgboost.txt
uv pip check --python .venv/bin/python
```

Restore only the eighteen mandatory original selections from their exact exported counterparts. Existing unequal files are errors, not overwrite targets:

```sh
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
from importlib.metadata import version
import sys
c=json.loads(Path('configs/shallow-catboost.json').read_text())
from baf.shallow_catboost_study import planning_evidence_path
plan=json.loads(planning_evidence_path().read_text())
assert sys.version.split()[0]==plan['python']
assert all(version(k)==v for k,v in plan['verified_pins'].items())
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(c['data_path'])==c['data_sha256']
pairs=[p for p in c['preserve_sha256'] if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(pairs)==18
for name in pairs:
    target=Path(name); source=Path(name.replace('outputs/','reports/',1))
    expected=c['preserve_sha256'][name]
    assert sha(source)==expected
    if target.exists(): assert sha(target)==expected
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    assert sha(target)==expected
from baf.shallow_catboost_study import guard
guard(c)
print('All 27 pins, Base, eighteen exact restores and ancestry verified.')
PY
```

Execute these stages in order. Use fresh XML/ledger/log paths; the test runner refuses existing evidence paths. This is a replay of the declared finite study, not permission for new settings. Save each command's output to a separate closed log.

```sh
.venv/bin/python -m baf.shallow_catboost_study verify_declaration
.venv/bin/python scripts/check_shallow_catboost.py --junit reports/baf-shallow-catboost-v1/replay-tests.xml --ledger reports/baf-shallow-catboost-v1/replay-fits.json
# Only after that complete invocation passes, and preserving its originals:
cp reports/baf-shallow-catboost-v1/replay-tests.xml reports/baf-shallow-catboost-v1/tests.xml
cp reports/baf-shallow-catboost-v1/replay-fits.json reports/baf-shallow-catboost-v1/test_model_fits.json
.venv/bin/python -m baf.shallow_catboost_study record_checks
.venv/bin/python -m baf.shallow_catboost_study preflight
.venv/bin/python -m baf.shallow_catboost_study miniature
.venv/bin/python -m baf.shallow_catboost_study prepare_development
.venv/bin/python -m baf.shallow_catboost_study validate
.venv/bin/python -m baf.shallow_catboost_report audit_development
.venv/bin/python -m baf.shallow_catboost_study freeze
.venv/bin/python -m baf.shallow_catboost_study exploratory
.venv/bin/python -m baf.shallow_catboost_report report
openspec validate --all --strict --no-interactive
```

`validate` creates the producer tables; the independent audit must pass before `freeze`. Freeze validates the actual 162-case winner/role/DAG contract and available-cache bounds. Only the resulting eight or twenty-two roles can transfer. Month 7 was already consumed; repeating these stages does not restore an independent holdout. The report command independently audits every saved development model again, the bounded transfer, all 1500 fixed-decision bootstrap draws and plotted numeric values. Inspect the regenerated image and verify closed-log, run and report manifests before synchronizing or archiving fully complete tasks.

Individual required reference access uses a complete local set first, then the original bagged-anchor origin and its declared earlier origins. A wholly absent individual set may be recreated using its exact original recipe only in this study's new output directory, after unchanged gates pass. A partial, corrupt or incompatible inspected set stops execution in place; do not delete members, fall through to other origins, replay an unrelated grid or invoke the conditional-expert gate. Derived reference scores preserve their original arithmetic/domain and aligned member populations. Missing historical full-size caches are not required merely to run the synthetic methodological suite.

## Checked replay evidence and fit accounting

The exported `clean-copy/setup_restore.json` and `clean-copy/receipt-verification.json` verify Python, all pins, Base, eighteen exact selection restores, all eighteen altered-record rejections, frozen scientific identity and planning-path resolution after a simulated archive. The disposable copy contained no prior model/prediction caches. Both primary and clean full suites passed 2136 tests (221 current), with eighteen current and 85 historical actual tiny model fits each. All seven invocations, including failed/focused histories, total 126 current and 255 historical tiny fits; default evidence copies are not extra invocations.

The clean copy shared the existing pinned virtual environment and a checksum-verified dataset symlink. It did not demonstrate fresh dependency installation or full-size clean-copy retraining. Those claims are deliberately absent. The installation commands above describe setup; tested versions are recorded in the receipts.

Scientific production work is separate: fourteen new native miniature fits, 36 full native development fits, 24 verified base reuses and fifteen derived reference reuses, plus 108 new blend calculations. Final work is bounded by the immutable selection and recorded in transfer_complete.json. Original nine-fit sampling/eight-fit older gates were audited, not newly fitted. Methodological fits, miniature fits, full fits, exact absent-source reproductions and reuse must never be pooled as one model count.

Codex assisted with protocol, implementation, tests, independent auditing and interpretation; official CatBoost API references are recorded in the protocol/findings. No public notebook was copied. Settings remain analyst-selected and open to group review.
