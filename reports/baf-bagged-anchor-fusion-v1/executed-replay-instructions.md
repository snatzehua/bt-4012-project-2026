 to the anchor (implementation in progress)

The separate [proposal](openspec/changes/test-bagged-anchor-fusion-f2/proposal.md), [design](openspec/changes/test-bagged-anchor-fusion-f2/design.md), delta specification and [tasks](openspec/changes/test-bagged-anchor-fusion-f2/tasks.md) define the next finite experiment. It keeps every source model fixed and adds balanced/unweighted seed ensembles to the existing anchor at weights 0.1/0.25/0.5, comparing logit of the probability mean with mean of logits. The prospective grid contains 37 strategies, 120 physical development scores, 222 forward operating rows and 947 matched comparisons. Five fixed references and all constituent/representation/weighting/aggregation controls remain visible.

The isolated [configuration](configs/bagged-anchor-fusion.json), [protocol](docs/bagged-anchor-fusion-protocol.md), scoring/source-adapter modules and independent analytical auditor are implemented. Three of fourteen tasks are verified: prospective declaration/preservation, fixed composition and analytical-grid verification. The full suite passed **1367 tests**, including **297 current-study checks**. Eighteen current-study tiny source-format fits across focused/full invocations are separate from historical-module tests; failures and retries remain recorded. The exported synthetic fixture verifies 120 physical scores, 222 operating rows and 947 contrasts. Those synthetic results are methodology checks, not real-data performance evidence. See [implementation status](reports/baf-bagged-anchor-fusion-v1/implementation-status.md).

The verified 55-source declaration preserves 735 mandatory historical paths, sixteen selection pairs and 3732 historical files. The actual Base checksum, Python 3.12.9 and all 27 installed pins were checked. No full-data blends or new consumed scores have been evaluated, and no execution/selection identity has been frozen. Full-data resource/source-ancestry gates, preparation/audit, bounded transfer, uncertainty, findings and isolated reproduction remain pending. Only 7/9/21/26 predetermined roles can transfer after independent development audit/freeze, with at most 11 bases/28 scores/six bagged fits and tighter per-case/cache bounds. F2 beta=2 has no quota or precision floor; month7 is consumed and the broader improvement goal remains active. Settings are analyst-selected.

For reproduction use Python 3.12.9 and the same 27 dependency pins in the three requirements files. Verify Base v2 SHA256 `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809` using the data-access instructions above. Model seed is 2026, threads six, bootstrap seed 1027 and 500 paired fixed-decision draws per month. Keep existing frozen evidence intact. Set up a disposable copy without ignored outputs; the following links the pinned environment and read-only dataset. A new environment instead uses the pinned installation above:

```bash
set -e
baf_original_root="$PWD"
baf_replay_root="$(mktemp -d "${TMPDIR:-/tmp}/baf-bagged-anchor-replay.XXXXXX")"
rsync -a --exclude=.git --exclude=.venv --exclude=data --exclude=outputs --exclude=.pytest_cache --exclude=__pycache__ ./ "$baf_replay_root/"
ln -s "$baf_original_root/.venv" "$baf_replay_root/.venv"
mkdir -p "$baf_replay_root/data/raw/baf" "$baf_replay_root/replay-evidence"
ln -s "$baf_original_root/data/raw/baf/Base.csv" "$baf_replay_root/data/raw/baf/Base.csv"
mv "$baf_replay_root/reports/baf-bagged-anchor-fusion-v1" "$baf_replay_root/replay-evidence/"
mkdir -p "$baf_replay_root/reports/baf-bagged-anchor-fusion-v1"
cd "$baf_replay_root"
```

Restore exactly sixteen ancestor selections, rejecting changed records before copying missing ones:

```bash
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import shutil
from baf.common import read_config, digest
c = read_config('configs/bagged-anchor-fusion.json')
pairs = [(Path(p), Path(p.replace('outputs/', 'reports/', 1)), h)
         for p, h in c['preserve_sha256'].items()
         if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(pairs) == 16
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

Run stages in order. The test runner records actual classifier-fit calls, separating six current-study tiny format fits from historical test-module fits. `validate` performs the full independent development audit; the explicit audit command repeats that verification before freeze. Preflight verifies preserved nine-fit and older miniature ancestry with zero new production miniature fits. Wholly absent required caches reproduce only their exact individual source recipes; partial/incompatible caches fail without fallback. Fresh development reproduction can require eighteen bagged, six additive and nine tree fits. Actual final bounds depend on the selected roles and source availability, with at most six bagged, two additive and three old-tree fits. No unrelated grid is rerun.

```bash
set -e
.venv/bin/python -m baf.bagged_anchor_study verify_declaration
.venv/bin/python scripts/check_bagged_anchor.py
.venv/bin/python -m baf.bagged_anchor_study record_checks
.venv/bin/python -m baf.bagged_anchor_study preflight
.venv/bin/python -m baf.bagged_anchor_study prepare_development
.venv/bin/python -m baf.bagged_anchor_study validate
.venv/bin/python -m baf.bagged_anchor_report audit_development
.venv/bin/python -m baf.bagged_anchor_study freeze
.venv/bin/python -m baf.bagged_anchor_study exploratory
.venv/bin/python -m baf.bagged_anchor_report report
```

These scientific commands are implemented; full execution and clean-copy verification are still in progress. Conditional row intervals exclude training-seed/retraining, adaptive search/selection, temporal/generative and multiplicity uncertainty. Full-size production retraining in a disposable copy must not be claimed unless actually performed. Failed gates retain evidence and stop completion; frozen execution/test/selection/access/report records must not be overwritten. Further tuning needs a separate study.

