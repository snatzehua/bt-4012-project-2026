"""Verify prospective artifacts and preservation only; never fit or evaluate a model."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.metadata
import json
import re
import sys

root = Path(__file__).resolve().parent
p = json.loads((root / 'planning-evidence.json').read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

for group in ('mandatory_preservation_sha256', 'historical_ignored_file_sha256'):
    for name, expected in p[group].items():
        assert Path(name).is_file(), name
        assert sha(name) == expected, name
    print(group, len(p[group]), 'verified', flush=True)
for name in p['historical_output_roots']:
    actual = {str(f) for f in Path(name).rglob('*') if f.is_file()}
    expected = {f for f in p['historical_ignored_file_sha256'] if f.startswith(name + '/')}
    assert actual == expected, (name, actual ^ expected)
absent = p['planned_source_paths'][-3:] + [
    'configs/conditional-lightgbm-experts.json',
    'docs/conditional-lightgbm-experts-protocol.md',
    'tests/test_conditional_experts.py', 'scripts/check_conditional_experts.py',
    'outputs/baf-conditional-lightgbm-experts-v1',
    'reports/baf-conditional-lightgbm-experts-v1',
]
for name in absent:
    assert not Path(name).exists(), name
s, cases, factors = p['strategies'], p['winner_policy_cases'], p['contrasts']
assert len(s) == len({x['id'] for x in s}) == 24
assert len(cases) == len({(x['winner'], x['policy']) for x in cases}) == 72
assert len(factors) == len({json.dumps(x, sort_keys=True) for x in factors}) == 738
assert Counter(len(x['roles']) for x in cases) == {8: 18, 10: 6, 17: 48}
assert max(len(x['dependencies']['full_scores']) for x in cases) == 22
assert max(x['current_new_capacity_fit_bound'] + x['current_new_expert_fit_bound'] for x in cases) == 15
assert p['new_model_fits'] == p['new_policy_or_composition_evaluations'] == 0
assert len(p['planned_source_paths']) == len(set(p['planned_source_paths'])) == 58
for pkg, version in p['verified_pins'].items():
    assert importlib.metadata.version(pkg) == version
assert sys.version.split()[0] == p['python'] == '3.12.9'
assert sha('data/raw/baf/Base.csv') == p['data_sha256']
status = json.loads((root / 'planning-status.json').read_text())
assert status['isPlanningComplete'] is True
assert all(x['status'] == 'done' for x in status['artifacts'])
# CLI isComplete denotes artifacts; actual task checkboxes prove implementation status.
tasks = (root / 'tasks.md').read_text()
spec = (root / 'specs/baf-conditional-lightgbm-expert-study/spec.md').read_text()
assert len(re.findall(r'^- \[ \] \d+\.\d+ ', tasks, re.M)) == 14
assert '- [x]' not in tasks
assert len(re.findall(r'^### Requirement:', spec, re.M)) == 8
assert len(re.findall(r'^#### Scenario:', spec, re.M)) == 12
assert not Path('openspec/specs/baf-conditional-lightgbm-expert-study').exists()
assert 'Totals: 25 passed, 0 failed (25 items)' in (root / 'planning-openspec-validation.log').read_text()
receipt = {
    'status': 'planning_complete_implementation_pending',
    'tasks_complete': 0, 'tasks_total': 14, 'requirements': 8, 'scenarios': 12,
    'new_model_fits': 0, 'new_proposed_evaluations': 0,
    'mandatory_paths_verified': 1053, 'historical_files_verified': 4369,
    'selection_pairs': 17, 'strategies': 24, 'winner_policy_cases': 72,
    'contrast_identities': 738, 'planned_sources': 58,
    'max_visible_roles': 17, 'max_full_transfer_scores': 22,
    'observed_training_only_max_new_transfer_fits': 15,
    'python': sys.version.split()[0], 'pins_verified': len(p['verified_pins']),
    'data_sha256': p['data_sha256'], 'strict_openspec': '25 passed,0 failed',
    'previous_goal_execution_turn': 'progress: seventeenth study completed with verified negative comparisons; intervening budget clarification did not execute a study',
    'current_progress': 'old development diagnostic changes next action; conditional experiment declared prospectively',
    'goal_achieved': False, 'implementation_absence_verified': True,
    'initial_mapping_failure_retained': True,
    'initial_status_semantics_failure_retained': True,
}
(root / 'planning-completion-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
