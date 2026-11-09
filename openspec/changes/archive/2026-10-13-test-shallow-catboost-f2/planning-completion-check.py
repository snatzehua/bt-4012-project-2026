"""Verify planning contracts only; never fit, infer or evaluate a candidate."""
from collections import Counter
from itertools import product
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def verify_hashes(items):
    for path, expected in items.items():
        assert Path(path).is_file() and sha(path) == expected, path


p = load(ROOT/'planning-evidence.json')
text = {name: (ROOT/name).read_text() for name in [
    'proposal.md', 'design.md', 'specs/baf-shallow-catboost-study/spec.md', 'tasks.md']}
status = json.loads(subprocess.check_output([
    'openspec', 'status', '--change', 'test-shallow-catboost-f2', '--json'], text=True))
assert status['isPlanningComplete'] and len(status['artifacts']) == 4
assert all(a['status'] == 'done' for a in status['artifacts'])
spec = text['specs/baf-shallow-catboost-study/spec.md']
requirements = re.findall(r'^### Requirement: (.+)$', spec, re.M)
scenarios = re.findall(r'^#### Scenario: (.+)$', spec, re.M)
assert len(requirements) == 9 and len(scenarios) == 12
assert spec.count('- **WHEN**') == spec.count('- **THEN**') == 12
tasks = re.findall(r'^- \[(.)\] (\d+\.\d+) (.+)$', text['tasks.md'], re.M)
assert len(tasks) == 14 and all(mark == ' ' for mark, _, _ in tasks)
task_ids = {n for mark, n, desc in tasks}
assert task_ids == {'1.1','1.2','2.1','2.2','2.3','3.1','3.2','3.3','4.1','4.2','5.1','5.2','5.3','5.4'}
assert all('verif' in desc.lower() or 'check' in desc.lower() for mark, n, desc in tasks)
assert '## Open Questions' not in text['design.md']
assert '`baf-shallow-catboost-study`' in text['proposal.md']

# Each delta requirement has explicit pending implementation/evidence tasks.
coverage = {
    requirements[0]: ['1.1','1.2','5.3','5.4'],
    requirements[1]: ['2.1','3.1','3.3'],
    requirements[2]: ['2.1','3.1','3.2','3.3'],
    requirements[3]: ['2.2','3.3'],
    requirements[4]: ['2.2','3.3'],
    requirements[5]: ['2.2','3.2','3.3','5.2'],
    requirements[6]: ['2.3','4.1','4.2'],
    requirements[7]: ['2.3','3.1','3.3','4.2','5.1','5.2','5.3','5.4'],
    requirements[8]: ['5.2','5.4']}
assert all(set(v) <= task_ids for v in coverage.values())

strategies = {s['id']: s for s in p['strategies']}
assert len(strategies) == 54 and Counter(s['kind'] for s in strategies.values()) == {
    'reference': 6, 'native': 12, 'blend': 36}
new = [s for s in strategies.values() if s['kind'] != 'reference']
coordinates = {(s['depth'],s['iterations'],s['positive_class_cost'],s['alpha_numerator']) for s in new}
assert coordinates == set(product([3,4,6],[200,400],[1,100],[10,25,50,100]))
assert len(p['recipes']) == 12
for recipe in p['recipes']:
    expected = dict(p['catboost_common_params'], depth=recipe['depth'],
        iterations=recipe['iterations'], class_weights=[1,recipe['positive_class_cost']],
        cat_features=p['categorical_columns'])
    assert recipe['params'] == expected
    assert not {'sample_weight','scale_pos_weight','auto_class_weights'} & set(expected)
assert len(p['winner_policy_cases']) == 162
assert {(v['winner'],v['policy']) for v in p['winner_policy_cases']} == set(
    product(strategies,['pooled','latest','rank_fraction']))
for case in p['winner_policy_cases']:
    s = strategies[case['winner']]
    roles = case['roles']
    assert roles['selected'] == [s['id'],case['policy']]
    assert roles['same_strategy'] == [s['id'],'pooled']
    assert all(roles[k] == v for k,v in p['reference_roles'].items())
    dag = case['dependencies']
    assert set(dag['old_bases']) == set(p['reference_dag']['bases'])
    if s['kind'] == 'reference':
        assert len(roles) == 8 and not dag['catboost_bases'] and not dag['new_blends']
    else:
        assert len(roles) == 22
        d,r,w,a = [s[k] for k in ['depth','iterations','positive_class_cost','alpha_numerator']]
        controls = {(v,r,w) for v in [3,4,6]} | {(d,v,w) for v in [200,400]} | {(d,r,v) for v in [1,100]}
        actual = {(strategies[n]['depth'],strategies[n]['iterations'],strategies[n]['positive_class_cost'])
                  for n in dag['catboost_bases']}
        assert actual == controls and len(actual) == 5
        for role, (n,k) in roles.items():
            if role not in {'selected','same_strategy',*p['reference_roles']}:
                assert k == case['policy']
        assert len(dag['new_blends']) == (3 if s['kind']=='native' else 7)
    assert case['model_object_bound'] == 8+len(dag['catboost_bases'])
    assert case['model_file_bound'] == 9+2*len(dag['catboost_bases'])
    assert case['physical_score_bound'] == len(dag['full_scores']) == {'reference':13,'native':21,'blend':25}[s['kind']]
    assert case['paired_interval_rows'] == 4*(len(roles)-1)

# Check full factor identities and held-constant coordinates independently.
contrasts = p['contrasts']
assert len(contrasts) == len({json.dumps(v,sort_keys=True) for v in contrasts}) == 1728
assert dict(Counter(v['kind'] for v in contrasts)) == p['contrast_kind_counts']
for v in contrasts:
    n,k,b,bk,kind = [v[f] for f in ['candidate','policy','comparator','comparator_policy','kind']]
    assert n in strategies and b in strategies
    if kind == 'fixed_reference':
        assert [b,bk] == p['reference_roles'][v['factors']['reference']]
    elif kind == 'own_policy':
        assert n == b and bk == 'pooled' and k in ['latest','rank_fraction']
    else:
        assert k == bk
        first, second = strategies[n], strategies[b]
        if kind in ['matched_depth','matched_rounds','matched_cost','higher_alpha']:
            field = {'matched_depth':'depth','matched_rounds':'iterations',
                     'matched_cost':'positive_class_cost','higher_alpha':'alpha_numerator'}[kind]
            for f in ['depth','iterations','positive_class_cost','alpha_numerator']:
                assert first[f] > second[f] if f == field else first[f] == second[f]
        elif kind == 'blend_vs_native':
            assert first['kind'] == 'blend' and second['kind'] == 'native'
            assert all(first[f] == second[f] for f in ['depth','iterations','positive_class_cost'])
        else:
            assert b == p['reference_roles']['bagged_anchor'][0]
            assert first['kind'] == ('native' if kind=='native_vs_anchor' else 'blend')

expected_rows = dict(physical_scores=183,selectable_scores=162,forward=324,
    macro=162,pooled=162,models=60,new_parameters=36,diversity=36,overlap=504,curves=180)
assert {k:len(v) for k,v in p['row_identity_sets'].items()} == expected_rows
assert all(len(v) == len({tuple(row) for row in v}) for v in p['row_identity_sets'].values())
assert p['new_model_fits'] == p['new_score_evaluations'] == 0
assert len(p['mandatory_preservation_sha256']) == 1860
verify_hashes(p['mandatory_preservation_sha256'])
assert len(p['historical_ignored_file_sha256']) == 4848
verify_hashes(p['historical_ignored_file_sha256'])
actual_history = {str(f) for root in p['historical_output_roots'] for f in Path(root).rglob('*') if f.is_file()}
assert actual_history == set(p['historical_ignored_file_sha256'])
selections = [n for n in p['mandatory_preservation_sha256'] if n.startswith('outputs/') and n.endswith('/selection.json')]
assert len(selections) == 18 and all(p['mandatory_preservation_sha256'][n] == p['mandatory_preservation_sha256'][n.replace('outputs/','reports/',1)] for n in selections)
verify_hashes(p['parent_fingerprint']['sources'])
assert len(p['planned_source_paths']) == len(set(p['planned_source_paths'])) == 61
assert len(p['reference_cache_presence']) == 32 and all(v['complete_origin_present'] for v in p['reference_cache_presence'])
assert p['reference_adapter_config'] == 'configs/bagged-anchor-fusion.json'
assert sha(p['parent_config']) == p['parent_fingerprint']['config']
for path in ['configs/shallow-catboost.json','docs/shallow-catboost-protocol.md',
    'baf/shallow_catboost.py','baf/shallow_catboost_study.py','baf/shallow_catboost_report.py',
    'tests/test_shallow_catboost.py','scripts/check_shallow_catboost.py',
    'outputs/baf-shallow-catboost-v1','reports/baf-shallow-catboost-v1']:
    assert not Path(path).exists(), 'Unexpected implementation/scientific artifact: '+path
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
validation = (ROOT/'planning-openspec-validation.log').read_text()
assert '0 failed' in validation and '26 passed' in validation
for name in ['README.md','AGENTS.md','openspec/config.yaml']:
    content = Path(name).read_text()
    assert 'test-shallow-catboost-f2' in content and '0/14' in content
    assert '4848' in content and '1860' in content

receipt = dict(planning_complete=True,implementation_tasks_complete=0,
    implementation_tasks_total=14,requirements=9,scenarios=12,
    requirement_task_coverage=coverage,expected_row_counts=expected_rows,
    strategies=54,winner_policy_cases=162,contrast_identities=1728,
    mandatory_paths=1860,historical_files=4848,selection_pairs=18,
    strict_openspec_passed=26,strict_openspec_failed=0,
    new_model_fits=0,new_score_evaluations=0,
    previous_goal_turn_classification='progress',
    previous_goal_turn_evidence='18th conditional study completed/archived; frozen outputs and all predecessor identities remain verified. Its negative result motivates the distinct shallow/short native grid.',
    current_goal_turn_classification='progress',
    broader_goal_achieved=False,
    reason='Verified prospective plan; no new experimental performance evidence yet.',
    planning_inputs={str(ROOT/n):sha(ROOT/n) for n in [*text,'planning-evidence.json','planning-enumeration.py']},
    context_snapshots={n:sha(n) for n in ['README.md','AGENTS.md','openspec/config.yaml']})
(ROOT/'planning-completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in [
    'requirement_task_coverage','planning_inputs','context_snapshots']},indent=2))
