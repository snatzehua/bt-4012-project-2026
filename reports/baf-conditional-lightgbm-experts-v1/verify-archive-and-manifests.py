"""Verify archived completed study without declaring performance success."""
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baf import conditional_expert_study as run
from baf.common import read_config,write_json,digest

c,s=run.load_selection();out=Path(c['output_dir']);dest=Path(c['report_dir'])
change=run.planning_evidence_path().parent
assert change.parent.name=='archive' and change.name=='2026-10-12-test-conditional-lightgbm-experts-f2'
assert not Path('openspec/changes/test-conditional-lightgbm-experts-f2').exists()
tasks=(change/'tasks.md').read_text()
assert len(re.findall(r'^- \[x\] \d+\.\d+ ',tasks,re.M))==14 and '- [ ]' not in tasks
delta=next((change/'specs').glob('*/spec.md'));main=Path('openspec/specs/baf-conditional-lightgbm-expert-study/spec.md')
assert main.is_file()
def body(p):return p.read_text().split('### Requirement:',1)[1]
assert body(delta)==body(main)
for p in [delta,main]:assert len(re.findall(r'^### Requirement:',p.read_text(),re.M))==8
for p in [delta,main]:assert len(re.findall(r'^#### Scenario:',p.read_text(),re.M))==12
assert not re.search(r'^## (ADDED|MODIFIED|REMOVED|RENAMED) Requirements',main.read_text(),re.M)
closure=read_config(dest/'completion-requirements.json')
assert closure['status']=='passed' and closure['requirements_verified']==8 and closure['scenarios_verified']==12 and closure['tasks_verified']==14
assert not closure['broader_goal_achieved']
for grid in ['requirements','scenarios','tasks']:
    for row in closure[grid]:run.verify_hashes(row['evidence'])
doc=read_config(dest/'final-documentation-verification.json')
assert doc['status']=='passed' and doc['phase']=='final'
run.verify_hashes(doc['current_root_files']);run.verify_hashes(doc['snapshot_files'])
for name in ['README.md','AGENTS.md','openspec/config.yaml']:
    text=Path(name).read_text()
    assert 'All fourteen tasks are verified' in text and 'broader' in text and 'goal remains active' in text
readme=Path('README.md').read_text()
assert str(change)+'/tasks.md' in readme and 'openspec/changes/test-conditional-lightgbm-experts-f2/' not in readme
assert '0 failed' in (dest/'final-openspec-validation.log').read_text()
run.verify_hashes(read_config(dest/'generated_report_hashes.json'))
run.verify_hashes(read_config(dest/'run_hashes.json'))
science=read_config(dest/'scientific-results-verification.json');figure=read_config(dest/'figure-verification.json')
assert science['status']=='passed' and not science['broader_goal_achieved'] and figure['visually_inspected']
preserved=read_config(out/'preservation.json')
assert len(preserved['mandatory'])==1053 and len(preserved['optional_present'])==4369
run.guard(c)
write_json(dest/'archive-verification.json',dict(status='passed',archive=str(change),main_spec=str(main),
    archived_change_files={str(p):digest(p) for p in change.rglob('*') if p.is_file()},main_spec_sha256=digest(main),
    requirements_verified=8,scenarios_verified=12,tasks_verified=14,fingerprint=s['fingerprint'],
    selection_sha256=digest(out/'selection.json'),mandatory_paths_preserved=1053,
    historical_files_preserved=4369,original_selection_pairs=17,scientific_and_methodological_outputs_verified=True,
    closed_scientific_logs=True,all_old_report_indices_preserved=True,broader_goal_achieved=False,
    created_at_utc=run.now()))
print('Verified archived 18th finite study, synchronized 8 requirements/12 scenarios, all 14 tasks, frozen identities and preservation; broader goal remains active.')
