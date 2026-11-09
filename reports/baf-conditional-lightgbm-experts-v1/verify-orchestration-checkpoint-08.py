"""Verify the closed full-suite orchestration checkpoint before production execution."""
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baf import conditional_expert_study as run
from baf.common import read_config,write_json,digest

dest=Path('reports/baf-conditional-lightgbm-experts-v1')
c=read_config(run.CONFIG);run.guard(c)
ledger=read_config(dest/'checkpoint-08-model-fits.json')
xml=dest/'checkpoint-08.xml';root=ET.parse(xml).getroot()
cases=list(root.iter('testcase'))
assert ledger['exit_code']==0 and ledger['junit_sha256']==digest(xml)
assert not ledger['deselected_nodes'] and len(cases)==1907
assert not any(list(root.iter(n)) for n in ['failure','error','skipped'])
run.verify_hashes(ledger['execution_source_sha256'])
snapshot=read_config(dest/'checkpoint-08-sources/source_hashes.json')
assert snapshot==ledger['execution_source_sha256']
assert all(digest(dest/'checkpoint-08-sources'/p)==h for p,h in snapshot.items())
current=[r for r in ledger['fits'] if r['nodeid'].startswith(run.TEST_SOURCE+'::')]
historical=[r for r in ledger['fits'] if r not in current]
assert len(current)==10 and len(historical)==75
assert all(r['status']=='completed' for r in ledger['fits'])
assert sum(t.get('classname','').endswith('test_conditional_experts') for t in cases)==406
planning=read_config(run.planning_evidence_path())
run.verify_hashes(planning['historical_ignored_file_sha256'])
actual={str(p) for folder in c['historical_output_roots'] for p in Path(folder).rglob('*') if p.is_file()}
assert actual==set(planning['historical_ignored_file_sha256'])
assert len(actual)==4369 and len(c['preserve_sha256'])==1053
assert digest(c['data_path'])==c['data_sha256']
out=Path(c['output_dir'])
assert not any((out/n).exists() for n in ['declaration.json','selection.json','exploratory_access.json','validation_complete.json'])
for folder in ['models','runs','predictions']:
    assert not any((out/folder).glob('*'))
for source,target in [('checkpoint-08.xml','tests.xml'),('checkpoint-08-model-fits.json','test_model_fits.json')]:
    src,dst=dest/source,dest/target
    if dst.exists():assert digest(src)==digest(dst),'Changed canonical receipt'
    else:shutil.copyfile(src,dst)
checks=run.record_checks()
all_ledgers=[]
for p in sorted(dest.glob('checkpoint-*.json')):
    r=read_config(p)
    if 'collected_nodes' not in r or 'fits' not in r:continue
    assert r['exit_code'] is not None,p
    all_ledgers.append((p,r))
current_total=sum(row['nodeid'].startswith(run.TEST_SOURCE+'::') for p,r in all_ledgers for row in r['fits'])
historical_total=sum(not row['nodeid'].startswith(run.TEST_SOURCE+'::') for p,r in all_ledgers for row in r['fits'])
assert len(all_ledgers)==8 and current_total==40 and historical_total==150
result=dict(status='orchestration_methodology_verified_before_production',
    fingerprint=run.fingerprint(),checks=checks,full_suite_passed=1907,current_tests=406,
    actual_current_tiny_fits=10,actual_historical_tiny_fits=75,
    cumulative_current_tiny_fits=current_total,cumulative_historical_tiny_fits=historical_total,
    historical_files_preserved=4369,mandatory_paths_preserved=1053,
    source_snapshot_sha256=digest(dest/'checkpoint-08-sources/source_hashes.json'),
    checkpoint_ledger_sha256=digest(dest/'checkpoint-08-model-fits.json'),
    preliminary_source_snapshot_preserved=True,production_identity_frozen=False,
    new_full_data_fits=0,new_month7_access=False,performance_goal_achieved=False,
    recorded_at_utc=run.now())
write_json(dest/'orchestration-checkpoint-verification.json',result)
print('Verified 1907 tests, 10 current/75 historical fits, 40 cumulative current tiny fits; production not yet frozen.')
