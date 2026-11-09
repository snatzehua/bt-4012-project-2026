"""Check preliminary implementation evidence; no fitting or new BAF evaluation."""
from pathlib import Path
from itertools import product
from collections import Counter
import importlib.metadata
import json
import sys
import shutil
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from baf import conditional_expert_study as run
from baf import conditional_expert_report as audit
from baf.common import read_config,write_json,digest

root=Path(__file__).resolve().parent
c=read_config(run.CONFIG);run.guard(c)
planning=read_config(run.planning_evidence_path())
run.verify_hashes(planning['historical_ignored_file_sha256'])
actual={str(p) for name in planning['historical_output_roots'] for p in Path(name).rglob('*') if p.is_file()}
assert actual==set(planning['historical_ignored_file_sha256'])
for pkg,version in planning['verified_pins'].items():assert importlib.metadata.version(pkg)==version
assert sys.version.split()[0]=='3.12.9' and digest(c['data_path'])==c['data_sha256']
full=read_config(root/'checkpoint-04-fit-ledger.json')
assert full['exit_code']==0
junit=root/'checkpoint-04-tests.xml';tree=ET.parse(junit).getroot();cases=list(tree.iter('testcase'))
assert len(cases)==1723 and not any(list(tree.iter(tag)) for tag in ['failure','error','skipped'])
assert len([x for x in cases if x.attrib['classname'].endswith('test_conditional_experts')])==222
assert full['junit_sha256']==digest(junit)
current=[r for r in full['fits'] if 'test_conditional_experts.py::' in r['nodeid']]
assert len(current)==10 and len(full['fits'])-len(current)==75
assert all(r['status']=='completed' for r in full['fits'])
changed=[]
for name,sha in full['execution_source_sha256'].items():
    assert digest(root/'checkpoint-04-sources'/name)==sha
    if digest(name)!=sha:changed.append(name)
assert set(changed)=={'baf/conditional_expert_study.py','baf/conditional_expert_report.py','tests/test_conditional_experts.py'}
latest=read_config(root/'checkpoint-05-analytical-fit-ledger.json')
assert latest['exit_code']==0 and latest['fits']==[]
run.verify_hashes(latest['execution_source_sha256'])
latest_xml=root/'checkpoint-05-analytical-tests.xml';tree=ET.parse(latest_xml).getroot()
assert len(list(tree.iter('testcase')))==214 and not any(list(tree.iter(tag)) for tag in ['failure','error','skipped'])
assert latest['junit_sha256']==digest(latest_xml)
cumulative=0
for name in ['checkpoint-01-fit-ledger.json','checkpoint-02-fit-ledger.json','checkpoint-03-analytical-fit-ledger.json','checkpoint-04-fit-ledger.json','checkpoint-05-analytical-fit-ledger.json']:
    ledger=read_config(root/name)
    cumulative+=sum('test_conditional_experts.py::' in r['nodeid'] for r in ledger['fits'])
assert cumulative==30
assert read_config(root/'checkpoint-02-fit-ledger.json')['exit_code']==1
assert not (Path(c['output_dir'])/'models').exists() and not (Path(c['output_dir'])/'predictions').exists()
assert not (Path(c['output_dir'])/'selection.json').exists()
assert not (Path(c['output_dir'])/'exploratory_access.json').exists()

# Reconstruct the synthetic frame independently and re-audit saved artifacts at current sources.
cells=list(product(['AA','AB','AC','AD'],['BA','BB','BC','BD','BE']))
frame=pd.DataFrame(dict(month=np.zeros(60000,dtype=int),payment_type=[x for x,y in cells for _ in range(3000)],
    housing_status=[y for x,y in cells for _ in range(3000)],fraud_bool=(np.arange(60000)%9==0).astype(int)))
future=[]
for m in [4,5,6]:
    future.append(pd.DataFrame(dict(month=[m]*42,
        payment_type=np.resize(np.array(['AA','AB','AC','AD','AE',None,'unknown'],dtype=object),42),
        housing_status=np.resize(np.array(['BA','BB','BC','BD','BE','BF','BG',None,'unknown'],dtype=object),42),
        fraud_bool=(np.arange(42)%7==0).astype(int))))
frame=pd.concat([frame,*future],ignore_index=True)
fixture=dict(c,output_dir=str(root/'synthetic-score-grid'))
fp={'fixture':'synthetic scores only; no classifier, preprocessor or operational-result claim'}
score_check=audit.audit_new_compositions(fixture,fp,frame,[4,5,6])
table_check=audit.audit_tables(fixture,frame)
assert score_check['member_subsets']==54 and score_check['new_full_compositions']==48
assert table_check['counts']['physical_score_diagnostics']==93
progress=read_config(root/'apply-status.json')['progress']
assert progress=={'total':14,'complete':4,'remaining':10}
assert 'Totals: 25 passed, 0 failed (25 items)' in (root/'checkpoint-openspec-validation.log').read_text()
for n in [4,5]:
    shutil.copyfile(Path(f'reports/baf-conditional-lightgbm-experts-v1-checkpoint-{n:02d}.log'),root/f'checkpoint-{n:02d}.log')
receipt=dict(status='four_tasks_verified_study_execution_pending',progress=progress,
    current_source_fingerprint=run.fingerprint(),configuration_sha256=digest(run.CONFIG),
    full_checkpoint_tests=1723,full_checkpoint_own_tests=222,current_nonfitting_tests=214,
    full_checkpoint_current_classifier_fits=10,full_checkpoint_historical_classifier_fits=75,
    cumulative_current_tiny_classifier_fits=30,full_checkpoint_status_counts=dict(Counter(r['status'] for r in full['fits'])),
    full_checkpoint_source_changes=changed,original_source_snapshot_files=len(full['execution_source_sha256']),
    mandatory_preserved_paths=1053,original_exported_selection_pairs=17,historical_preserved_files=4369,
    pins_verified=27,python=sys.version.split()[0],data_sha256=c['data_sha256'],
    current_synthetic_score_reaudit=score_check,current_synthetic_table_reaudit=table_check,
    actual_new_full_study_fits=0,actual_new_baf_policy_evaluations=0,new_month7_evaluations=0,
    production_execution_identity_frozen=False,goal_achieved=False,
    pending='final orchestration/checks, resource/source/16-fit miniature gates, full fits and audits, freeze/transfer, uncertainty/figures, clean-copy reproduction and completion')
write_json(root/'implementation-checkpoint-verification.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='current_source_fingerprint'},indent=2),flush=True)
