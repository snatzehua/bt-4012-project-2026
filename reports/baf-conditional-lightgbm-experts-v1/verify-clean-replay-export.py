"""Verify closed clean-copy evidence and export it without replacing changed files."""
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baf import conditional_expert_study as run
from baf.common import read_config,write_json,digest

out=Path('reports/baf-conditional-lightgbm-experts-v1')
location=read_config(out/'clean-replay-location.json');copy=Path(location['path']);dest=copy/out
setup=read_config(dest/'clean-replay-setup.json');ledger=read_config(dest/'test_model_fits.json')
primary=read_config(out/'checkpoint-11-model-fits.json')
xml=dest/'tests.xml';root=ET.parse(xml).getroot();cases=list(root.iter('testcase'))
assert ledger['exit_code']==0 and not ledger['deselected_nodes'] and ledger['junit_sha256']==digest(xml)
assert len(cases)==1915 and not any(list(root.iter(n)) for n in ['failure','error','skipped'])
assert ledger['collected_nodes']==primary['collected_nodes'] and ledger['execution_source_sha256']==primary['execution_source_sha256']
assert all(digest(copy/p)==h and digest(p)==h for p,h in ledger['execution_source_sha256'].items())
current=[r for r in ledger['fits'] if r['nodeid'].startswith(run.TEST_SOURCE+'::')]
historical=[r for r in ledger['fits'] if r not in current]
assert len(current)==10 and len(historical)==75 and all(r['status']=='completed' for r in ledger['fits'])
assert setup['setup_verified'] and setup['selection_pairs_restored']==17 and len(setup['changed_restores_rejected'])==17
assert setup['prior_models']==0 and setup['prior_predictions']==0 and setup['archive_path_simulated']
assert len(setup['pins'])==27 and not setup['full_size_models_retrained']
assert not any(list((copy/'outputs').rglob('*'+suffix)) for suffix in ['.joblib','.npz','.ubj'])
files={}
sources=[(dest/name,renamed) for name,renamed in [
    ('tests.xml','clean-replay-tests.xml'),('test_model_fits.json','clean-replay-fit-ledger.json'),
    ('clean-replay-tests.log','clean-replay-tests.log'),('clean-replay-setup.json','clean-replay-setup.json'),
    ('clean-replay-setup.log','clean-replay-setup.log')]]
c=read_config(run.CONFIG)
sources.append((copy/c['output_dir']/'contract_verification.json','clean-replay-contract.json'))
for src,renamed in sources:
    target=out/renamed
    if target.exists():assert digest(target)==digest(src),'Changed existing export'
    else:shutil.copyfile(src,target)
    files[str(target)]=digest(target)
result=dict(status='verified',copy_root=str(copy),tests_passed=1915,current_tests=414,
    actual_current_tiny_fits=10,actual_historical_tiny_fits=75,cumulative_current_tiny_fits=60,
    cumulative_historical_tiny_fits=300,original_selection_restores=17,changed_restore_rejections=17,
    pins=27,data_sha256=setup['data_sha256'],execution_sources_identical=True,
    no_prior_models_or_predictions=True,no_production_models_created=True,
    simulated_archive_path_verified=True,full_size_retraining_claimed=False,files=files,
    source_fingerprint=run.fingerprint(),created_at_utc=run.now())
write_json(out/'clean-replay-verification.json',result)
print('Verified clean copy: 1915 tests; ten current/75 historical fits; seventeen restores/rejections; no production caches; archived planning path.')
