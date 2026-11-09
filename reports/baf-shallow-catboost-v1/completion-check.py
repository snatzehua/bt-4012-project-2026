"""Verify finite-study evidence; never fit, score, tune or change frozen sources."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import xml.etree.ElementTree as ET
import pandas as pd
from baf import shallow_catboost_study as run
from baf.common import digest, read_config, write_json

D=Path('reports/baf-shallow-catboost-v1')
c,s=run.load_selection(); O=Path(c['output_dir']); fp=run.fingerprint()
assert s['fingerprint']==fp and len(fp['sources'])==61
run.guard(c)
preserve=read_config(O/'preservation.json')
assert len(preserve['mandatory'])==1860 and len(preserve['optional_present'])==4848
run.verify_hashes(preserve['mandatory']);run.verify_hashes(preserve['optional_present'])
for v in read_config(D/'planning-original/copy-verification.json').values():
    assert digest(v['preserved_copy'])==v['original_sha256']
history=read_config(D/'implementation-history.json');run.verify_hashes(history['artifacts'])
assert history['production_execution_sources_edited_after_freeze'] is False
checks=read_config(O/'checks.json');assert checks['fingerprint']==fp
assert checks['tests_passed']==2136 and checks['own_tests']==221
assert checks['actual_current_test_fits']==18 and checks['actual_historical_test_fits']==85
root=ET.parse(D/'tests.xml').getroot();cases=list(root.iter('testcase'))
assert len(cases)==2136 and not any(list(root.iter(n)) for n in ['failure','error','skipped'])
assert checks['junit_sha256']==digest(D/'tests.xml')
ledger=read_config(D/'test_model_fits.json');run.verify_hashes(ledger['execution_source_sha256'])
assert checks['actual_fit_ledger_sha256']==digest(D/'test_model_fits.json')
inv=read_config(D/'test_invocation_history.json');assert len(inv['closed_invocations'])==7
assert (inv['current_tiny_fit_attempts'],inv['historical_fit_attempts'])==(126,255)
for r in inv['closed_invocations']:assert digest(r['ledger'])==r['sha256']
gate=read_config(D/'execution-gates-verification.json');run.verify_hashes(gate['gate_files']);assert gate['production_miniature_fits']==14
prep=read_config(O/'preparation.json');assert prep['invocation_new_native_fits']==36 and prep['invocation_exact_reference_fits']==0
assert prep['reference_reuses']==24 and prep['reference_derived_reuses']==15
assert prep['new_compositions']==108 and prep['base_results']==60
expected={'crossfit':324,'summary':162,'pooled':162,'contrasts':1728,'physical_score_diagnostics':183,'score_diagnostics':162,'diversity':36,'decision_overlap':504}
for n,count in expected.items():assert len(pd.read_csv(O/(n+'.csv')))==count
assert len(pd.read_csv(O/'model_work.csv'))==60
for n in ['native_parameters.json','native_preprocessing.json']:assert len(read_config(O/n))==36
original=read_config(O/'independent_development_audit.json');audit=read_config(D/'audit.json')
assert original['saved_model_checks']==60 and original['actual_source_models_audited'] is True
assert audit['development']['saved_model_checks']==60 and audit['development']['actual_source_models_audited'] is True
assert audit['development']['counts']==expected and audit['selection_sha256']==digest(O/'selection.json')
assert audit['final_models']==8 and audit['visible_roles']==8
freeze=read_config(D/'freeze-verification.json');assert freeze['prospective_cases_verified']==162 and freeze['selection_bytes_unchanged'] is True
assert freeze['selection_sha256']==digest(O/'selection.json') and 'differ' in freeze['changed_dependency_rejection']
b=s['final_fit_bounds'];assert (b['unique_base_results'],b['physical_scores'],b['model_files'],b['max_new_native_fits'])==(8,13,9,0)
transfer=read_config(O/'transfer_complete.json');assert (transfer['role_count'],transfer['model_artifacts'],transfer['prediction_artifacts'],transfer['model_files'])==(8,8,13,9)
assert transfer['invocation_new_native_fits']==transfer['invocation_exact_reference_fits']==0
assert transfer['reference_reuses']==8 and transfer['reference_derived_reuses']==5
access=read_config(O/'exploratory_access.json');assert s['created_at_utc']<access['access_at_utc']
assert access['selection_sha256']==transfer['selection_sha256']==digest(O/'selection.json')
assert transfer['table_sha256']==digest(O/'exploratory.csv')
for n in run.TABLES+run.WORK_TABLES+['selection.json','exploratory.csv','exploratory_access.json','transfer_complete.json']:
    assert digest(O/n)==digest(D/n)
di=pd.read_csv(D/'development_intervals.csv');fi=pd.read_csv(D/'consumed_intervals.csv')
dd=pd.read_csv(D/'development_draws.csv');fd=pd.read_csv(D/'consumed_draws.csv')
assert (len(di),len(fi),len(dd),len(fd))==(21,7,1000,500)
for scope in ['5','6','macro']:assert (di.scope.astype(str)==scope).sum()==7
for alias in ['bagged_anchor','same_strategy']:
    for t in [di,fi]:assert (t[t.reference==alias][['difference','ci_low','ci_high']].to_numpy()==0).all()
    assert (dd[alias]==0).all() and (fd[alias]==0).all()
assert audit['figure']==dict(operating_points=162,curves=36,endpoint_rows=180,matched_effects=288,reference_intervals=12)
figure=read_config(D/'figure-verification.json');assert figure['regenerated_png_identical'] is True and figure['visually_inspected'] is True
assert figure['png_sha256']==digest(D/'shallow_catboost_operating.png')
clean=read_config(D/'clean-copy/receipt-verification.json');run.verify_hashes(clean['files'])
setup=read_config(D/'clean-copy/setup_restore.json');assert len(setup['restored'])==len(setup['changed_selection_rejections'])==18
assert setup['no_prior_model_prediction_caches'] is True and len(setup['verified_pins'])==27
assert clean['tests']==2136 and clean['source_identity_matches_primary'] is True and clean['full_size_clean_copy_retraining'] is False
setup2=read_config(D/'clean-copy/documented-setup-verification.json');assert setup2['exit_code']==0
assert setup2['closed_log_sha256']==digest(D/'clean-copy/documented-setup.log')
assert audit['measured_gain_against_all_six_in_both_scopes'] is False
assert audit['development_gain']['bagged_anchor']==audit['consumed_gain']['bagged_anchor']==0
assert audit['consumed_gain']['unweighted']<0
change=run.planning_evidence_path().parent;spec=(change/'specs/baf-shallow-catboost-study/spec.md').read_text();tasks=(change/'tasks.md').read_text()
requirements=[x.removeprefix('### Requirement: ') for x in spec.splitlines() if x.startswith('### Requirement: ')]
scenarios=[x.removeprefix('#### Scenario: ') for x in spec.splitlines() if x.startswith('#### Scenario: ')]
assert len(requirements)==9 and len(scenarios)==12
complete=tasks.count('- [x] ');assert complete in [13,14]
proofs=[['contract_verification.json','preservation.json','clean-copy/setup_restore.json'],['native_preprocessing.json','independent_development_audit.json','tests.xml'],['native_parameters.json','miniature_gate.json','tests.xml'],['contrasts.csv','independent_development_audit.json'],['crossfit.csv','summary.csv','independent_development_audit.json'],list(expected)+['model_work.csv'],['freeze-verification.json','transfer_complete.json','audit.json'],['checks.json','execution-gates-verification.json','clean-copy/receipt-verification.json','figure-verification.json','audit.json'],['audit.json','findings.md']]
result=dict(checked_at_utc=run.now(),stage='postarchive' if complete==14 and change.parent.name=='archive' else 'prearchive',scientific_fingerprint=fp,selection_sha256=digest(O/'selection.json'),preserved_mandatory_files=1860,preserved_historical_files=4848,requirements=[dict(requirement=n,evidence=e,verified=True) for n,e in zip(requirements,proofs)],scenarios=[dict(scenario=n,verified=True,evidence='2136 passed exact methodological identities plus direct real-model/clean-copy/final audits above') for n in scenarios],tasks_complete=complete,tasks_total=14,development_audited_models=60,transfer_audited_models=8,bootstrap_draws=1500,interval_rows=28,negative_result_preserved=True,broader_goal_achieved=False,manifests_require_separate_closed_log_verification=True)
write_json(D/'completion-audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['scientific_fingerprint','requirements','scenarios']},indent=2))
