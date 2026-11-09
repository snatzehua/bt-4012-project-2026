from pathlib import Path
import json, re, shutil, xml.etree.ElementTree as ET
from baf.common import read_config, digest, write_json
from baf import unweighted_fusion_study as r
import pandas as pd

dest=Path('reports/baf-unweighted-reference-fusion-v1')
out=Path('outputs/baf-unweighted-reference-fusion-v1')
change=Path('openspec/changes/test-unweighted-reference-fusion-f2')
def checked(value, message):
    if not value: raise AssertionError(message)
def junit(name, expected):
    root=ET.parse(dest/name).getroot()
    cases=list(root.iter('testcase'))
    checked(len(cases)==expected, name+' count')
    checked(not list(root.iter('failure')) and not list(root.iter('error')), name+' failures')
    return cases

c,s=r.load_selection()
fp=r.fingerprint()
checked(len(fp['sources'])==49,'source closure')
checked(fp==read_config(dest/'audit.json')['fingerprint'],'frozen reporting identity')
r.verify_hashes(read_config(dest/'generated_report_hashes.json'))
checked(len(read_config(dest/'generated_report_hashes.json'))==53,'original generated export count')
for original,copy in [(r.CONFIG,'config.json'),(c['protocol_path'],'protocol.md'),(out/'selection.json','selection.json')]:
    checked(digest(original)==digest(dest/copy),'copied identity '+copy)
initial=read_config(out/'preimplementation_preservation.json')
r.verify_hashes(initial)
checked(len(initial)==3092,'historical available snapshot')
revisions=read_config(out/'implementation_revisions.json')
r.verify_hashes(revisions['prior_attempt_files'])
checked(len(revisions['prior_attempt_files'])==87,'stopped-format attempt')
done=read_config(out/'validation_complete.json')
r.verify_hashes({str(out/p):h for p,h in done['files'].items()})
aud=read_config(dest/'independent_development_audit.json')
checked(aud['validation_sha256']==digest(out/'validation_complete.json'),'validation audit binding')
checked(aud['independent_statistics_inference_and_gradient'] and aud['independent_policies_metrics_factors'],'full audit scope')
for key,n in dict(scores=132,selectable_scores=120,operating_rows=240,macro_rows=120,contrasts=1199,clipping_rows=18,diversity_rows=48,feature_rows=522,pair_rows=18,coefficient_rows=1728,saved_model_checks=18).items():
    checked(aud[key]==n,key)
for name,n in {'crossfit':240,'summary':120,'pooled':120,'score_diagnostics':120,'contrasts':1199,'clipping':18,'diversity':48,'feature_diagnostics':522,'pair_diagnostics':18,'coefficients':1728,'development_draws':1000,'consumed_draws':500,'development_intervals':18,'consumed_intervals':6,'exploratory':7}.items():
    checked(len(pd.read_csv(dest/(name+'.csv')))==n,'export '+name)
checked(s['created_at_utc']<read_config(out/'exploratory_access.json')['access_at_utc'],'freeze before access')
checked(read_config(out/'exploratory_access.json')['selection_sha256']==digest(out/'selection.json'),'access binding')
transfer=read_config(dest/'transfer_complete.json')
checked(transfer['new_fits']==0 and transfer['base_reuses']==5 and transfer['total_scores']==8,'actual transfer work')
checks=read_config(dest/'checks.json')
checked(checks['test_source_sha256']==digest(dest/'methodological-source-at-selection.py'),'execution test source')
checked(checks['junit_sha256']==digest(dest/'tests.xml'),'execution junit')
junit('tests.xml',922)
cases=junit('clean-replay-tests.xml',941)
checked(sum('test_unweighted_fusion' in x.get('classname','') for x in cases)==322,'current-study tests')
portable=read_config(dest/'portable_replay_verification.json')
checked(portable['source_test_sha256']==digest(dest/'methodological-source-portable-replay.py'),'portable test source')
checked(portable['total_new_cache_test_fits']==28 and portable['pinned_packages_verified']==27,'test/environment accounting')
for phase in portable['phases']:
    checked(phase['junit_sha256']==digest(phase['junit']),'test phase bytes')
clean=read_config(dest/'clean_replay_preparation.json')
checked(clean['prior_models']==clean['prior_prediction_caches']==0 and clean['selection_restores']==14 and clean['changed_selection_rejected'],'clean restore')
checked(digest(c['data_path'])==c['data_sha256']==clean['data_sha256'],'data checksum')
figure=read_config(dest/'figure_verification.json')
checked(figure['figure_sha256']==digest(dest/'fusion_operating.png') and figure['visually_inspected'] and figure['operating_points_checked']==120 and figure['numerical_regeneration_identical'],'figure')
report=read_config(dest/'audit.json')
checked(report['direct_draws_checked']==1500 and not report['measured_gain_on_both_scopes'],'goal acceptance')
readme=Path('README.md').read_text()
section=readme[readme.index('## Unweighted-reference'):readme.index('## Acknowledging assistance')]
stages=['pytest -q','record_checks','preflight','prepare_development','validate','freeze','exploratory','unweighted_fusion_report report']
commands=next(b for b in re.findall(r'```bash\n(.*?)```',section,re.S) if 'pytest -q' in b)
locations=[commands.index(x) for x in stages]
checked(locations==sorted(locations),'README ordered stages')
checked('assert len(pairs) == 14' in section and 'digest(original) == expected' in section and 'Preserve existing frozen' in section and '941' in section,'README reproduction contracts')
checked('Codex assisted with the unweighted-reference' in readme,'assistance attribution')
for p in ['AGENTS.md','openspec/config.yaml']:
    checked('test-unweighted-reference-fusion-f2' in Path(p).read_text() and '941' in Path(p).read_text(),'current context '+p)

requirements=[
 ('Separate finite preserved addition study',['declaration.json','preparation_verified.json','implementation_revisions.json','clean_replay_preparation.json'],['test_four_missing_cache_formats_fit_once','test_partial_required_cache_never_falls_back_or_fits']),
 ('Fixed domain-correct label-free margin addition',['independent_development_audit.json'],['test_composition_arithmetic_domains_and_label_batch_invariance','test_invalid_compositions_rejected']),
 ('Canonical historical identity and reference precedence',['freeze_verified.json','preparation_verified.json'],['test_alias_uses_historical_direct_arithmetic_and_retains_ties','test_all_winner_policy_roles']),
 ('Comparable chronological actual-F2 selection',['independent_development_audit.json','selection.json','crossfit.csv'],['test_complete_analytical_export_and_independent_audit']),
 ('Complete matched effects and diagnostic evidence',['contrasts.csv','matched-effects.md','feature_blocks.json','interaction_blocks.json','numerical.csv','coefficients.csv'],['test_independent_audit_rejects_corrupted_export','test_constant_diversity_has_null_correlations_and_full_joint_counts']),
 ('Frozen bounded exploratory transfer and honest acceptance',['freeze_verified.json','transfer_verified.json','exploratory_access.json','audit.json','findings.md'],['test_all_winner_policy_roles','test_exact_final_type_bounds_and_score_dag']),
 ('Verified reproducible completion and retained gate failures',['portable_replay_verification.json','figure_verification.json','reproduction-verification.md','implementation-history.md','postfreeze-test-history.md'],['test_failed_gate_evidence_prevents_retry','test_resource_preflight_retains_failure_and_prevents_retry','test_paired_uncertainty_retains_every_role_and_alias'])]
spec=(change/'specs/baf-unweighted-reference-fusion-study/spec.md').read_text()
checked(re.findall(r'^### Requirement: (.+)$',spec,re.M)==[x[0] for x in requirements],'every requirement mapped')
names=[x.get('name','') for x in cases]
mapped=[]
for title,files,tests in requirements:
    checked(all((dest/f).is_file() for f in files),'requirement artifacts '+title)
    checked(all(any(n.startswith(t) for n in names) for t in tests),'requirement methodological coverage '+title)
    mapped.append(dict(requirement=title,evidence={str(dest/f):digest(dest/f) for f in files},test_families=tests,status='verified'))
scenarios=re.findall(r'^#### Scenario: (.+)$',spec,re.M)
checked(len(scenarios)==11,'scenario enumeration')
scenario_tests=[['test_four_missing_cache_formats_fit_once'],['test_partial_required_cache_never_falls_back_or_fits'],['test_composition_arithmetic_domains_and_label_batch_invariance'],['test_composition_arithmetic_domains_and_label_batch_invariance'],['test_alias_uses_historical_direct_arithmetic_and_retains_ties'],['test_complete_analytical_export_and_independent_audit'],['test_complete_analytical_export_and_independent_audit'],['test_all_winner_policy_roles'],['test_complete_analytical_export_and_independent_audit'],['test_failed_gate_evidence_prevents_retry','test_resource_preflight_retains_failure_and_prevents_retry'],['test_complete_analytical_export_and_independent_audit']]
for tests in scenario_tests: checked(all(any(n.startswith(t) for n in names) for t in tests),'scenario tests')
tasks=(change/'tasks.md').read_text()
checked(tasks.count('- [x]')==12 and tasks.count('- [ ]')==2,'expected pending tasks')
(change/'tasks.md').write_text(tasks.replace('- [ ] 5.3','- [x] 5.3',1))
write_json(dest/'completion_requirements.json',dict(created_at_utc=r.now(),requirements=mapped,scenarios=[dict(scenario=n,test_families=t,status='verified') for n,t in zip(scenarios,scenario_tests)],production_fingerprint=fp,mandatory_preservation_paths=529,ancestor_selection_pairs=14,historical_ignored_files=len(initial),stopped_format_attempt_files=87,original_generated_report_files=53,implementation_tasks_verified=13,final_archive_manifest_pending=True,goal_achieved=False))
shutil.copyfile('/tmp/unweighted-prearchive-tests.xml',dest/'prearchive-tests.xml')
shutil.copyfile('/tmp/unweighted-prearchive-tests.log',dest/'prearchive-tests.log')
shutil.copyfile('tests/test_unweighted_fusion.py',dest/'methodological-source-archive-portability.py')
print('Verified all seven requirements, eleven scenarios and task 5.3; final closure pending.')
