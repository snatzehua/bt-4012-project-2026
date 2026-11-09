from pathlib import Path
import re,xml.etree.ElementTree as ET
import pandas as pd
from baf.common import read_config,write_json,digest
from baf import bagged_boosting as run
out=Path('outputs/baf-bagged-lightgbm-seeds-v1');dest=Path('reports/baf-bagged-lightgbm-seeds-v1');change=Path('openspec/changes/test-bagged-lightgbm-seed-ensembles-f2')
c=read_config(run.CONFIG);run.guard(c);s=read_config(out/'selection.json');a=read_config(dest/'audit.json')
assert s['fingerprint']==run.fingerprint() and a['fingerprint']==s['fingerprint']
assert not a['measured_gain_on_both_scopes'] and a['direct_draws_checked']==1500
assert a['saved_development_models_checked']==33 and a['saved_final_models_checked']==5
assert len(pd.read_csv(dest/'development_draws.csv'))==1000 and len(pd.read_csv(dest/'consumed_draws.csv'))==500
assert len(pd.read_csv(dest/'development_intervals.csv'))==18 and len(pd.read_csv(dest/'consumed_intervals.csv'))==6
assert read_config(dest/'figure_verification.json')['visual_inspection_complete']
assert read_config(dest/'reproduction_verification.json')['full_suite_passed']==1070
assert read_config(dest/'test_fit_counts.json')['total_new_cache_test_fits']==6
assert len(list((out/'failures').glob('*.json')))==0
for name in run.TABLES+['selection.json','exploratory_access.json','transfer_complete.json','exploratory.csv','declaration.json','checks.json','preparation.json','resource_preflight.json','source_miniature_gate.json','miniature_gate.json']:
 assert digest(out/name)==digest(dest/name),name
assert digest(run.CONFIG)==digest(dest/'config.json') and digest(c['protocol_path'])==digest(dest/'protocol.md')
original=read_config(dest/'generated_report_hashes.json');run.verify_hashes(original)
proofs=[
 ['declaration_verification.json','preparation_verification.json','reproduction_verification.json','tests.xml'],
 ['miniature_gate.json','preexecution_verification.json','model_diagnostics.csv','preprocessing.json','independent_development_audit.json'],
 ['independent_development_audit.json','tests.xml','clipping.csv'],
 ['crossfit.csv','summary.csv','selection.json','independent_development_audit.json'],
 ['contrasts.csv','matched-effects.md','seed_diversity.csv','decision_overlap.csv','model_diagnostics.csv','reference_diagnostics.json'],
 ['freeze_verification.json','transfer_verification.json','selection.json','exploratory_access.json','audit.json','findings.md'],
 ['audit.json','figure_verification.json','reproduction_verification.json','execution-history.md','test_fit_counts.json','prearchive-openspec-validation.log','compile-verification.json']]
families=[
 ['test_declared_counts_and_preserved_contract','test_absent_reference_delegates_exact_individual_recipe','test_partial_reference_never_falls_back'],
 ['test_sampling_factory_changes_only_effective_declared_parameters','test_new_bagged_cache_format','test_bagged_failure_evidence_blocks_retry'],
 ['test_fixed_aggregate_arithmetic_and_label_row_invariance','test_invalid_aggregates_rejected'],
 ['test_complete_analytical_grid_and_independent_audit'],
 ['test_independent_audit_checks_exact_table_keys','test_independent_audit_rejects_numeric_corruption','test_constant_correlations_and_complete_overlap'],
 ['test_every_winner_policy','test_conditional_draws_keep_all_roles_and_aliases'],
 ['test_bagged_failure_evidence_blocks_retry','test_independent_composition_provenance_corruption','test_independent_json_evidence_corruption']]
root=ET.parse(dest/'tests.xml').getroot();tests=[t for t in root.iter('testcase')];assert len(tests)==1070 and not any(list(root.iter(n)) for n in ['failure','error','skipped'])
passed={t.get('name').split('[')[0] for t in tests};spec=(change/'specs/baf-bagged-lightgbm-seed-study/spec.md').read_text();requirements=re.findall(r'^### Requirement: (.+)$',spec,re.M);scenarios=re.findall(r'^#### Scenario: (.+)$',spec,re.M)
assert len(requirements)==7 and len(scenarios)==10
records=[]
for label,files,names in zip(requirements,proofs,families):
 assert set(names)<=passed
 records.append(dict(requirement=label,status='verified',evidence={str(dest/p):digest(dest/p) for p in files},test_families=names))
scenario_requirements=[0,1,1,2,3,4,5,5,6,6]
scenarios=[dict(scenario=label,status='verified',requirement=requirements[i],test_families=families[i]) for label,i in zip(scenarios,scenario_requirements)]
tasks=(change/'tasks.md').read_text();assert tasks.count('- [x]')==13 and tasks.count('- [ ]')==1 and '- [ ] 5.4' in tasks
write_json(dest/'completion_requirements.json',dict(requirements=records,scenarios=scenarios,implementation_tasks_verified=13,final_sync_archive_and_manifest_pending=True,production_fingerprint=s['fingerprint'],mandatory_preservation_paths=634,ancestor_selection_pairs=15,historical_ignored_files=3522,new_development_fits=18,new_final_fits=0,source_test_fits=6,synthetic_integration_fits=9,original_generated_report_files=len(original),goal_achieved=False))
write_json(dest/'frozen_run_artifacts.json',{str(p):digest(p) for p in sorted(out.rglob('*')) if p.is_file()})
print('Audited7 requirements,10 scenarios,13 verified tasks;final sync/archive/manifest remains. Broader goal unachieved.')
