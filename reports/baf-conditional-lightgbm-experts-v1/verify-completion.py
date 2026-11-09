"""Requirement-by-requirement closure; a negative experiment is not goal success."""
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baf import conditional_expert_study as run
from baf.common import read_config,write_json,digest

c,s=run.load_selection();out=Path(c['output_dir']);dest=Path(c['report_dir'])
change=run.planning_evidence_path().parent
spec=next((change/'specs').glob('*/spec.md')).read_text()
run.guard(c);run.execution_gate(c,s['fingerprint'])
science=read_config(dest/'scientific-results-verification.json')
figure=read_config(dest/'figure-verification.json')
copy=read_config(dest/'clean-replay-verification.json')
documentation=read_config(dest/'documentation-verification.json')
compile=read_config(dest/'compile-verification.json')
assert science['status']=='passed' and not science['broader_goal_achieved']
assert figure['visually_inspected'] and figure['pixel_identical_regeneration']
assert copy['tests_passed']==1915 and copy['current_tests']==414
assert copy['original_selection_restores']==copy['changed_restore_rejections']==17
assert copy['no_prior_models_or_predictions'] and not copy['full_size_retraining_claimed']
assert documentation['status']=='passed' and compile['exit_code']==0
run.verify_hashes(copy['files']);run.verify_hashes(compile['files'])
run.verify_hashes(read_config(dest/'generated_report_hashes.json'))
run.verify_hashes(read_config(dest/'run_hashes.json'))
ledger=read_config(dest/'test_model_fits.json')
run.verify_hashes(ledger['execution_source_sha256'])
assert ledger['exit_code']==0 and ledger['junit_sha256']==digest(dest/'tests.xml')
cases=list(ET.parse(dest/'tests.xml').getroot().iter('testcase'))
assert len(cases)==1915 and sum(t.get('classname','').endswith('test_conditional_experts') for t in cases)==414
assert not any(list(t) for t in cases)
names={t.get('name','').split('[')[0] for t in cases if t.get('classname','').endswith('test_conditional_experts')}
required_checks=['test_four_new_formats_fit_audit_reuse_and_reject_corruption',
    'test_six_reference_formats_through_new_destination','test_failed_eligible_fit_retains_failure_and_never_falls_back',
    'test_exact_stitching_fallbacks_membership_and_label_invariance','test_eligibility_only_uses_earlier_training',
    'test_global_costs_are_independently_reconstructible','test_complete_analytical_tables_independently_reconstructed',
    'test_all_analytical_table_corruptions_fail','test_policy_uses_earlier_labels_and_selection_uses_actual_f2',
    'test_forward_rule_or_chronology_corruption_fails','test_freeze_and_reload_every_winner_policy_contract',
    'test_selection_corruption_rejected','test_final_bounds_all_cases_with_wholly_absent_original_caches',
    'test_all_role_type_uncertainty_reconstructs_1500_draws','test_figure_values_endpoints_legends_and_regeneration',
    'test_source_gate_failure_retained_and_retry_rejected','test_historical_snapshot_accepts_whole_absence_and_checks_all_present_files']
assert set(required_checks)<=names
def proof(paths):
    result={}
    for name in paths:
        p=dest/name;assert p.is_file(),name;result[str(p)]=digest(p)
    return result
requirements=[
 ('Prospective isolated preserved study',['contract_verification.json','production-gates-verification.json','development-preparation-verification.json','clean-replay-verification.json','tests.xml'],
  'Frozen declaration precedes new fits; exact individual adapters and six genuine tiny old formats; 1053 mandatory/4369 historical paths and seventeen exact selection restores. No historical grid rerun.'),
 ('Earlier-only conditional populations and global fallback',['eligibility.csv','routing.csv','stratum_errors.csv','independent_development_audit.json','tests.xml'],
  'Raw earlier membership/minima, all 54 saved local populations and 54 subsets, all 96 routes/2304 stratum rows including empty/missing/unseen groups, exactly-once scores; failed eligible-fit and invariance fixtures.'),
 ('Shared train-only representation and whole-training costs',['miniature-parameter-metadata-correction.json','miniature-metadata-note.md','production-gates-verification.json','conditional_model_work.csv','conditional_inference_work.csv','work-accounting-note.md','independent_development_audit.json'],
  'Sixteen saved miniature audits and 60 actual models verify global preprocessing/weights/no duplicate weighting, parameters, effective rounds and saved inference. Original balanced getter-only defect preserved/disclosed; no fit or tolerance changed. Work is scoped as logical exposure versus measured fit costs.'),
 ('Canonical fixed score compositions and references',['physical_score_diagnostics.csv','independent_development_audit.json','selection.json','plotted_alpha_curves.csv','tests.xml'],
  'All 93 full scores and 54 separate subsets audited; exact raw arithmetic, 24 strategies, six fixed reference identities, canonical alpha endpoints and visible selected/reference aliases.'),
 ('Comparable chronological actual-F2 selection',['crossfit.csv','summary.csv','independent_development_audit.json','freeze-verification.json','selection.json'],
  'All 144 earlier-only rules/masks/metrics reconstructed on identical applications; actual month5/6 macro F2 winner and fixed original references verified before freeze. No review quota or precision floor.'),
 ('Complete comparable effects and group diagnostics',['contrasts.csv','matched_effect_summary.csv','eligibility.csv','routing.csv','stratum_errors.csv','conditional_model_work.csv','conditional_inference_work.csv','independent_development_audit.json'],
  'Complete 738 unique factor identities/numerical differences and every grid value independently reconstructed. All 12 pure-vs-global200 F2 effects are negative. No causal population/equal-runtime claim.'),
 ('Frozen bounded exploratory transfer and honest acceptance',['freeze-verification.json','selection.json','exploratory_access.json','transfer_complete.json','audit.json','required_reference_comparisons.csv','scientific-results-verification.json','findings.md'],
  'Actual reference winner freezes eight visible roles/eight bases/thirteen scores before access; all required saved final inference/rules/masks/metrics verified. Zero new final fits, eight base/five derived reuses. Unchanged anchor ties itself and fails fusion/unweighted transfer comparisons; objective remains active. Generic conditional bounds are tested across all 72 prospective cases, not swept on month7.'),
 ('Independently verified reproducible completion',['audit.json','scientific-results-verification.json','figure-verification.json','clean-replay-verification.json','documentation-verification.json','compile-verification.json','execution-history.md','test_model_fits.json','tests.xml'],
  'All 1500 paired fixed-decision draws/28 interval rows reconstructed; figure values/regeneration plus actual visual inspection; exact data/pins/restores and no-prior-cache full suite; closed logs and run/report hashes. Retraining/adaptivity/temporal/generative/multiplicity exclusions disclosed.')]
assert re.findall(r'^### Requirement: (.+)$',spec,re.M)==[n for n,p,t in requirements]
req=[dict(requirement=n,status='verified',evidence=proof(p),scope=t) for n,p,t in requirements]
scenario_map={
 'Encounter a partial required source':(['tests.xml'],'Actual tiny model/cache corruption and partial-set rejection; present historical hashes unchanged.'),
 'Reproduce a wholly absent required source':(['tests.xml','clean-replay-verification.json'],'Six exact tiny old source formats genuinely fitted and audited on reuse; whole-absence copy has no production cache; no unsupported full-size reproduction claim.'),
 'Predict an unseen future membership':(['tests.xml','production-gates-verification.json','routing.csv'],'Synthetic original-minimum gate independently reconstructs unseen/missing fallback; complete actual route grid retains empty buckets.'),
 'Fail an eligible expert':(['tests.xml'],'Failure retained and no automatic retry/fallback in explicit numerical-failure fixture; actual 60 production fits passed.'),
 'Train a group with different class prevalence':(['production-gates-verification.json','conditional_model_work.csv','independent_development_audit.json'],'Differentiated miniature prevalences and all 54 actual local models retain whole-training class costs and preprocessing.'),
 'Display a unit conditional weight':(['figure-verification.json','plotted_alpha_curves.csv'],'All twelve curves alias alpha0 to original global200 and alpha1 to declared pure expert; no extra fit/strategy.'),
 'Decide for month six':(['crossfit.csv','independent_development_audit.json'],'Actual model training0..5 and rule learning4/5 or latest5, excluding current labels, independently rebuilt.'),
 'Observe a conditional improvement':(['contrasts.csv','matched_effect_summary.csv'],'Every pure/blend/global200/global400/weighting/partition factor retained, including negative effects; no observed pure-vs200 gain assumed.'),
 'Transfer a conditional winner':(['tests.xml','freeze-verification.json'],'All 72 exact role/DAG/eligibility/type/availability freeze contracts tested, including 17-role conditional cases; actual winner is a reference with eight roles, so no losing conditional grid transfers.'),
 'Fail a required reference comparison':(['audit.json','required_reference_comparisons.csv','findings.md'],'Actual anchor tie and negative transfer differences versus tree fusion/unweighted retained; no broader goal achievement.'),
 'Fail a declared gate':(['tests.xml','miniature-metadata-note.md','production-gates-verification.json'],'Retained gate-failure/no-retry fixtures and original failed metadata assertion; producer/gate artifacts unchanged, effective fitted balanced state and exact equivalence independently established.'),
 'Finish a negative finite experiment':(['audit.json','findings.md','scientific-results-verification.json'],'All declared output grids checked; performance criterion remains false and goal active. Finite completion does not certify improvement.')}
scenarios=re.findall(r'^#### Scenario: (.+)$',spec,re.M)
assert set(scenarios)==set(scenario_map)
sc=[dict(scenario=n,status='verified',evidence=proof(scenario_map[n][0]),scope=scenario_map[n][1]) for n in scenarios]
task_map={
 '1.1':['contract_verification.json'], '1.2':['tests.xml','clean-replay-verification.json'],
 '2.1':['production-gates-verification.json','independent_development_audit.json','tests.xml'],
 '2.2':['independent_development_audit.json','tests.xml'],
 '2.3':['test_model_fits.json','tests.xml','clean-replay-verification.json'],
 '3.1':['production-gates-verification.json','resource_preflight.json','miniature_gate.json'],
 '3.2':['development-preparation-verification.json','preparation.json'],
 '3.3':['independent_development_audit.json'], '4.1':['freeze-verification.json','selection.json','tests.xml'],
 '4.2':['audit.json','transfer_complete.json','scientific-results-verification.json'],
 '5.1':['audit.json','development_draws.csv','consumed_draws.csv','development_intervals.csv','consumed_intervals.csv'],
 '5.2':['findings.md','matched_effect_summary.csv','figure-verification.json','work-accounting-note.md'],
 '5.3':['clean-replay-verification.json','documentation-verification.json','execution-history.md'],
 '5.4':['compile-verification.json','prearchive-openspec-validation.log','scientific-results-verification.json']}
tasks=(change/'tasks.md').read_text()
assert re.findall(r'^- \[ \] (\d+\.\d+) ',tasks,re.M)==['5.4']
ids=re.findall(r'^- \[[ x]\] (\d+\.\d+) ',tasks,re.M);assert set(ids)==set(task_map)
assert '0 failed' in (dest/'prearchive-openspec-validation.log').read_text()
result=dict(status='passed',requirements=req,scenarios=sc,
    tasks=[dict(task=n,status='verified',evidence=proof(task_map[n])) for n in ids],
    requirements_verified=len(req),scenarios_verified=len(sc),tasks_verified=len(ids),
    current_test_names=sorted(names),required_named_test_coverage=required_checks,
    fingerprint=s['fingerprint'],selection_sha256=digest(out/'selection.json'),broader_goal_achieved=False,
    created_at_utc=run.now())
write_json(dest/'completion-requirements.json',result)
p=change/'tasks.md';p.write_text(tasks.replace('- [ ] 5.4 ','- [x] 5.4 '))
print('Verified eight requirements, twelve scenarios and all fourteen tasks; broader performance goal remains active.')
