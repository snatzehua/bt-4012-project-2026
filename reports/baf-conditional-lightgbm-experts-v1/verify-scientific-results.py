"""Verify closed scientific exports, bounded work and all reference conclusions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from baf import conditional_expert_study as run
from baf import conditional_expert_report as report
from baf.bagged_anchor_report import audit_intervals
from baf.common import read_config,write_json,digest

c,s=run.load_selection();out=Path(c['output_dir']);dest=Path(c['report_dir'])
a=read_config(dest/'audit.json');prep=read_config(out/'preparation.json');transfer=read_config(out/'transfer_complete.json')
run.verify_hashes(read_config(dest/'generated_report_hashes.json'))
assert a['fingerprint']==s['fingerprint'] and a['selection_sha256']==digest(out/'selection.json')
assert a['saved_development_models_checked']==84 and a['saved_final_models_checked']==8
assert a['visible_roles']==8 and a['development_intervals']==21 and a['consumed_intervals']==7
assert a['direct_draws_checked']==1500 and a['figure_regeneration_identical']
assert not a['measured_gain_on_both_scopes'] and not a['conditional_clear_development_gain'] and not a['conditional_clear_consumed_gain']
assert s['selected_candidate']=='bagged_anchor_balanced_logit_a25' and s['selected_policy']=='pooled'
assert transfer['base_results']==transfer['reference_reuses']==8
assert transfer['local_fits']==transfer['capacity_fits']==transfer['invocation_new_conditional_fits']==0
assert transfer['invocation_exact_reference_fits']==0 and transfer['reference_derived_reuses']==5
assert transfer['model_artifacts']==8 and transfer['model_files']==9 and transfer['prediction_artifacts']==13
assert transfer['total_scores']==13 and transfer['subset_scores']==0 and transfer['role_count']==8
for t in ['bagged','spline','tree']:assert transfer['reproduced_'+t+'_fits']==0
assert prep['invocation_new_conditional_fits']==60 and prep['invocation_exact_reference_fits']==0
refs=run.scores.REFERENCE_ROLES
summary=pd.read_csv(out/'summary.csv');final=pd.read_csv(out/'exploratory.csv')
index=summary.set_index(['candidate','policy']);fi=final.set_index('role')
assert np.isclose(float(fi.loc['selected','f2']),5*int(fi.loc['selected','tp'])/(5*int(fi.loc['selected','tp'])+4*int(fi.loc['selected','fn'])+int(fi.loc['selected','fp'])),atol=1e-12,rtol=0)
comparisons=pd.read_csv(dest/'required_reference_comparisons.csv').set_index('reference')
assert set(comparisons.index)==set(refs)
for name,nk in refs.items():
    dev=float(s['selected_development_f2']-index.loc[tuple(nk),'f2'])
    consumed=float(fi.loc['selected','f2']-fi.loc[name,'f2'])
    assert np.isclose(dev,a['development_gain'][name],atol=1e-12,rtol=0)
    assert np.isclose(consumed,a['consumed_gain'][name],atol=1e-12,rtol=0)
    assert np.isclose(comparisons.loc[name,'development_difference'],dev,atol=1e-12,rtol=0)
    assert np.isclose(comparisons.loc[name,'consumed_difference'],consumed,atol=1e-12,rtol=0)
    assert bool(comparisons.loc[name,'development_pass'])==(dev>0)
    assert bool(comparisons.loc[name,'consumed_pass'])==(consumed>0)
assert a['development_gain']['bagged_anchor']==a['consumed_gain']['bagged_anchor']==0
assert a['consumed_gain']['fusion']<0 and a['consumed_gain']['unweighted']<0
dint=pd.read_csv(dest/'development_intervals.csv',dtype={'scope':str})
ddraws=pd.read_csv(dest/'development_draws.csv',dtype={'scope':str})
fint=pd.read_csv(dest/'consumed_intervals.csv',dtype={'scope':str})
fdraws=pd.read_csv(dest/'consumed_draws.csv')
assert len(ddraws)==1000 and len(fdraws)==500
assert set(zip(ddraws.scope,ddraws.replicate))=={(m,i) for m in ['5','6'] for i in range(500)}
assert set(fdraws.replicate)==set(range(500))
audit_intervals(dint,ddraws,fint,fdraws,s['choices'],500)
figure=report.audit_figure(summary,dest)
assert figure=={'operating_points':72,'curve_rows':60,'capacity_lines':6}
effects=pd.read_csv(out/'contrasts.csv');pure=effects[effects.contrast=='expert_vs200']
assert len(pure)==12 and (pure.f2_difference<0).all()
new=summary[summary.candidate.str.startswith(('expert_','capacity400_'))].iloc[0]
assert new.candidate=='expert_blend_housing_status_balanced_a25' and new.policy=='latest'
assert float(new.f2)<float(s['selected_development_f2'])
text=(dest/'findings.md').read_text()
for phrase in ['no review quota or precision floor','all six references','All 1500','adaptive','already consumed','no independent confirmation','retraining','multiplicity','Codex']:
    assert phrase in text,phrase
correction=read_config(dest/'miniature-parameter-metadata-correction.json')
assert correction['original_artifacts_preserved'] and correction['new_model_fits']==0
assert correction['exact_original_explicit_prediction_equivalence'] and not correction['numerical_recipe_relaxed']
assert (dest/'work-accounting-note.md').is_file()
run.guard(c)
files={str(p):digest(p) for p in out.rglob('*') if p.is_file()}
write_json(dest/'run_hashes.json',files)
result=dict(status='passed',fingerprint=s['fingerprint'],selection_sha256=digest(out/'selection.json'),
    development_f2=s['selected_development_f2'],consumed_f2=float(fi.loc['selected','f2']),
    best_new_candidate=str(new.candidate),best_new_policy=str(new.policy),best_new_f2=float(new.f2),
    new_development_fits=60,new_final_fits=0,final_reference_reuses=8,visible_roles=8,
    development_models=84,final_models=8,direct_draws_checked=1500,
    interval_rows=28,figure=figure,all_pure_vs200_effects_negative=True,
    getter_only_miniature_defect_disclosed=True,logical_inference_work_scope_disclosed=True,
    preservation_checked=True,run_file_count=len(files),broader_goal_achieved=False,created_at_utc=run.now())
write_json(dest/'scientific-results-verification.json',result)
print('Verified negative finite experiment: 84 development/eight final models; 1500 direct draws/28 intervals; 60 new development/zero final fits; all six comparisons and preservation.')
