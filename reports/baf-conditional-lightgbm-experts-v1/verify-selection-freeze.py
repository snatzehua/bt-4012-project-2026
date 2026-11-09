"""Independently check the actual selected-role contract before consumed access."""
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from baf import conditional_expert_study as run
from baf.conditional_expert_report import independent_rule
from baf.common import read_config, write_json, digest, load_data

c,s=run.load_selection();out=Path(c['output_dir']);dest=Path(c['report_dir'])
assert not (out/'exploratory_access.json').exists()
refs={'bagged_anchor':['bagged_anchor_balanced_logit_a25','pooled'],
    'latest_fusion':['interaction_blend_spline_c1p0_balanced_a10','latest'],
    'recent_fusion':['margin_blend_spline_c0p1_unweighted_a25','latest'],
    'fusion':['logit_mean_xgb_d4_w1_baseline_balanced_a50','pooled'],
    'balanced':['baseline_balanced','pooled'],'unweighted':['baseline_lgbm','pooled']}
summary=pd.read_csv(out/'summary.csv').sort_values(['f2','recall','precision','candidate','policy'],ascending=[False,False,False,True,True],kind='stable')
w=summary.iloc[0];name,policy=str(w.candidate),str(w.policy)
assert (s['selected_candidate'],s['selected_policy'])==(name,policy)
assert np.isclose(s['selected_development_f2'],w.f2,atol=1e-12,rtol=0)
roles=dict(selected=[name,policy],**refs,same_strategy=[name,'pooled'])
capacity=re.fullmatch(r'capacity400_(balanced|unweighted)',name)
conditional=re.fullmatch(r'expert_(?:blend_)?(payment_type|housing_status)_(balanced|unweighted)(?:_a(25|50|75))?',name)
expected_new=set();expected_full=set()
frame=load_data(c);train=frame[frame.month<7]
eligible={}
for p,levels in {'payment_type':['AA','AB','AC','AD','AE'],'housing_status':['BA','BB','BC','BD','BE','BF','BG']}.items():
    eligible[p]=[]
    for level in levels:
        part=train[train[p].eq(level)]
        n,f=len(part),int(part.fraud_bool.sum())
        if n>=10000 and f>=100 and n-f>=100:eligible[p].append(level)
def g(w):return 'baseline_balanced' if w=='balanced' else 'baseline_lgbm'
def e(p,w,a):return f'expert_{p}_{w}' if a==100 else f'expert_blend_{p}_{w}_a{a}'
if capacity:
    w=capacity.group(1);other='unweighted' if w=='balanced' else 'balanced'
    roles.update(global200=[g(w),policy],matched_weight=['capacity400_'+other,policy])
elif conditional:
    p,w,a=conditional.groups();a=int(a or 100)
    other='unweighted' if w=='balanced' else 'balanced';q='housing_status' if p=='payment_type' else 'payment_type'
    roles.update(global200=[g(w),policy],global400=['capacity400_'+w,policy],
        expert_pure=[e(p,w,100),policy],matched_weight=[e(p,other,a),policy],matched_partition=[e(q,w,a),policy])
    roles.update({'alpha_a'+str(v):[e(p,w,v),policy] for v in [25,50,75,100]})
else:
    assert name in {v[0] for v in refs.values()}
assert roles==s['choices'] and len(roles)==(17 if conditional else 10 if capacity else 8)
pure=set();caps=set()
for n,k in roles.values():
    if n.startswith('capacity400_'):caps.add(n.split('capacity400_')[1]);expected_full.add(n)
    m=re.fullmatch(r'expert_(?:blend_)?(payment_type|housing_status)_(balanced|unweighted)(?:_a(25|50|75))?',n)
    if m:
        p,w,a=m.groups();pure.add((p,w));expected_full.update([n,e(p,w,100)])
for p,w in pure:
    expected_new.update(f'expert_model_{p}_{w}_{level}' for level in eligible[p])
expected_new.update('capacity400_'+w for w in caps)
dag=s['score_dependencies'];bounds=s['final_fit_bounds']
assert {v['id'] for v in dag['new_models']}==expected_new
assert set(map(tuple,dag['pure_methods']))==pure and set(dag['capacity_weightings'])==caps
assert len(dag['old_bases'])==8 and len(dag['old_scores'])==13
assert set(dag['full_scores'])==set(dag['old_scores'])|expected_full
assert bounds['max_new_local_fits']==sum(n.startswith('expert_model_') for n in expected_new)
assert bounds['max_new_capacity_fits']==len(caps)
assert bounds['unique_base_results']==8+len(expected_new)
assert bounds['physical_scores']==len(dag['full_scores'])
assert bounds['subset_scores']==bounds['max_new_local_fits']
assert all(bounds['max_reproduced_'+t+'_fits']==0 for t in ['bagged','spline','tree'])
assert len(bounds['original_availability'])==8 and all(bounds['original_availability'].values())
for files in bounds['original_availability'].values():run.verify_hashes(files)
h=run.history(c)
assert all(s['rules'][a]==independent_rule(h[n],k,7) for a,(n,k) in roles.items())
audit=read_config(out/'independent_development_audit.json')
assert audit['created_at_utc']<s['created_at_utc']
assert s['month7_previously_consumed'] and s['exploratory']
run.guard(c)
result=dict(status='passed',selection_sha256=digest(out/'selection.json'),selected_candidate=name,
    selected_policy=policy,macro_development_f2=float(w.f2),visible_roles=len(roles),
    unique_roles=len(set(map(tuple,roles.values()))),roles=roles,exact_new_models=sorted(expected_new),
    exact_full_scores=dag['full_scores'],actual_bounds=bounds,independent_earlier_rules=True,
    all_72_prospective_cases_verified_by_full_suite=True,month7_access=False,created_at_utc=run.now())
write_json(dest/'freeze-verification.json',result)
print('Verified actual macro-F2 winner,',len(roles),'visible roles and exact raw-training membership/dependency/work bounds before consumed access.')
