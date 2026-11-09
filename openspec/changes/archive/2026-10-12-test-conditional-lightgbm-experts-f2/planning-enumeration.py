"""Prospective declaration and old development diagnostics only; no model fitting."""
from pathlib import Path
from itertools import product,combinations
import sys,importlib.metadata
import numpy as np
import pandas as pd
from baf import bagged_anchor_study as parent
from baf.common import read_config,write_json,digest,load_data
from baf.improvements import prediction
from baf.recall_weighted_metrics import fit_policy,measure,learn,decisions
ROOT=Path(__file__).resolve().parent
c=read_config(parent.CONFIG);parent.guard(c)
REFS={'bagged_anchor':['bagged_anchor_balanced_logit_a25','pooled'],**parent.scores.REFERENCE_ROLES}
# The parent's mapping is role -> [canonical score, fixed policy].
assert len(REFS)==6
W=['balanced','unweighted'];P=['payment_type','housing_status'];K=['pooled','latest','rank_fraction'];A=[25,50,75,100]
LEVELS={'payment_type':['AA','AB','AC','AD','AE'],'housing_status':['BA','BB','BC','BD','BE','BF','BG']}
def global200(w):return 'baseline_balanced' if w=='balanced' else 'baseline_lgbm'
def capacity(w):return 'capacity400_'+w
def expert(p,w):return 'expert_'+p+'_'+w
def candidate(p,w,a):return expert(p,w) if a==100 else 'expert_blend_'+p+'_'+w+'_a'+str(a)
S=[dict(id=n,kind='reference') for n,k in REFS.values()]+[dict(id=capacity(w),kind='capacity',weighting=w) for w in W]
S += [dict(id=candidate(p,w,a),kind='expert' if a==100 else 'blend',partition=p,weighting=w,alpha_numerator=a,alpha_denominator=100) for p,w,a in product(P,W,A)]
assert len(S)==len({s['id'] for s in S})==24
LOOK={s['id']:s for s in S};OLD=parent.scores.dependencies([n for n,k in REFS.values()]);assert (len(OLD['bases']),len(OLD['scores']))==(8,13)
def roles(n,k):
 r={'selected':[n,k],**REFS,'same_strategy':[n,'pooled']};s=LOOK[n]
 if s['kind']=='reference':return r
 w=s['weighting'];other=W[1-W.index(w)];r['global200']=[global200(w),k]
 if s['kind']=='capacity':r['matched_weight']=[capacity(other),k];return r
 p=s['partition'];a=s['alpha_numerator'];q=P[1-P.index(p)]
 r.update(global400=[capacity(w),k],expert_pure=[expert(p,w),k],matched_weight=[candidate(p,other,a),k],matched_partition=[candidate(q,w,a),k])
 r.update({'alpha_a'+str(v):[candidate(p,w,v),k] for v in A});return r
frame=load_data(c);frame=frame[frame.month<7]
groups=[]
for m in [4,5,6,7]:
 t=frame[frame.month<m]
 for p in P:
  for level in LEVELS[p]:
   labels=t.loc[t[p]==level,'fraud_bool'];n=len(labels);fraud=int(labels.sum());eligible=n>=10000 and fraud>=100 and n-fraud>=100
   groups.append(dict(eval_month=m,partition=p,level=level,train_rows=n,train_fraud=fraud,train_legitimate=n-fraud,eligible=eligible))
 for p,w in product(P,W):
  assert sum(g['eligible'] for g in groups if g['eval_month']==m and g['partition']==p)==(4 if p=='payment_type' else 5)
def dag(rolemap):
 names={n for n,k in rolemap.values()};new=set();capacities=set();local=[];pure=set()
 for n in names:
  s=LOOK[n]
  if s['kind']=='reference':continue
  if s['kind']=='capacity':capacities.add(n);new.add(n);continue
  p,w=s['partition'],s['weighting'];pure.add((p,w));new.add(expert(p,w));new.add(n)
 for p,w in sorted(pure):
  local += ['expert_model_'+p+'_'+w+'_'+g['level'] for g in groups if g['eval_month']==7 and g['partition']==p and g['eligible']]
 return dict(old_bases=OLD['bases'],old_scores=OLD['scores'],capacity_bases=sorted(capacities),expert_models=sorted(local),new_full_scores=sorted(new),full_scores=sorted(set(OLD['scores'])|new))
cases=[]
for s,k in product(S,K):
 r=roles(s['id'],k);g=dag(r);a=len(g['capacity_bases']);b=len(g['expert_models']);kind=s['kind']
 assert len(r)==(8 if kind=='reference' else 10 if kind=='capacity' else 17)
 assert len(g['full_scores'])<=(13 if kind=='reference' else 15 if kind=='capacity' else 22)
 assert a+b<=(0 if kind=='reference' else 2 if kind=='capacity' else 15)
 cases.append(dict(winner=s['id'],policy=k,kind=kind,roles=r,dependencies=g,current_new_capacity_fit_bound=a,current_new_expert_fit_bound=b,fresh_reference_fit_bounds={'bagged':3,'spline':2,'tree':3}))
# Complete unique contrast identities. No new model predictions enter these.
factors=[]
def add(t,n,k,b,bk,**extra):factors.append((t,n,k,b,bk,tuple(sorted(extra.items()))))
for s,k in product(S,K):
 n=s['id']
 for role,(b,bk) in REFS.items():add('fixed_reference',n,k,b,bk,reference=role)
 if k!='pooled':add('own_policy',n,k,n,'pooled')
for w,k in product(W,K):add('capacity400_vs200',capacity(w),k,global200(w),k)
for p,w,k in product(P,W,K):
 for b,t in [(global200(w),'expert_vs200'),(capacity(w),'expert_vs400')]:add(t,expert(p,w),k,b,k)
 for a in A[:-1]:
  n=candidate(p,w,a)
  for b,t in [(global200(w),'blend_vs200'),(capacity(w),'blend_vs400'),(expert(p,w),'blend_vspure')]:add(t,n,k,b,k)
 for lo,hi in combinations(A,2):add('higher_alpha',candidate(p,w,hi),k,candidate(p,w,lo),k,lower=lo,higher=hi)
for p,a,k in product(P,A,K):add('matched_weighting',candidate(p,'balanced',a),k,candidate(p,'unweighted',a),k)
for w,a,k in product(W,A,K):add('matched_partition',candidate('housing_status',w,a),k,candidate('payment_type',w,a),k)
assert len(factors)==len(set(factors))==738
preserve=dict(c['preserve_sha256'])
for p in parent.SOURCES[-3:]+[parent.CONFIG,c['protocol_path'],str(Path(c['output_dir'])/'selection.json'),str(Path(c['report_dir'])/'report_hashes.json')]:preserve[p]=digest(p)
preserve.update(read_config(Path(c['report_dir'])/'report_hashes.json'));parent.verify_hashes(preserve);assert len(preserve)==1053
pairs=[p for p in preserve if p.startswith('outputs/') and p.endswith('/selection.json')];assert len(pairs)==17
assert all(preserve[p]==preserve[p.replace('outputs/','reports/',1)] for p in pairs)
roots=sorted(set(c['historical_output_roots']+[c['output_dir']]))
snapshot={str(p):digest(p) for root in roots for p in Path(root).rglob('*') if p.is_file()}
assert len(snapshot)==4369
sources=parent.SOURCES+['baf/conditional_experts.py','baf/conditional_expert_study.py','baf/conditional_expert_report.py'];assert len(sources)==58==len(set(sources))
# Recompute OLD label-aware diagnostics only after the old freeze, excluding them
# from new selection. Never open an exploratory/month7 prediction here.
optima=[];errors=[];n=REFS['bagged_anchor'][0];history={m:prediction(c,'validation',m,n) for m in [4,5,6]}
for name,k in REFS.values():
 for m in [4,5,6]:
  d=prediction(c,'validation',m,name);threshold=fit_policy(d,'latest');q=fit_policy(d,'rank_fraction');v=measure(d,threshold)
  optima.append(dict(candidate=name,month=m,threshold=threshold['threshold'],diagnostic_fraction=q['numerator']/q['denominator'],diagnostic_f2=v['f2'],diagnostic_recall=v['recall']))
for m in [5,6]:
 d=history[m];mask=decisions(d,learn({t:v for t,v in history.items() if t<m},'pooled',m));y=d['labels'].astype(bool)
 for p in P:
  values=frame.loc[d['row_ids'],p].astype(object).to_numpy()
  for level in LEVELS[p]:
   take=values==level;errors.append(dict(month=m,partition=p,level=level,n=int(take.sum()),fraud=int((take&y).sum()),tp=int((take&y&mask).sum()),fp=int((take&~y&mask).sum()),fn=int((take&y&~mask).sum())))
pins={}
for file in ['requirements.txt','requirements-improvements.txt','requirements-xgboost.txt']:
 for line in Path(file).read_text().splitlines():
  if '==' in line:
   pkg,version=line.split('==');assert importlib.metadata.version(pkg)==version;pins[pkg]=version
assert len(pins)==27 and sys.version.split()[0]=='3.12.9' and digest(c['data_path'])==c['data_sha256']
counts=dict(selectable_strategies=24,strategy_policy_cases=72,physical_full_scores=93,selectable_score_months=72,reference_base_results=24,reference_derived_results=15,new_capacity_fits=6,new_expert_fits=54,new_full_fits=60,eligibility_rows=72,fallback_model_records=18,member_subset_scores=54,stitched_scores=12,new_blend_scores=36,forward_rows=144,macro_rows=72,pooled_rows=72,contrasts=738,routing_rows=96,stratum_error_rows=2304,curve_rows=60,operating_points=72,production_miniature_fits=16,current_tiny_fits_per_full_invocation=10)
write_json(ROOT/'planning-evidence.json',dict(parent_config=parent.CONFIG,parent_fingerprint=parent.fingerprint(),parent_selection_sha256=digest(Path(c['output_dir'])/'selection.json'),parent_report_manifest_sha256=digest(Path(c['report_dir'])/'report_hashes.json'),planned_source_paths=sources,mandatory_preservation_sha256=preserve,historical_output_roots=roots,historical_ignored_file_sha256=snapshot,reference_policies=REFS,strategies=S,partitions=LEVELS,eligibility=groups,winner_policy_cases=cases,contrasts=factors,expected_counts=counts,existing_development_diagnostics=optima,existing_forward_group_errors=errors,diagnostic_scope='Old frozen rankings,months4/5/6 only,label-aware hindsight diagnostic excluded from new selection; no new model/composition evaluated and no month7 outcome inspected.',verified_pins=pins,python=sys.version.split()[0],data_sha256=c['data_sha256'],new_model_fits=0,new_policy_or_composition_evaluations=0))
print('Prospective declaration:',counts)
print('Preservation:',len(preserve),'mandatory;',len(snapshot),'historical;',len(pairs),'selection pairs;',len(sources),'planned sources')
print('Case maxima:',max(len(v['roles']) for v in cases),'roles;',max(len(v['dependencies']['full_scores']) for v in cases),'full scores;',max(v['current_new_capacity_fit_bound']+v['current_new_expert_fit_bound'] for v in cases),'new fits')
