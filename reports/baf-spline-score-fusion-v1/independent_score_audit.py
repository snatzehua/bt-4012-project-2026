"""Additional read-only score/parent and contrast-factor audit; run from repo root."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from baf.common import digest,features,load_data,read_config,write_json
from baf import spline_fusion_study as study

c,s=study.load_selection();out=Path(c['output_dir']);frame=load_data(c);base=features(frame)
lookup={x['id']:x for x in c['strategies']};members={x['id']:x for x in c['candidates']}
rows=[]
for phase,months in [('validation',[4,5,6]),('exploratory',[7])]:
 for month in months:
  names=set(lookup) if phase=='validation' else {n for n,k in s['choices'].values()}
  names.update(['baseline_balanced','baseline_lgbm','xgb_d4_w1',study.FUSION])
  def descend(n):
   if n==study.FUSION: parents=c['fusion_spec']['members']
   elif n in lookup and lookup[n]['kind'] in ['logit','margin_blend']:parents=lookup[n]['members']
   else:return
   for parent in parents:
    if parent not in names:names.add(parent);descend(parent)
  for n in list(names):descend(n)
  actual={p.stem.removeprefix(f'{phase}-m{month}-') for p in (out/'predictions').glob(f'{phase}-m{month}-*.npz')}
  assert actual==names,(phase,month,actual^names)
  data={n:study.prediction(c,phase,month,n) for n in names}
  def logit(p):
   assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
   q=np.clip(p,1e-6,1-1e-6);return np.log(q/(1-q))
  expected_ids=frame.index[frame.month==month].to_numpy()
  labels=frame.loc[expected_ids,'fraud_bool'].to_numpy()
  for n in sorted(names):
   d=data[n];r=read_config(study.paths(c,phase,month,n)[0])
   assert np.array_equal(d['row_ids'],expected_ids) and np.array_equal(d['labels'],labels)
   assert r['prediction_sha256']==digest(study.paths(c,phase,month,n)[1])
   parents=[]
   if n in members:
    a=joblib.load(study.newton.model_path(c,phase,month,n))
    assert a['spec']==members[n] and list(a['model'].classes_)==[0,1]
    matrix=a['inputs'].transform(base.loc[expected_ids]);m=a['model']
    expected=expit(np.asarray(matrix@m.coef_[0]).ravel()+m.intercept_[0])
   elif n==study.FUSION:
    parents=c['fusion_spec']['members'];expected=.5*logit(data[parents[0]]['scores'])+.5*logit(data[parents[1]]['scores'])
   elif n in lookup and lookup[n]['kind']=='logit':
    parents=lookup[n]['members'];expected=logit(data[parents[0]]['scores'])
   elif n in lookup and lookup[n]['kind']=='margin_blend':
    spec=lookup[n];parents=spec['members'];raw=parents[0].removeprefix('logit_');alpha=spec['alpha']
    expected=alpha*logit(data[raw]['scores'])+(1-alpha)*data[study.FUSION]['scores']
   else:
    # Saved-tree inference/recipe audits are separate frozen helper checks.
    expected=d['scores']
   np.testing.assert_allclose(d['scores'],expected,atol=1e-12,rtol=1e-12)
   if parents:
    assert set(r['dependencies'])==set(parents)
    for parent in parents:
     rp,pp=study.paths(c,phase,month,parent)
     assert r['dependencies'][parent]['record']==digest(rp)
     assert r['dependencies'][parent]['prediction']==digest(pp)
   rows.append(dict(phase=phase,month=month,candidate=n,parents=parents,checked=True))

contrasts=pd.read_csv(out/'contrasts.csv');actual=set(contrasts[['contrast','candidate','policy','reference','reference_policy']].itertuples(index=False,name=None))
expected=set();policies=['pooled','latest','rank_fraction'];raw=list(members)
for n in lookup:
 for k in policies:
  for ref in [study.FUSION,'baseline_balanced','baseline_lgbm']:expected.add(('old pooled reference',n,k,ref,'pooled'))
  if k!='pooled':expected.add(('policy minus own pooled',n,k,n,'pooled'))
for n in raw:
 for k in policies:expected.add(('logit minus raw','logit_'+n,k,n,k))
 for a in [.1,.25,.5]:
  b=f'margin_blend_{n}_a{int(a*100)}'
  for k in policies:
   for ref in ['logit_'+n,study.FUSION]:expected.add(('blend minus common-margin member',b,k,ref,k))
 for lo,hi in [(.1,.25),(.1,.5),(.25,.5)]:
  for k in policies:expected.add(('higher minus lower Newton weight',f'margin_blend_{n}_a{int(hi*100)}',k,f'margin_blend_{n}_a{int(lo*100)}',k))
for weight in ['unweighted','balanced']:
 for k in policies:
  for prefix in ['','logit_']:expected.add(('spline minus linear single',prefix+'spline_c0p1_'+weight,k,prefix+'linear_c0p1_'+weight,k))
  for a in [10,25,50]:expected.add(('spline minus linear blend',f'margin_blend_spline_c0p1_{weight}_a{a}',k,f'margin_blend_linear_c0p1_{weight}_a{a}',k))
for rep in ['linear','spline']:
 for k in policies:
  for prefix in ['','logit_']:expected.add(('balanced minus unweighted single',prefix+rep+'_c0p1_balanced',k,prefix+rep+'_c0p1_unweighted',k))
  for a in [10,25,50]:expected.add(('balanced minus unweighted blend',f'margin_blend_{rep}_c0p1_balanced_a{a}',k,f'margin_blend_{rep}_c0p1_unweighted_a{a}',k))
assert actual==expected and len(actual)==len(contrasts)==433
study.guard(c)
write_json(out/'independent_score_audit.json',dict(selection_sha256=digest(out/'selection.json'),auditor_sha256=digest(__file__),
 score_results=len(rows),newton_coefficient_sigmoid_checks=sum(r['candidate'] in members for r in rows),
 independent_contrast_factor_count=433,complete_score_parent_sets=True,results=rows))
print('Independent score/parent/factor audit passed:',len(rows),'scores, 433 contrasts')
