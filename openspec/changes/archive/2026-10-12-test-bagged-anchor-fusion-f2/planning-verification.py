"""Separate declarative reconstruction of prospective IDs, DAGs and role schemas."""
from pathlib import Path
from itertools import product
from collections import Counter
import re
from baf.common import read_config,digest,write_json

root=Path(__file__).parent;e=read_config(root/'planning-evidence.json')
H='interaction_blend_spline_c1p0_balanced_a10';T='logit_mean_xgb_d4_w1_baseline_balanced_a50';R='margin_blend_spline_c0p1_unweighted_a25'
refs={'latest_fusion':[H,'latest'],'recent_fusion':[R,'latest'],'fusion':[T,'pooled'],'balanced':['baseline_balanced','pooled'],'unweighted':['baseline_lgbm','pooled']}
weights=['balanced','unweighted'];policies=['pooled','latest','rank_fraction'];baseline={'balanced':'baseline_balanced','unweighted':'baseline_lgbm'}
seeds=[f'bagged_{w}_s{s}' for w,s in product(weights,[17,29,43])]
means=[f'bagged_{w}_{f}_mean' for w,f in product(weights,['prob','logit'])]
logits=['logit_'+p for p in seeds+[f'bagged_{w}_prob_mean' for w in weights]+list(baseline.values())]
blends=[f'bagged_anchor_{w}_{f}_a{a}' for w,f,a in product(weights,['prob','logit'],['10','25','50'])]
expected=set([v[0] for v in refs.values()]+seeds+means+logits+blends)
assert len(expected)==37 and set(e['strategies'])==expected and len(e['winner_policy_cases'])==111
old={H:['spline_c1p0_balanced',T],R:['spline_c0p1_unweighted',T],T:['xgb_d4_w1','baseline_balanced']}
oldbase={'spline_c1p0_balanced','spline_c0p1_unweighted','xgb_d4_w1','baseline_balanced','baseline_lgbm'}
def graph(names):
 scores=set();bases=set()
 def visit(n):
  if n in scores:return
  scores.add(n)
  if n in oldbase or n in seeds:bases.add(n);return
  if n in old:members=old[n]
  elif n.startswith('logit_'):members=[n[6:]]
  elif n in means:
   w='balanced' if n.startswith('bagged_balanced_') else 'unweighted';members=[f'bagged_{w}_s{s}' for s in [17,29,43]]
  else:
   m=re.fullmatch(r'bagged_anchor_(balanced|unweighted)_(prob|logit)_a(10|25|50)',n);assert m
   w,f,_=m.groups();members=[H,('logit_bagged_'+w+'_prob_mean') if f=='prob' else ('bagged_'+w+'_logit_mean')]
  for p in members:visit(p)
 for n in names:visit(n)
 return dict(bases=sorted(bases),scores=sorted(scores),derived=sorted(scores-bases),bagged_bases=sorted(bases&set(seeds)))
common={'selected','same_strategy',*refs}
seen=set();sizes=Counter()
for case in e['winner_policy_cases']:
 n,k=case['winner'],case['policy'];assert n in expected and k in policies and (n,k) not in seen;seen.add((n,k));roles=case['roles']
 assert roles['selected']==[n,k] and roles['same_strategy']==[n,'pooled'] and all(roles[a]==v for a,v in refs.items())
 if n in {v[0] for v in refs.values()}:kind='reference';keys=common
 elif n in ['logit_baseline_balanced','logit_baseline_lgbm']:
  kind='baseline_logit';keys=common|{'native_member','matched_weight'}
  assert roles['native_member']==[n[6:],k]
  assert roles['matched_weight']==['logit_baseline_lgbm' if n.endswith('balanced') else 'logit_baseline_balanced',k]
 else:
  isblend=n in blends;kind='blend' if isblend else ('single' if n.removeprefix('logit_') in seeds else 'aggregate')
  w='balanced' if re.search(r'_(balanced)_',n) else 'unweighted';other='unweighted' if w=='balanced' else 'balanced'
  keys=common|{f'member_s{s}' for s in [17,29,43]}|{f'member_logit_s{s}' for s in [17,29,43]}|{'probability_mean','probability_mean_logit','logit_mean','unbagged_member','unbagged_logit'}
  for s in [17,29,43]:
   assert roles[f'member_s{s}']==[f'bagged_{w}_s{s}',k] and roles[f'member_logit_s{s}']==[f'logit_bagged_{w}_s{s}',k]
  for a,v in {'probability_mean':f'bagged_{w}_prob_mean','probability_mean_logit':f'logit_bagged_{w}_prob_mean','logit_mean':f'bagged_{w}_logit_mean','unbagged_member':baseline[w],'unbagged_logit':'logit_'+baseline[w]}.items():assert roles[a]==[v,k]
  assert roles['matched_weight']==[n.replace('_'+w+'_','_'+other+'_'),k]
  if isblend:
   keys|={'anchor_at_policy','added_member_native','added_member_margin','matched_weight','matched_form','alpha_a10','alpha_a25','alpha_a50'}
   f,a=re.fullmatch(r'bagged_anchor_(?:balanced|unweighted)_(prob|logit)_a(10|25|50)',n).groups()
   native=f'bagged_{w}_{f}_mean';margin='logit_'+native if f=='prob' else native
   assert roles['anchor_at_policy']==[H,k] and roles['added_member_native']==[native,k] and roles['added_member_margin']==[margin,k]
   assert roles['matched_form']==[f'bagged_anchor_{w}_{"logit" if f=="prob" else "prob"}_a{a}',k]
   for q in ['10','25','50']:assert roles['alpha_a'+q]==[f'bagged_anchor_{w}_{f}_a{q}',k]
  else:
   keys|={'native_member','transformed_member','matched_weight'};native=n.removeprefix('logit_');margin='logit_'+native if native in seeds or native.endswith('prob_mean') else native
   assert roles['native_member']==[native,k] and roles['transformed_member']==[margin,k]
 assert case['kind']==kind and set(roles)==keys and all(v[0] in expected for v in roles.values())
 assert case['dependencies']==graph([v[0] for v in roles.values()]);assert case['current_new_bagged_fit_bound']==len(case['dependencies']['bagged_bases'])
 assert case['current_reference_reproduction_bound']==0 and case['fresh_additive_reproduction_bound']==2 and case['fresh_old_tree_reproduction_bound']==3
 sizes[kind]+=1
assert seen==set(product(expected,policies))
assert graph(expected)==e['complete_dependencies']
# Independently reconstruct every factor identity from the finite names.
facts=set()
def factor(label,n,k,p,pk=None):facts.add((label,n,k,p,k if pk is None else pk))
for n,k in product(expected,policies):
 for label,(p,pk) in refs.items():factor('fixed reference '+label,n,k,p,pk)
 if k!='pooled':factor('policy minus own pooled',n,k,n,'pooled')
for n,k in product(blends,policies):
 w,f,a=re.fullmatch(r'bagged_anchor_(balanced|unweighted)_(prob|logit)_a(10|25|50)',n).groups()
 g=f'logit_bagged_{w}_prob_mean' if f=='prob' else f'bagged_{w}_logit_mean'
 factor('blend minus anchor',n,k,H);factor('blend minus added margin',n,k,g)
for w,f,k in product(weights,['prob','logit'],policies):
 for low,high in [('10','25'),('10','50'),('25','50')]:factor('higher minus lower blend weight',f'bagged_anchor_{w}_{f}_a{high}',k,f'bagged_anchor_{w}_{f}_a{low}')
for f,a,k in product(['prob','logit'],['10','25','50'],policies):factor('blend balanced minus unweighted',f'bagged_anchor_balanced_{f}_a{a}',k,f'bagged_anchor_unweighted_{f}_a{a}')
for w,a,k in product(weights,['10','25','50'],policies):factor('blend mean-logit minus logit-mean-probability',f'bagged_anchor_{w}_logit_a{a}',k,f'bagged_anchor_{w}_prob_a{a}')
for n,k in product(logits,policies):factor('transformed minus native',n,k,n[6:])
for w,k in product(weights,policies):factor('added mean-logit minus logit-mean-probability',f'bagged_{w}_logit_mean',k,f'logit_bagged_{w}_prob_mean')
for f,k in product(['s17','s29','s43','prob_mean','logit_mean'],policies):factor('native balanced minus unweighted',f'bagged_balanced_{f}',k,f'bagged_unweighted_{f}')
for source_b,source_u in [(f'bagged_balanced_s{seed}',f'bagged_unweighted_s{seed}') for seed in [17,29,43]]+[('bagged_balanced_prob_mean','bagged_unweighted_prob_mean'),('baseline_balanced','baseline_lgbm')]:
 for k in policies:factor('logit balanced minus unweighted','logit_'+source_b,k,'logit_'+source_u)
for w,seed,k in product(weights,[17,29,43],policies):
 factor('probability aggregate minus native member',f'bagged_{w}_prob_mean',k,f'bagged_{w}_s{seed}')
 factor('mean-logit aggregate minus logit member',f'bagged_{w}_logit_mean',k,f'logit_bagged_{w}_s{seed}')
 factor('logit-probability aggregate minus logit member',f'logit_bagged_{w}_prob_mean',k,f'logit_bagged_{w}_s{seed}')
controls=expected-set(blends)-{v[0] for v in refs.values()}-{'logit_baseline_balanced','logit_baseline_lgbm'};assert len(controls)==18
for n,k in product(controls,policies):
 w='balanced' if '_balanced_' in n else 'unweighted';margin=n.startswith('logit_') or n.endswith('_logit_mean')
 factor('randomized minus matched unbagged',n,k,('logit_' if margin else '')+baseline[w])
assert len(facts)==947 and facts=={tuple(v) for v in e['contrasts']}

assert Counter(v[0] for v in e['contrasts'])==e['contrast_group_counts']
assert sum(e['contrast_group_counts'].values())==947 and len({tuple(v) for v in e['contrasts']})==947
for label,n,k,p,pk in e['contrasts']:assert n in expected and p in expected and k in policies and pk in policies
assert not any('_a00' in n or '_a100' in n for n in expected)
assert not any(n=='logit_bagged_'+w+'_logit_mean' for w in weights for n in expected)
assert e['expected_counts']['expression_leaf_rows']==6*6+6*4
assert e['production_fits_made']==e['new_blend_scores_evaluated']==e['new_consumed_bagged_scores_evaluated']==0
assert all(not Path(p).exists() for p in ['configs/bagged-anchor-fusion.json','docs/bagged-anchor-fusion-protocol.md','outputs/baf-bagged-anchor-fusion-v1','reports/baf-bagged-anchor-fusion-v1',*e['planned_source_paths'][-3:]])
assert len(e['mandatory_preservation_sha256'])==735 and len(e['historical_ignored_file_sha256'])==3732 and len(e['original_exported_selection_pairs'])==16
for p,h in e['mandatory_preservation_sha256'].items():assert digest(p)==h,p
for p,h in e['historical_ignored_file_sha256'].items():assert digest(p)==h,p
proof=dict(evidence_sha256=digest(root/'planning-evidence.json'),independent_strategy_ids=37,independent_winner_policy_cases=111,independent_role_mappings=111,independent_exact_dags=111,physical_development_scores=120,factor_rows_checked=947,independent_factor_keys=947,source_paths=55,mandatory_paths=735,selection_pairs=16,historical_files=3732,implementation_files_created=False,new_model_fits=0,new_blend_evaluations=0)
write_json(root/'planning-verification.json',proof);print(proof)
