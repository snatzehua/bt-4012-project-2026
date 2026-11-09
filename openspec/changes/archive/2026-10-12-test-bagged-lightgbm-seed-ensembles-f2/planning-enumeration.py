from pathlib import Path
from collections import Counter
from itertools import product
from baf.common import read_config,digest,write_json
from baf import unweighted_fusion_study as previous
from baf.unweighted_fusion_scores import REFERENCE_ROLES,dependencies as old_dependencies

c=read_config(previous.CONFIG)
previous.guard(c)
report=Path(c['report_dir'])
manifest=read_config(report/'report_hashes.json')
previous.verify_hashes(manifest)
closure=previous.SOURCES+['baf/bagged_boosting.py','baf/bagged_boosting_scores.py','baf/bagged_boosting_report.py']
assert len(closure)==len(set(closure))==52
preserved=dict(c['preserve_sha256'])
for p in previous.SOURCES[-3:]+[previous.CONFIG,c['protocol_path'],str(Path(c['output_dir'])/'selection.json'),str(report/'report_hashes.json')]:preserved[p]=digest(p)
preserved.update(manifest)
previous.verify_hashes(preserved)
selections=[p for p in preserved if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(selections)==15
assert all(preserved[p.replace('outputs/','reports/',1)]==preserved[p] for p in selections)
seeds=[17,29,43];weights=['balanced','unweighted'];policies=['pooled','latest','rank_fraction']
refs={n:k for n,k in REFERENCE_ROLES.values()}
seed_ids={f'bagged_{w}_s{s}':(w,s) for w,s in product(weights,seeds)}
means={f'bagged_{w}_{kind}_mean':(w,kind) for w,kind in product(weights,['prob','logit'])}
new={**seed_ids,**means};strategies=list(refs)+list(new)
assert len(strategies)==15
factors=[]
for n,k,ref in product(strategies,policies,REFERENCE_ROLES):factors.append(('fixed_reference',n,k,ref))
for n,k in product(strategies,['latest','rank_fraction']):factors.append(('own_policy',n,k))
for n,k in product(new,policies):factors.append(('unbagged',n,k))
for n,k,s in product(means,policies,seeds):factors.append(('member',n,k,s))
for w,k in product(weights,policies):factors.append(('aggregation',w,k))
for form,k in product(['s17','s29','s43','prob_mean','logit_mean'],policies):factors.append(('weighting',form,k))
assert len(factors)==len(set(factors))==342
refdag=old_dependencies(list(refs))
assert len(refdag['bases'])==5 and len(refdag['scores'])==8
def roles(n,k):
    result={'selected':[n,k],**{a:list(v) for a,v in REFERENCE_ROLES.items()},'same_strategy':[n,'pooled']}
    if n in refs:return result
    w,form=new[n];other='unweighted' if w=='balanced' else 'balanced'
    result.update({f'member_s{s}':[f'bagged_{w}_s{s}',k] for s in seeds})
    result.update(probability_mean=[f'bagged_{w}_prob_mean',k],logit_mean=[f'bagged_{w}_logit_mean',k],matched_weight=[f'bagged_{other}_s{form}' if n in seed_ids else f'bagged_{other}_{form}_mean',k],unbagged_member=['baseline_balanced' if w=='balanced' else 'baseline_lgbm',k])
    return result
def dag(names):
    scores=set(refdag['scores']);bases=set(refdag['bases'])
    def visit(n):
        if n in scores:return
        if n in means:
            w,_=means[n]
            for s in seeds:visit(f'bagged_{w}_s{s}')
        elif n in seed_ids:bases.add(n)
        else:raise ValueError(n)
        scores.add(n)
    for n in names:visit(n)
    return {'bases':sorted(bases),'scores':sorted(scores),'new_bagged_bases':sorted(bases&set(seed_ids))}
cases=[]
for n,k in product(strategies,policies):
    choices=roles(n,k);d=dag([v[0] for v in choices.values()]);kind='reference' if n in refs else ('single' if n in seed_ids else 'aggregate')
    expected={'reference':(7,5,8,0),'single':(14,9,14,4),'aggregate':(14,11,17,6)}[kind]
    assert (len(choices),len(d['bases']),len(d['scores']),len(d['new_bagged_bases']))==expected
    cases.append(dict(winner=n,policy=k,kind=kind,roles=choices,dependencies=d))
roots=sorted(set(c['historical_output_roots']+[c['output_dir']]))
historical={str(p):digest(p) for root in roots for p in Path(root).rglob('*') if p.is_file()}
reference_cache={}
for phase,months in [('validation',[4,5,6]),('exploratory',[7])]:
    for m,n in product(months,refdag['bases']):
        spec=next(s for s in c['candidates']+c['presets']+c['references'] if s['id']==n)
        paths=previous.cache_paths(c,spec,m,phase)
        assert all(p.is_file() for p in paths),(phase,m,n)
        reference_cache[f'{phase}-{m}-{n}']={str(p):digest(p) for p in paths}
out=Path('openspec/changes/test-bagged-lightgbm-seed-ensembles-f2')
write_json(out/'planning-evidence.json',dict(created_at_utc=previous.now(),no_new_model_fits=True,no_new_scores_evaluated=True,previous_fingerprint=previous.fingerprint(),source_paths=closure,planned_source_paths=closure[-3:],mandatory_preservation_sha256=preserved,mandatory_preservation_count=len(preserved),ancestor_selection_pairs=15,previous_report_files=len(manifest),previous_report_manifest_sha256=digest(report/'report_hashes.json'),historical_ignored_file_sha256=historical,historical_ignored_files=len(historical),reference_cache_sha256=reference_cache,strategies=strategies,contrast_family_counts=dict(Counter(x[0] for x in factors)),contrasts=[list(x) for x in factors],winner_policy_cases=cases,case_counts=dict(Counter(x['kind'] for x in cases)),expected_counts=dict(new_development_fits=18,reference_bases=15,physical_scores=54,selectable_scores=45,forward_rows=90,summary_rows=45,contrasts=342,seed_score_diversity=18,seed_decision_overlap=36,clipping=33,tiny_integration_fits=9,new_cache_test_fits_per_full_invocation=2)))
print('Verified planning:',len(preserved),'preserved paths;',len(historical),'historical files;45 roles/DAGs;342 contrasts;zero fits/scores.')
