"""Prospective enumeration and development-only source diagnostics; no fits/blends."""
from pathlib import Path
from itertools import product,combinations
from collections import Counter
import json
import numpy as np
from scipy.stats import rankdata
from baf.common import read_config,write_json,digest
from baf import bagged_boosting as parent
from baf.spline_fusion_report import independent_rule,independent_mask

HERE=Path(__file__).parent
W=['balanced','unweighted'];FORMS=['prob','logit'];ALPHAS=[10,25,50];POLICIES=['pooled','latest','rank_fraction']
REFS=parent.REFERENCE_ROLES;H=REFS['latest_fusion'][0]
PARENT=parent.CONFIG
NEW_SOURCES=['baf/bagged_anchor_scores.py','baf/bagged_anchor_study.py','baf/bagged_anchor_report.py']
BAG={s['id']:s for s in parent.recipes()}
BASELINE={'balanced':'baseline_balanced','unweighted':'baseline_lgbm'}
PROBMEAN={w:f'bagged_{w}_prob_mean' for w in W}
LOGITMEAN={w:f'bagged_{w}_logit_mean' for w in W}
LOGITS={f'logit_{n}':n for n in list(BAG)+list(PROBMEAN.values())+list(BASELINE.values())}
BLENDS={f'bagged_anchor_{w}_{f}_a{a:02d}':dict(weighting=w,form=f,alpha=a/100,anchor=H,
        margin=f'logit_{PROBMEAN[w]}' if f=='prob' else LOGITMEAN[w]) for w,f,a in product(W,FORMS,ALPHAS)}
PARENT_IDS={s['id'] for s in parent.strategies()}
NAMES=list(parent.REFERENCE_POLICIES)+list(BAG)+[f'bagged_{w}_{f}_mean' for w,f in product(W,FORMS)]+list(LOGITS)+list(BLENDS)
assert len(NAMES)==len(set(NAMES))==37

def dag(names):
    scores=set();bases=set();active=set()
    def visit(n):
        if n in scores:return
        assert n not in active;active.add(n)
        if n in PARENT_IDS:
            d=parent.dependencies([n]);scores.update(d['scores']);bases.update(d['bases'])
        elif n in LOGITS:visit(LOGITS[n]);scores.add(n)
        elif n in BLENDS:visit(H);visit(BLENDS[n]['margin']);scores.add(n)
        else:raise AssertionError(n)
        active.remove(n)
    for n in names:visit(n)
    return dict(bases=sorted(bases),scores=sorted(scores),derived=sorted(scores-bases),bagged_bases=sorted(bases&set(BAG)))

def other(w):return 'unweighted' if w=='balanced' else 'balanced'
def weighting(n):return 'balanced' if '_balanced' in n else 'unweighted'
def matched(n):
    if n=='baseline_balanced':return 'baseline_lgbm'
    if n=='baseline_lgbm':return 'baseline_balanced'
    if n=='logit_baseline_balanced':return 'logit_baseline_lgbm'
    if n=='logit_baseline_lgbm':return 'logit_baseline_balanced'
    w=weighting(n);return n.replace('_'+w+'_','_'+other(w)+'_',1)

def choices(n,k):
    result=dict(selected=[n,k],**{a:list(v) for a,v in REFS.items()},same_strategy=[n,'pooled'])
    if n in parent.REFERENCE_POLICIES:return result,'reference'
    if n in {'logit_baseline_balanced','logit_baseline_lgbm'}:
        result.update(native_member=[LOGITS[n],k],matched_weight=[matched(n),k]);return result,'baseline_logit'
    w=BLENDS[n]['weighting'] if n in BLENDS else weighting(n)
    for seed in [17,29,43]:
        result[f'member_s{seed}']=[f'bagged_{w}_s{seed}',k]
        result[f'member_logit_s{seed}']=[f'logit_bagged_{w}_s{seed}',k]
    result.update(probability_mean=[PROBMEAN[w],k],probability_mean_logit=['logit_'+PROBMEAN[w],k],logit_mean=[LOGITMEAN[w],k],
                  unbagged_member=[BASELINE[w],k],unbagged_logit=['logit_'+BASELINE[w],k])
    if n in BLENDS:
        s=BLENDS[n];f=s['form'];a=round(s['alpha']*100)
        result.update(anchor_at_policy=[H,k],added_member_native=[PROBMEAN[w] if f=='prob' else LOGITMEAN[w],k],
                      added_member_margin=[s['margin'],k],matched_weight=[matched(n),k],
                      matched_form=[f'bagged_anchor_{w}_{"logit" if f=="prob" else "prob"}_a{a:02d}',k])
        for q in ALPHAS:result[f'alpha_a{q:02d}']=[f'bagged_anchor_{w}_{f}_a{q:02d}',k]
        return result,'blend'
    native=LOGITS[n] if n in LOGITS else n
    margin=('logit_'+native) if native in BAG or native in PROBMEAN.values() else native
    result.update(native_member=[native,k],transformed_member=[margin,k],matched_weight=[matched(n),k])
    return result,'single' if native in BAG else 'aggregate'

def factors():
    rows=[]
    def add(group,n,k,r,rk=None):rows.append((group,n,k,r,k if rk is None else rk))
    for n,k in product(NAMES,POLICIES):
        for a,(r,rk) in REFS.items():add('fixed reference '+a,n,k,r,rk)
        if k!='pooled':add('policy minus own pooled',n,k,n,'pooled')
    for n,s in BLENDS.items():
        for k in POLICIES:
            add('blend minus anchor',n,k,H);add('blend minus added margin',n,k,s['margin'])
    for w,f,(a,b),k in product(W,FORMS,list(combinations(ALPHAS,2)),POLICIES):
        add('higher minus lower blend weight',f'bagged_anchor_{w}_{f}_a{b:02d}',k,f'bagged_anchor_{w}_{f}_a{a:02d}')
    for f,a,k in product(FORMS,ALPHAS,POLICIES):add('blend balanced minus unweighted',f'bagged_anchor_balanced_{f}_a{a:02d}',k,f'bagged_anchor_unweighted_{f}_a{a:02d}')
    for w,a,k in product(W,ALPHAS,POLICIES):add('blend mean-logit minus logit-mean-probability',f'bagged_anchor_{w}_logit_a{a:02d}',k,f'bagged_anchor_{w}_prob_a{a:02d}')
    for n,p in LOGITS.items():
        for k in POLICIES:add('transformed minus native',n,k,p)
    for w,k in product(W,POLICIES):add('added mean-logit minus logit-mean-probability',LOGITMEAN[w],k,'logit_'+PROBMEAN[w])
    for form,k in product(['s17','s29','s43','prob_mean','logit_mean'],POLICIES):add('native balanced minus unweighted',f'bagged_balanced_{form}',k,f'bagged_unweighted_{form}')
    for n,k in product([n for n in LOGITS if weighting(n)=='balanced'],POLICIES):add('logit balanced minus unweighted',n,k,matched(n))
    for w,k in product(W,POLICIES):
        for seed in [17,29,43]:
            add('probability aggregate minus native member',PROBMEAN[w],k,f'bagged_{w}_s{seed}')
            add('mean-logit aggregate minus logit member',LOGITMEAN[w],k,f'logit_bagged_{w}_s{seed}')
            add('logit-probability aggregate minus logit member','logit_'+PROBMEAN[w],k,f'logit_bagged_{w}_s{seed}')
    bagged_controls=list(BAG)+list(PROBMEAN.values())+list(LOGITMEAN.values())+[n for n,p in LOGITS.items() if p.startswith('bagged_')]
    assert len(bagged_controls)==18
    for n,k in product(bagged_controls,POLICIES):
        w=weighting(n);is_margin=n.startswith('logit_') or n in LOGITMEAN.values()
        add('randomized minus matched unbagged',n,k,('logit_' if is_margin else '')+BASELINE[w])
    assert len(rows)==len(set(rows))==947
    assert all(n in NAMES and r in NAMES for _,n,_,r,_ in rows)
    return rows


def main():
    c=read_config(PARENT);parent.guard(c);fp=parent.fingerprint();selection=read_config(Path(c['output_dir'])/'selection.json')
    assert fp==selection['fingerprint'];assert len(fp['sources'])==52
    preserved=dict(c['preserve_sha256'])
    for p in NEW_SOURCES:assert not Path(p).exists()
    for p in parent.SOURCES[-3:]+[PARENT,c['protocol_path'],str(Path(c['output_dir'])/'selection.json'),str(Path(c['report_dir'])/'report_hashes.json')]:preserved[p]=digest(p)
    preserved.update(read_config(Path(c['report_dir'])/'report_hashes.json'));parent.verify_hashes(preserved);assert len(preserved)==735
    pairs=[dict(original=p,exported=p.replace('outputs/','reports/',1),sha256=h) for p,h in preserved.items() if p.startswith('outputs/') and p.endswith('/selection.json')]
    assert len(pairs)==16 and all(preserved[v['exported']]==v['sha256'] for v in pairs)
    roots=sorted(set(c['historical_output_roots']+[c['output_dir']]))
    present={str(p):digest(p) for root in roots for p in Path(root).rglob('*') if p.is_file()};assert len(present)==3732
    available={}
    for spec in parent.reference_specs(c):
        ps=parent.previous.cache_paths(c,spec,7,'exploratory');assert parent.cache_state(ps);available[spec['id']]={str(p):digest(p) for p in ps}
    for n in BAG:assert not parent.cache_state(parent.bagged_paths(c,'exploratory',7,n))
    cases=[]
    for n,k in product(NAMES,POLICIES):
        roles,kind=choices(n,k);d=dag([v[0] for v in roles.values()]);expected={'reference':7,'baseline_logit':9,'single':21,'aggregate':21,'blend':26}[kind]
        assert len(roles)==expected and len(d['bases'])<=11 and len(d['scores'])<=28
        cases.append(dict(winner=n,policy=k,kind=kind,roles=roles,dependencies=d,
                          current_new_bagged_fit_bound=len(d['bagged_bases']),current_reference_reproduction_bound=0,
                          fresh_additive_reproduction_bound=2,fresh_old_tree_reproduction_bound=3))
    assert len(cases)==111 and Counter(v['kind'] for v in cases)==dict(reference=15,baseline_logit=6,single=36,aggregate=18,blend=36)
    whole=dag(NAMES);assert (len(whole['bases']),len(whole['scores']),len(whole['derived']))==(11,40,29)
    pairs_diag=[('anchor versus '+w+' '+f,H,'logit_'+PROBMEAN[w] if f=='prob' else LOGITMEAN[w]) for w,f in product(W,FORMS)]
    pairs_diag += [('aggregation forms '+w,'logit_'+PROBMEAN[w],LOGITMEAN[w]) for w in W]
    # Existing-source diagnostics only: do not compute any new blend or its F2.
    history=parent.history(c);diagnostics=[]
    for w,f,m,k in product(W,FORMS,[5,6],POLICIES):
        n=PROBMEAN[w] if f=='prob' else LOGITMEAN[w];a,b=history[H][m],history[n][m]
        parent.aligned(a,b);pa=independent_mask(a,independent_rule(history[H],k,m));pb=independent_mask(b,independent_rule(history[n],k,m))
        y=a['labels'].astype(bool)
        diagnostics.append(dict(weighting=w,form=f,month=m,policy=k,anchor_tp=int((pa&y).sum()),member_tp=int((pb&y).sum()),
            fraud_member_only=int((~pa&pb&y).sum()),fraud_anchor_only=int((pa&~pb&y).sum()),
            legitimate_member_only=int((~pa&pb&~y).sum()),legitimate_anchor_only=int((pa&~pb&~y).sum()),
            spearman=float(np.corrcoef(rankdata(a['scores']),rankdata(b['scores']))[0,1])))
    closure=parent.SOURCES+NEW_SOURCES;assert len(set(closure))==55
    group_counts={kind:dict(cases=sum(v['kind']==kind for v in cases),roles=len(next(v for v in cases if v['kind']==kind)['roles']),
                      max_bases=max(len(v['dependencies']['bases']) for v in cases if v['kind']==kind),
                      max_scores=max(len(v['dependencies']['scores']) for v in cases if v['kind']==kind),
                      max_new_bagged_fits=max(v['current_new_bagged_fit_bound'] for v in cases if v['kind']==kind)) for kind in ['reference','baseline_logit','single','aggregate','blend']}
    evidence=dict(parent_config=PARENT,parent_fingerprint=fp,parent_selection_sha256=digest(Path(c['output_dir'])/'selection.json'),
        planned_source_paths=closure,existing_source_sha256=fp['sources'],mandatory_preservation_sha256=preserved,
        original_exported_selection_pairs=pairs,historical_output_roots=roots,historical_ignored_file_sha256=present,
        strategies=NAMES,logit_controls=LOGITS,blends=BLENDS,alphas=[.1,.25,.5],epsilon=1e-6,
        complete_dependencies=whole,winner_policy_cases=cases,winner_case_summary=group_counts,
        contrasts=factors(),contrast_group_counts=dict(Counter(v[0] for v in factors())),
        final_reference_availability=available,development_source_diagnostics=diagnostics,diagnostic_pair_declarations=pairs_diag,
        expected_counts=dict(strategies=37,physical_development_scores=120,selectable_scores=111,forward_rows=222,
            macro_rows=111,pooled_rows=111,score_rows=111,contrasts=947,source_bases=33,source_derived=21,
            new_derived=66,diversity_rows=18,overlap_rows=36,clipping_rows=39,expression_leaf_rows=60,winner_policy_cases=111,
            development_draws=1000,consumed_draws=500,new_production_miniature_fits=0,cache_test_fits_per_full_invocation=6),
        production_fits_made=0,new_blend_scores_evaluated=0,new_consumed_bagged_scores_evaluated=0)
    write_json(HERE/'planning-evidence.json',evidence)
    print(json.dumps(dict(source_paths=55,preserved_paths=len(preserved),selection_pairs=len(pairs),historical_files=len(present),
         counts=evidence['expected_counts'],winner_case_summary=group_counts,contrast_groups=evidence['contrast_group_counts']),indent=2))

if __name__=='__main__':main()
