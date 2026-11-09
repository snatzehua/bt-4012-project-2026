"""Finite randomized-row sources and fixed, aligned three-seed aggregates."""
from itertools import product, combinations
import numpy as np
from scipy.stats import rankdata
from .recall_weighted_metrics import POLICIES, aligned, check_data
from .unweighted_fusion_scores import REFERENCE_ROLES, ANCHORS, dependencies as old_dependencies
from .score_fusion import clipping

SEEDS = [17, 29, 43]
WEIGHTINGS = ['balanced', 'unweighted']
EPSILON = 1e-6
REFERENCE_POLICIES = {n:k for n,k in REFERENCE_ROLES.values()}
REFERENCE_BASES = old_dependencies(list(REFERENCE_POLICIES))['bases']


def recipes():
    return [dict(id=f'bagged_{w}_s{s}',kind='bagged',weighting=w,bagging_seed=s,
                 fraction=.8,frequency=1) for w,s in product(WEIGHTINGS,SEEDS)]


def strategies():
    return ([dict(id=n,kind='reference') for n in REFERENCE_POLICIES]+
            recipes()+[dict(id=f'bagged_{w}_{form}_mean',kind='aggregate',weighting=w,
                            form=form,members=[f'bagged_{w}_s{s}' for s in SEEDS])
                       for w,form in product(WEIGHTINGS,['prob','logit'])])


def lookup(name):
    found=next((s for s in strategies() if s['id']==name),None)
    if found is None:raise ValueError('Undeclared strategy')
    return found


def probability(values):
    p=np.asarray(values,dtype=np.float64)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probability-like member')
    return p


def compose(spec,items):
    if spec not in strategies() or spec['kind']!='aggregate':
        raise ValueError('Undeclared aggregate')
    if set(items)!=set(spec['members']) or len(items)!=3:
        raise ValueError('Exactly three declared members required')
    ds=[items[n] for n in spec['members']]
    values=[]
    for d in ds:
        aligned(ds[0],d);p=probability(d['scores'])
        if spec['form']=='logit':
            p=np.clip(p,EPSILON,1-EPSILON);p=np.log(p)-np.log1p(-p)
        values.append(p)
    result=dict(row_ids=ds[0]['row_ids'],labels=ds[0]['labels'],
                scores=np.add.reduce(np.stack(values),axis=0,dtype=np.float64)/3.)
    check_data(result)
    return result


def unbagged(weighting):
    if weighting not in WEIGHTINGS:raise ValueError('Undeclared weighting')
    return 'baseline_balanced' if weighting=='balanced' else 'baseline_lgbm'


def matched_weight(name):
    s=lookup(name)
    if s['kind']=='reference':raise ValueError('Reference has fixed identity')
    other='unweighted' if s['weighting']=='balanced' else 'balanced'
    return (f'bagged_{other}_s{s["bagging_seed"]}' if s['kind']=='bagged'
            else f'bagged_{other}_{s["form"]}_mean')


def choices_for(name,policy):
    s=lookup(name)
    if policy not in POLICIES:raise ValueError('Undeclared policy')
    result=dict(selected=[name,policy],**{a:list(v) for a,v in REFERENCE_ROLES.items()},
                same_strategy=[name,'pooled'])
    if name in REFERENCE_POLICIES:return result
    w=s['weighting']
    result.update({f'member_s{seed}':[f'bagged_{w}_s{seed}',policy] for seed in SEEDS})
    result.update(probability_mean=[f'bagged_{w}_prob_mean',policy],
                  logit_mean=[f'bagged_{w}_logit_mean',policy],
                  matched_weight=[matched_weight(name),policy],
                  unbagged_member=[unbagged(w),policy])
    return result


def dependencies(names):
    scores=set();bases=set();active=set()
    def visit(n):
        if n in scores:return
        if n in active:raise ValueError('Cyclic dependency')
        active.add(n)
        if n in REFERENCE_POLICIES:
            dag=old_dependencies([n]);scores.update(dag['scores']);bases.update(dag['bases'])
        else:
            s=lookup(n)
            if s['kind']=='bagged':bases.add(n)
            else:
                for member in s['members']:visit(member)
            scores.add(n)
        active.remove(n)
    for n in names:visit(n)
    return dict(bases=sorted(bases),scores=sorted(scores),derived=sorted(scores-bases),
                bagged_bases=sorted(bases-{*REFERENCE_BASES}))


def contrast_factors():
    result=[];new=[s for s in strategies() if s['kind']!='reference']
    for s,k in product(strategies(),POLICIES):
        n=s['id']
        result += [('fixed reference '+a,n,k,r,rk) for a,(r,rk) in REFERENCE_ROLES.items()]
        if k!='pooled':result.append(('policy minus own pooled',n,k,n,'pooled'))
    for s,k in product(new,POLICIES):
        n=s['id'];result.append(('randomized minus unbagged',n,k,unbagged(s['weighting']),k))
        if s['kind']=='aggregate':
            result += [('aggregate minus member',n,k,member,k) for member in s['members']]
    for w,k in product(WEIGHTINGS,POLICIES):
        result.append(('logit mean minus probability mean',f'bagged_{w}_logit_mean',k,
                       f'bagged_{w}_prob_mean',k))
    for s,k in product([s for s in new if s['weighting']=='balanced'],POLICIES):
        result.append(('balanced minus unweighted',s['id'],k,matched_weight(s['id']),k))
    if len(result)!=342 or len(set(result))!=342:raise ValueError('Contrast union differs')
    return result


def correlations(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)!=len(b) or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid correlation population')
    constant=np.ptp(a)==0 or np.ptp(b)==0
    return dict(pearson=None if constant else float(np.corrcoef(a,b)[0,1]),
                spearman=None if constant else float(np.corrcoef(rankdata(a),rankdata(b))[0,1]),
                correlation_reason='constant member' if constant else 'defined',
                mean_absolute_difference=float(np.abs(a-b).mean()),
                unequal_scores=int((a!=b).sum()))


def overlap(a,b,pa,pb):
    aligned(a,b)
    pa,pb=np.asarray(pa,dtype=bool),np.asarray(pb,dtype=bool)
    if pa.shape!=a['labels'].shape or pb.shape!=pa.shape:raise ValueError('Invalid decision shape')
    masks=dict(both=pa&pb,first_only=pa&~pb,second_only=~pa&pb,neither=~pa&~pb)
    result={k:int(v.sum()) for k,v in masks.items()}
    result.update({label+'_'+k:int(((a['labels']==y)&v).sum())
                   for y,label in [(1,'fraud'),(0,'legitimate')] for k,v in masks.items()})
    return result


def declaration_counts():
    return dict(strategies=15,new_development_fits=18,reference_bases=15,all_scores=54,
                selectable_scores=45,forward_rows=90,macro_rows=45,pooled_rows=45,score_rows=45,
                contrasts=342,diversity_rows=18,overlap_rows=36,clipping_rows=33,
                old_derived_results=9,new_derived_results=12,tiny_integration_fits=9,
                new_cache_test_fits_per_full_invocation=2,winner_policy_cases=45,
                development_draws=1000,consumed_draws=500)
