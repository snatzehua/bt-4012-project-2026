"""Label-free current-batch percentile composition and deterministic controls."""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from .recall_weighted_metrics import aligned,check_data


def percentile(scores):
    s=np.asarray(scores,dtype=float)
    if s.ndim!=1 or not len(s) or not np.isfinite(s).all():raise ValueError('Invalid percentile batch')
    return rankdata(s,method='average')/len(s)


def blend_scores(a,b,alpha):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid score member batch')
    if not np.isfinite(alpha) or not 0<alpha<1:raise ValueError('Invalid blend weight')
    return alpha*percentile(a)+(1-alpha)*percentile(b)


def strategies(new_ids,reference_ids):
    raw=list(new_ids)+list(reference_ids);pct=list(new_ids)+['baseline_balanced','baseline_lgbm']
    return [dict(id=n,kind='raw',members=[n]) for n in raw]+[
        dict(id='pct_'+n,kind='percentile',members=[n]) for n in pct]+[
        dict(id=f'blend_{n}_a{weight}',kind='blend',members=[n,'baseline_balanced'],alpha=weight/100)
        for n in new_ids for weight in [25,50,75]]


def compose(spec,members):
    items=[members[n] for n in spec['members']]
    for d in items:aligned(d,items[0])
    d=items[0]
    if spec['kind']=='raw':s=np.asarray(d['scores'])
    elif spec['kind']=='percentile':s=percentile(d['scores'])
    elif spec['kind']=='blend':s=blend_scores(d['scores'],items[1]['scores'],spec['alpha'])
    else:raise ValueError('Unknown score strategy')
    result=dict(row_ids=d['row_ids'],labels=d['labels'],scores=s);check_data(result);return result


def choices_for(spec,policy):
    name=spec['id'];choices=dict(selected=[name,policy],balanced=['baseline_balanced','pooled'],
        unweighted=['baseline_lgbm','pooled'],same_model=[name,'pooled'])
    if spec['kind']=='blend':
        choices.update(xgb_member=['pct_'+spec['members'][0],policy],balanced_member=['pct_baseline_balanced',policy],
                       raw_xgb=[spec['members'][0],policy])
    elif spec['kind']=='percentile':choices['raw_member']=[spec['members'][0],policy]
    elif name!='native31':choices['rank_member']=['pct_'+name,policy]
    return choices
