"""Fixed common-margin composition with explicit Newton probability domains."""
from itertools import combinations, product

import numpy as np
from scipy.stats import rankdata

from .recall_weighted_metrics import aligned, check_data, POLICIES
from .score_fusion import log_odds, clipping

FUSION = 'logit_mean_xgb_d4_w1_baseline_balanced_a50'
REFERENCES = [FUSION, 'baseline_balanced', 'baseline_lgbm']
MEMBERS = [f'{rep}_c0p1_{weight}' for rep in ['linear', 'spline']
           for weight in ['unweighted', 'balanced']]
ALPHAS = [.1, .25, .5]
EPSILON = 1e-6


def blend_id(member, alpha):
    if member not in MEMBERS or alpha not in ALPHAS:
        raise ValueError('Undeclared Newton member or weight')
    return f'margin_blend_{member}_a{int(alpha*100)}'


def strategies():
    return ([dict(id=n, kind='raw', members=[n]) for n in MEMBERS]
            + [dict(id='logit_'+n, kind='logit', members=[n]) for n in MEMBERS]
            + [dict(id=n, kind='reference', members=[n]) for n in REFERENCES]
            + [dict(id=blend_id(n,a), kind='margin_blend', members=['logit_'+n,FUSION],
                    alpha=a) for n in MEMBERS for a in ALPHAS])


def common_margin(probabilities, fusion_margin, alpha, epsilon=EPSILON):
    """Alpha is the Newton weight. Only probabilities receive a logit."""
    if alpha not in ALPHAS or epsilon != EPSILON:
        raise ValueError('Undeclared composition weight or epsilon')
    u = log_odds(probabilities, epsilon)
    g = np.asarray(fusion_margin, dtype=float)
    if g.ndim != 1 or g.shape != u.shape or not np.isfinite(g).all():
        raise ValueError('Invalid fusion margin')
    result = alpha*u + (1-alpha)*g
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite composition')
    return result


def compose(spec, items):
    if spec not in strategies() or spec['kind'] not in ['logit','margin_blend']:
        raise ValueError('Undeclared derived strategy')
    names=spec['members']
    if any(n not in items for n in names):
        raise ValueError('Missing score member')
    ds=[items[n] for n in names]
    for d in ds: aligned(d,ds[0])
    if spec['kind']=='logit':
        scores=log_odds(ds[0]['scores'],EPSILON)
    else:
        # Immediate parent is already a checked Newton logit margin.
        # Validate its bounded domain without transforming it twice.
        u=np.asarray(ds[0]['scores'],dtype=float)
        lo,hi=log_odds(np.array([0.,1.]),EPSILON)
        if ((u<lo)|(u>hi)).any():raise ValueError('Newton logit outside declared range')
        scores=spec['alpha']*u+(1-spec['alpha'])*ds[1]['scores']
    result=dict(row_ids=ds[0]['row_ids'],labels=ds[0]['labels'],scores=scores)
    check_data(result)
    return result


def matched(member, factor):
    if member not in MEMBERS:raise ValueError('Unknown Newton member')
    rep,_,weight=member.split('_')
    if factor=='representation':rep='spline' if rep=='linear' else 'linear'
    elif factor=='weight':weight='balanced' if weight=='unweighted' else 'unweighted'
    else:raise ValueError('Unknown matching factor')
    return f'{rep}_c0p1_{weight}'


def choices_for(name, policy):
    if policy not in POLICIES:raise ValueError('Undeclared policy')
    spec=next((s for s in strategies() if s['id']==name),None)
    if spec is None:raise ValueError('Undeclared winner')
    roles=dict(selected=[name,policy],fusion=[FUSION,'pooled'],
               balanced=['baseline_balanced','pooled'],unweighted=['baseline_lgbm','pooled'],
               same_model=[name,'pooled'])
    if spec['kind']=='reference':return roles
    raw=name if spec['kind']=='raw' else spec['members'][0].removeprefix('logit_')
    for role,factor in [('matched_representation','representation'),('matched_weight','weight')]:
        member=matched(raw,factor)
        target=(member if spec['kind']=='raw' else 'logit_'+member if spec['kind']=='logit'
                else blend_id(member,spec['alpha']))
        roles[role]=[target,policy]
    if spec['kind']=='logit':roles['raw_member']=[raw,policy]
    elif spec['kind']=='margin_blend':
        roles.update(transformed_newton_member=['logit_'+raw,policy],
                     fusion_member=[FUSION,policy],raw_newton_member=[raw,policy])
    return roles


def contrast_factors():
    rows=[]
    for s in strategies():
        n=s['id']
        for k in POLICIES:
            rows.extend(('old pooled reference',n,k,r,'pooled') for r in REFERENCES)
            if k!='pooled':rows.append(('policy minus own pooled',n,k,n,'pooled'))
            if s['kind']=='logit':rows.append(('logit minus raw',n,k,s['members'][0],k))
            if s['kind']=='margin_blend':
                rows.extend(('blend minus common-margin member',n,k,r,k) for r in s['members'])
    for weight,a,k in product(['unweighted','balanced'],ALPHAS,POLICIES):
        rows.append(('spline minus linear blend',blend_id('spline_c0p1_'+weight,a),k,
                     blend_id('linear_c0p1_'+weight,a),k))
    for rep,a,k in product(['linear','spline'],ALPHAS,POLICIES):
        rows.append(('balanced minus unweighted blend',blend_id(rep+'_c0p1_balanced',a),k,
                     blend_id(rep+'_c0p1_unweighted',a),k))
    for n,(lo,hi),k in product(MEMBERS,list(combinations(ALPHAS,2)),POLICIES):
        rows.append(('higher minus lower Newton weight',blend_id(n,hi),k,blend_id(n,lo),k))
    for prefix,weight,k in product(['','logit_'],['unweighted','balanced'],POLICIES):
        rows.append(('spline minus linear single',prefix+'spline_c0p1_'+weight,k,
                     prefix+'linear_c0p1_'+weight,k))
    for prefix,rep,k in product(['','logit_'],['linear','spline'],POLICIES):
        rows.append(('balanced minus unweighted single',prefix+rep+'_c0p1_balanced',k,
                     prefix+rep+'_c0p1_unweighted',k))
    if len(rows)!=433 or len(set(rows))!=433:raise ValueError('Incomplete contrasts')
    return rows


def diversity(u,g):
    u,g=np.asarray(u,dtype=float),np.asarray(g,dtype=float)
    if u.ndim!=1 or u.shape!=g.shape or not len(u) or not np.isfinite(u).all() or not np.isfinite(g).all():
        raise ValueError('Invalid diversity inputs')
    undefined=len(u)<2 or np.ptp(u)==0 or np.ptp(g)==0
    return dict(n=len(u),newton_min=float(u.min()),newton_max=float(u.max()),
                fusion_min=float(g.min()),fusion_max=float(g.max()),correlation_undefined=undefined,
                pearson=None if undefined else float(np.corrcoef(u,g)[0,1]),
                spearman=None if undefined else float(np.corrcoef(rankdata(u),rankdata(g))[0,1]))


def fixtures():
    p=np.array([0.,1.,.2,.8,.5]);g=np.array([-100.,100.,-2.,3.,0.])
    u=np.log(np.clip(p,EPSILON,1-EPSILON)/ (1-np.clip(p,EPSILON,1-EPSILON)))
    for a in ALPHAS:
        if not np.allclose(common_margin(p,g,a),a*u+(1-a)*g,atol=1e-12,rtol=0):
            raise ValueError('Independent composition arithmetic differs')
    return dict(model_fits=0,strategies=len(strategies()),contrasts=len(contrast_factors()),
                alpha_values=ALPHAS,endpoint_clipping=clipping(p),extreme_margins_checked=True)
