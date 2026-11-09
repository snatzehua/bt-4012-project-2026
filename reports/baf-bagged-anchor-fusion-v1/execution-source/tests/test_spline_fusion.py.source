"""Methodological checks for mixed-domain fixed composition and frozen controls."""
from collections import Counter
from copy import deepcopy
import json

import numpy as np
import pytest

from baf.spline_fusion_scores import (ALPHAS,EPSILON,FUSION,MEMBERS,REFERENCES,
    blend_id,choices_for,common_margin,compose,contrast_factors,diversity,fixtures,strategies)
from baf.spline_fusion_report import independent_rule,independent_mask
from baf.recall_weighted_metrics import learn,decisions
from baf import spline_fusion_study as run


def data(scores,labels=None,ids=None):
    return dict(scores=np.asarray(scores,dtype=float),labels=np.asarray(labels if labels is not None else [0,1,0,1],dtype=int),
                row_ids=np.asarray(ids if ids is not None else [10,11,12,13],dtype=int))


@pytest.mark.parametrize('alpha',ALPHAS)
def test_common_margin_endpoints_and_weight_meaning(alpha):
    p=np.array([0.,1.,.1,.9]);g=np.array([-100.,100.,-2.,3.])
    clipped=np.maximum(EPSILON,np.minimum(1-EPSILON,p));u=np.log(clipped/(1-clipped))
    np.testing.assert_allclose(common_margin(p,g,alpha),alpha*u+(1-alpha)*g,atol=1e-12,rtol=0)
    assert np.isfinite(common_margin(p,g,alpha)).all()
    # Feeding the existing margin through a probability logit would reject it.
    assert (g<0).any() and (g>1).any()


@pytest.mark.parametrize('p,g,a,eps',[
    ([-.1,.2,.3,.4],[0,1,2,3],.1,EPSILON),([.1,1.1,.3,.4],[0,1,2,3],.1,EPSILON),
    ([0,1,np.nan,.4],[0,1,2,3],.1,EPSILON),([0,1,.3,.4],[0,1,np.inf,3],.1,EPSILON),
    ([0,1,.3,.4],[0,1],.1,EPSILON),([[0,1],[.3,.4]],[0,1,2,3],.1,EPSILON),
    ([0,1,.3,.4],[0,1,2,3],0,EPSILON),([0,1,.3,.4],[0,1,2,3],.75,EPSILON),
    ([0,1,.3,.4],[0,1,2,3],np.nan,EPSILON),([0,1,.3,.4],[0,1,2,3],.1,.01),
])
def test_reject_invalid_domains(p,g,a,eps):
    with pytest.raises(ValueError):common_margin(p,g,a,eps)


def test_row_label_permutation_and_other_row_invariance():
    n=MEMBERS[0];p=data([.1,.9,.3,.7]);g=data([-3,5,-1,2])
    logit=next(s for s in strategies() if s['id']=='logit_'+n)
    blend=next(s for s in strategies() if s['id']==blend_id(n,.25))
    u=compose(logit,{n:p});actual=compose(blend,{'logit_'+n:u,FUSION:g})
    changed_p=deepcopy(p);changed_g=deepcopy(g)
    changed_p['labels']=1-p['labels'];changed_g['labels']=changed_p['labels'].copy()
    changed_p['scores'][2]=.99;changed_g['scores'][2]=-1000
    changed=compose(blend,{'logit_'+n:compose(logit,{n:changed_p}),FUSION:changed_g})
    np.testing.assert_array_equal(actual['scores'][[0,1,3]],changed['scores'][[0,1,3]])
    order=np.array([3,0,2,1]);permuted={key:value[order] for key,value in p.items()};pg={key:value[order] for key,value in g.items()}
    result=compose(blend,{'logit_'+n:compose(logit,{n:permuted}),FUSION:pg})
    np.testing.assert_array_equal(result['scores'],actual['scores'][order])


@pytest.mark.parametrize('failure',['labels','ids','missing','undeclared','double_logit','nonfinite'])
def test_alignment_and_strategy_guards(failure):
    n=MEMBERS[0];spec=next(s for s in strategies() if s['id']==blend_id(n,.1))
    a=data([-1,2,-3,4]);b=data([-5,6,-7,8]);items={'logit_'+n:a,FUSION:b}
    if failure=='labels':b['labels']=1-b['labels']
    if failure=='ids':b['row_ids']=b['row_ids'][::-1]
    if failure=='missing':items.pop(FUSION)
    if failure=='undeclared':spec={**spec,'alpha':.75}
    if failure=='double_logit':a['scores'][0]=-100
    if failure=='nonfinite':b['scores'][0]=np.inf
    with pytest.raises(ValueError):compose(spec,items)


def test_strategy_and_contrast_factor_sets():
    specs=strategies();assert len(specs)==23 and len({s['id'] for s in specs})==23
    assert Counter(s['kind'] for s in specs)==dict(raw=4,logit=4,reference=3,margin_blend=12)
    rows=contrast_factors();assert Counter(r[0] for r in rows)=={
        'old pooled reference':207,'policy minus own pooled':46,'logit minus raw':12,
        'blend minus common-margin member':72,'spline minus linear blend':18,
        'balanced minus unweighted blend':18,'higher minus lower Newton weight':36,
        'spline minus linear single':12,'balanced minus unweighted single':12}
    lookup={s['id']:s for s in specs}
    for label,n,k,r,rk in rows:
        assert n in lookup and r in lookup
        if label in ['spline minus linear blend','balanced minus unweighted blend']:
            assert lookup[n]['alpha']==lookup[r]['alpha'] and k==rk
        if label=='spline minus linear blend':
            assert 'spline' in n and 'linear' in r and n.split('_c0p1_')[1]==r.split('_c0p1_')[1]
        if label=='higher minus lower Newton weight':
            assert lookup[n]['members']==lookup[r]['members'] and lookup[n]['alpha']>lookup[r]['alpha']


@pytest.mark.parametrize('spec',strategies(),ids=lambda s:s['id'])
def test_all_winner_types_have_complete_controls(spec):
    roles=choices_for(spec['id'],'latest');kind=spec['kind'];n={'reference':5,'raw':7,'logit':8,'margin_blend':10}[kind]
    assert len(roles)==n
    assert roles['same_model']==[spec['id'],'pooled'] and roles['selected']==[spec['id'],'latest']
    assert all(name in {s['id'] for s in strategies()} for name,k in roles.values())
    if kind=='margin_blend':
        lookup={s['id']:s for s in strategies()}
        for role in ['matched_representation','matched_weight']:
            assert lookup[roles[role][0]]['alpha']==spec['alpha'] and roles[role][1]=='latest'
        assert len(run.required_newton({name for name,k in roles.values()}))==3


def test_undefined_correlations_are_explicit():
    row=diversity([1,1,1],[2,3,4]);assert row['correlation_undefined'] and row['pearson'] is None and row['spearman'] is None
    row=diversity([1,2,3],[6,4,2]);assert not row['correlation_undefined'];assert row['pearson']==pytest.approx(-1)


@pytest.mark.parametrize('kind',['pooled','latest','rank_fraction'])
def test_independent_policy_matches_earlier_only_with_ties(kind):
    h={4:data([.1,.5,.5,.9]),5:data([.8,.2,.2,.6],[1,0,1,0],[20,21,22,23])}
    expected=independent_rule(h,kind,6);assert expected==learn(h,kind,6)
    d=data([.2,.2,.2,.9],[1,0,0,1],[40,41,42,43])
    np.testing.assert_array_equal(independent_mask(d,expected),decisions(d,expected))
    changed={**h,6:d};assert independent_rule(changed,kind,6)==expected
    assert expected['source_months']==([5] if kind=='latest' else [4,5])


def test_partial_cache_and_missing_cache_rejected(tmp_path):
    paths=[tmp_path/n for n in ['record','prediction','model']]
    assert not run.tree.cache_state(paths)
    paths[0].write_text('partial')
    with pytest.raises(ValueError,match='Partial'):run.tree.cache_state(paths)
    for p in paths:p.write_text('whole')
    assert run.tree.cache_state(paths)
    hashes={str(p):run.digest(p) for p in paths};run.verify_hashes(hashes)
    paths[2].unlink()
    with pytest.raises(ValueError,match='Preserved'):run.verify_hashes(hashes)


def test_declaration_and_no_fit_fixtures():
    c=run.read_config(run.CONFIG);run.guard(c);result=fixtures()
    assert result['model_fits']==0 and result['strategies']==23 and result['contrasts']==433
    assert len(run.SOURCES)==40 and len(set(run.SOURCES))==40
    for field,value in [('beta',1),('epsilon',.001),('alphas',[.5]),('review_quota',.01)]:
        wrong=deepcopy(c);wrong[field]=value
        with pytest.raises(ValueError):run.guard(wrong)


@pytest.mark.parametrize('name',['baseline_balanced','baseline_lgbm'])
def test_wholly_absent_reference_reproduces_exact_recipe_only_in_destination(tmp_path,name):
    c=run.read_config(run.CONFIG);c['output_dir']=str(tmp_path/'new-study')
    source=run.read_config(c['source_config']);source['output_dir']=str(tmp_path/'absent-origin')
    source_config=tmp_path/'source.json';source_config.write_text(json.dumps(source));c['source_config']=str(source_config)
    base,y=run.newton.miniature_frame();frame=base.copy();frame['fraud_bool']=y
    frame['month']=np.repeat([0,1,2,3,4],32)
    spec=next(s for s in c['references'] if s['id']==name);identity={'fixture':'absent-reference-cache'}
    result=run.prepare_base(c,identity,frame,base,spec,4)
    assert not result['reused'] and result['reference_model_new_fit']
    assert result['train_rows']==128 and result['train_month_max']==3
    assert result['warnings']==[]
    assert not (tmp_path/'absent-origin').exists()
    files=run.cache_paths(c,spec,4,'validation');hashes={str(p):run.digest(p) for p in files}
    assert all(p.is_file() for p in files)
    # Repeated preparation verifies the same saved inference without another fit.
    assert run.prepare_base(c,identity,frame,base,spec,4)==result
    assert all(run.digest(p)==h for p,h in hashes.items())
    files[-1].unlink()
    with pytest.raises(ValueError,match='Partial'):
        run.prepare_base(c,identity,frame,base,spec,4)
