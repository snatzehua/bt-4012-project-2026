"""Methodological fixtures plus four explicitly counted tiny estimator fits."""
import copy
import itertools
import json
from collections import Counter
from pathlib import Path
import shutil
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from baf.common import digest, read_config, write_json
from baf import unweighted_fusion_study as run
from baf import unweighted_fusion_report as report
from baf import unweighted_fusion_scores as scores
from baf.interaction_inputs import InteractionInputs, storage
from baf.spline_inputs import AdditiveInputs




def data(values, offset=0):
    return dict(row_ids=np.arange(len(values))+offset, labels=np.arange(len(values))%2,
                scores=np.asarray(values,dtype=float))

def planning_evidence_path(root=Path('openspec/changes')):
    paths=[root/'test-unweighted-reference-fusion-f2'/'planning-evidence.json',
        root/'archive'/'2026-10-11-test-unweighted-reference-fusion-f2'/'planning-evidence.json']
    for path in paths:
        if path.is_file():return path
    raise FileNotFoundError('Preserved planning evidence is absent from active and archived locations')


def test_declared_counts_ancestry_and_independent_factors():
    c=read_config(run.CONFIG);run.guard(c)
    assert len(run.SOURCES)==len(set(run.SOURCES))==49
    assert len(c['preserve_sha256'])==529 and len(scores.strategies())==40
    evidence=read_config(planning_evidence_path())
    assert Counter(r[0] for r in scores.contrast_factors())==evidence['contrast_family_counts']
    assert set(scores.contrast_factors())==set(report.expected_factors())


@pytest.mark.parametrize('spec',[s for s in scores.strategies() if s['kind']!='native'])
def test_composition_arithmetic_domains_and_label_batch_invariance(spec):
    values=np.array([0.,1.,.2,.8,.5]);margin=np.array([-100.,100.,-2.,3.,0.])
    items={n:data(values if n in scores.BASES else margin) for n in spec['members']}
    actual=scores.compose(spec,items)
    def logit(v):
        v=np.maximum(1e-6,np.minimum(1-1e-6,v));return np.log(v)-np.log(1-v)
    u=logit(items[spec['members'][0]]['scores'])
    if spec['id']==scores.ALIAS:expected=.25*u+.75*logit(items[spec['members'][1]]['scores'])
    elif spec['id']==scores.FUSION:expected=.5*u+.5*logit(items[spec['members'][1]]['scores'])
    elif spec['kind']=='logit':expected=u
    else:
        w=spec['alpha'] if spec['kind']=='anchor' else spec['weight'];expected=w*u+(1-w)*margin
    np.testing.assert_allclose(actual['scores'],expected,atol=1e-12,rtol=1e-12)
    changed=copy.deepcopy(items)
    for d in changed.values():d['labels']=1-d['labels'];d['scores'][1:]=.4
    assert scores.compose(spec,changed)['scores'][0]==actual['scores'][0]
    single={n:{k:v[:1] for k,v in d.items()} for n,d in items.items()}
    assert scores.compose(spec,single)['scores'][0]==actual['scores'][0]


@pytest.mark.parametrize('change',['probability','nonfinite','shape','ids','labels','weight','recipe'])
def test_invalid_compositions_rejected(change):
    spec=scores.lookup(scores.strategy_id('latest','unweighted',.25))
    items={spec['members'][0]:data([.1,.9]),spec['members'][1]:data([-5.,6.])}
    if change=='probability':items[spec['members'][0]]['scores'][0]=-.01
    elif change=='nonfinite':items[spec['members'][1]]['scores'][0]=np.inf
    elif change=='shape':items[spec['members'][1]]['scores']=np.array([[1.,2.]])
    elif change=='ids':items[spec['members'][1]]['row_ids']=np.array([8,9])
    elif change=='labels':items[spec['members'][1]]['labels']=np.array([1,0])
    elif change=='weight':spec={**spec,'weight':1.}
    else:spec={**spec,'id':'renamed'}
    with pytest.raises(ValueError):scores.compose(spec,items)


@pytest.mark.parametrize('name',[s['id'] for s in scores.strategies()])
@pytest.mark.parametrize('policy',['pooled','latest','rank_fraction'])
def test_all_winner_policy_roles(name,policy):
    roles=scores.choices_for(name,policy);spec=scores.lookup(name)
    for role,value in scores.REFERENCE_ROLES.items():assert roles[role]==value
    assert roles['same_strategy']==[name,'pooled']
    if name in scores.ORIGINAL_POLICIES:assert len(roles)==7
    elif spec['kind']=='anchor':
        assert len(roles)==8 and roles['matched_anchor']==[scores.ANCHORS['latest'],policy]
    elif spec['kind']=='logit':
        assert len(roles)==9 and roles['native_added_member']==[scores.ADDED[spec['added']],policy]
    else:
        assert len(roles)==(15 if spec['anchor'] in ['latest','interaction'] else 14)
        assert roles['anchor_member']==[scores.ANCHORS[spec['anchor']],policy]
        assert roles['transformed_added_member']==['logit_'+scores.ADDED[spec['added']],policy]
        assert {scores.lookup(n)['weight'] for r,(n,k) in roles.items() if r.startswith('other_lambda_')}==set(scores.LAMBDAS)-{spec['weight']}
        matched=scores.lookup(roles['matched_added_weight'][0])
        assert matched['anchor']==spec['anchor'] and matched['weight']==spec['weight'] and matched['added']!=spec['added']
        if 'matched_anchor' in roles:
            matched=scores.lookup(roles['matched_anchor'][0])
            assert matched['added']==spec['added'] and matched['weight']==spec['weight'] and matched['anchor']!=spec['anchor']
    dag=run.dependencies([n for n,k in roles.values()])
    assert set(dag['derived'])==set(dag['scores'])-set(dag['bases'])
    assert len(dag['bases'])<=6 and len(dag['scores'])<=17
    evidence=read_config(planning_evidence_path())
    expected=next(v for v in evidence['all_winner_policy_cases'] if v['winner']==name and v['policy']==policy)
    assert (len(roles),len(dag['bases']),len(dag['scores']))==(expected['roles'],expected['bases'],expected['scores'])


def test_alias_uses_historical_direct_arithmetic_and_retains_ties():
    x=np.array([.010002,.010002,.8,0,1,.93]);b=np.array([.8,.8,.1,1,0,.23])
    spec=scores.lookup(scores.ALIAS);items={'xgb_d4_w1':data(x),'baseline_balanced':data(b)}
    actual=scores.compose(spec,items)['scores']
    old=.25*scores.log_odds(x)+.75*scores.log_odds(b)
    nested=.5*scores.log_odds(b)+.5*(.5*scores.log_odds(x)+.5*scores.log_odds(b))
    np.testing.assert_array_equal(actual,old)
    np.testing.assert_allclose(actual,nested,atol=1e-12,rtol=1e-12)
    assert actual[0]==actual[1] and sum(s['id']==scores.ALIAS for s in scores.strategies())==1
    assert len(scores.choices_for(scores.ALIAS,'pooled'))==14
    assert scores.leaf_coefficients(scores.ALIAS)=={'xgb_d4_w1':.25,'baseline_balanced':.75}


def tiny_context(tmp_path):
    x,y=run.newton.miniature_frame()
    frame=x.assign(month=np.where(np.arange(len(x))<120,3,4),fraud_bool=y)
    c=read_config(run.CONFIG);c.update(output_dir=str(tmp_path/'own'),threads=1)
    for key,identity in [('fusion_source_config','fusion_source_fingerprint'),
        ('interaction_source_config','interaction_source_fingerprint'),('newton_source_config','newton_fit_fingerprint'),
        ('recent_source_config','recent_source_fingerprint'),('source_config','source_fingerprint')]:
        p=tmp_path/(key+'.json');write_json(p,{**c,'output_dir':str(tmp_path/key)})
        c[key]=str(p);c[identity]={'fixture_origin':key}
    return c,{'fixture':True},frame,x


def source_spec(c,kind):
    return (next(s for s in run.recipes() if s['representation']==('spline' if kind=='additive' else 'interaction') and s['C']==1.)
        if kind in ['interaction','additive'] else c['references'][0] if kind=='lightgbm' else c['presets'][0])


@pytest.mark.parametrize('kind',['interaction','additive','lightgbm','xgboost'])
def test_four_missing_cache_formats_fit_once(tmp_path,kind):
    c,fp,frame,x=tiny_context(tmp_path)
    spec=source_spec(c,kind)
    r=run.prepare_base(c,fp,frame,x,spec,4)
    assert not r['reused'] and r['fingerprint']==fp
    assert all(p.is_file() for p in run.cache_paths(c,spec,4,'validation'))
    for key in ['fusion_source_config','interaction_source_config','newton_source_config','recent_source_config','source_config']:
        assert not Path(read_config(c[key])['output_dir']).exists()
    with patch.object(run.parent.parent,'fit_interaction',side_effect=AssertionError('no refit')), \
         patch.object(run.newton,'fit_new',side_effect=AssertionError('no refit')), \
         patch.object(run.tree.old,'fit_model',side_effect=AssertionError('no refit')), \
         patch.object(run.additive,'reproduce_reference',side_effect=AssertionError('no refit')):
        again=run.prepare_base(c,fp,frame,x,spec,4)
    assert again['prediction_sha256']==r['prediction_sha256']

@pytest.mark.parametrize('kind',['interaction','additive','lightgbm','xgboost'])
@pytest.mark.parametrize('where',['own','first_origin','fallback_origin'])
def test_partial_required_cache_never_falls_back_or_fits(tmp_path,kind,where):
    c,fp,frame,x=tiny_context(tmp_path)
    spec=source_spec(c,kind)
    origins=run.source_origins(c,spec)
    target=c if where=='own' else origins[0 if where=='first_origin' else -1][0]
    p=run.cache_paths(target,spec,4,'validation')[0];write_json(p,{'partial':True})
    with patch.object(run.parent.parent,'fit_interaction',side_effect=AssertionError('no fit')), \
         patch.object(run.newton,'fit_new',side_effect=AssertionError('no fit')), \
         patch.object(run.tree.old,'fit_model',side_effect=AssertionError('no fit')), \
         patch.object(run.additive,'reproduce_reference',side_effect=AssertionError('no fit')):
        with pytest.raises(ValueError,match='Partial'):run.prepare_base(c,fp,frame,x,spec,4)

def test_failed_gate_evidence_prevents_retry(tmp_path):
    c,fp,frame,x=tiny_context(tmp_path);spec=run.recipes()[0]
    error=run.newton.NumericalGate('gradient failure',{'gradient_max_abs':.5})
    with patch.object(run.newton,'fit_new',side_effect=error):
        with pytest.raises(ValueError,match='gradient'):run.prepare_base(c,fp,frame,x,spec,4)
    rp=run.cache_paths(c,spec,4,'validation')[0]
    evidence=read_config(Path(c['output_dir'])/'failures'/(rp.stem+'.json'))
    assert evidence['numerical']['gradient_max_abs']==.5 and evidence['spec']==spec
    with patch.object(run.newton,'fit_new',side_effect=AssertionError('no retry')):
        with pytest.raises(ValueError,match='Prior failed gate'):run.prepare_base(c,fp,frame,x,spec,4)

def test_constant_diversity_has_null_correlations_and_full_joint_counts():
    a,b=data([.5]*4),data([-2.,-1.,1.,2.]);pa=np.array([False,True,False,True]);pb=np.array([False,False,True,True])
    r=scores.diversity(a,b,pa,pb)
    assert r['pearson'] is None and r['spearman'] is None and r['correlation_reason']=='constant member'
    assert sum(r[k] for k in ['both_flagged','member_only','tree_only','neither_flagged'])==4
    for label in ['fraud','legitimate']:
        assert sum(r[label+'_'+k] for k in ['both','member_only','tree_only','neither'])==2

@pytest.fixture(scope='module')
def analytical_grid(tmp_path_factory):
    """Complete finite export with zero fitted estimators; balanced duplicate pairs."""
    root=tmp_path_factory.mktemp('analytical_fusion');x,_=run.newton.miniature_frame()
    x=x.iloc[:56].iloc[np.repeat(np.arange(56),2)].reset_index(drop=True)
    frame=x.assign(month=np.repeat(np.arange(7),16),fraud_bool=np.tile([0,1],56))
    c=read_config(run.CONFIG);c['output_dir']=str(root/'own');fp={'analytical_fixture':True}
    for key,identity in [('fusion_source_config','fusion_source_fingerprint'),('source_config','source_fingerprint')]:
        origin=root/(key+'.json');write_json(origin,{**c,'output_dir':str(root/key)})
        c[key]=str(origin);c[identity]={'absent_fixture':True}
    for m in [4,5,6]:
        train=frame.index[frame.month<m];ev=frame.index[frame.month==m]
        for spec in run.recipes():
            inp=(InteractionInputs().fit(x.loc[train]) if spec['representation']=='interaction' else AdditiveInputs('spline').fit(x.loc[train]))
            z=inp.transform(x.loc[train]);model=LogisticRegression(**run.newton.solver_parameters(spec))
            model.classes_=np.array([0,1]);model.n_features_in_=len(inp.names);model.n_iter_=np.array([1])
            model.coef_=np.zeros((1,len(inp.names)));model.intercept_=np.zeros(1)
            numerical=dict(**run.newton.objective_gradient(model,z,frame.loc[train,'fraud_bool'].to_numpy(),spec),
                fallback_count=0,fallback_function_sha256=run.newton.fallback_identity(),iterations=1,warnings=[],
                estimated_hessian_bytes=8*(len(inp.names)+1)**2)
            rp,pp,mp=run.cache_paths(c,spec,m,'validation');mp.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(dict(model=model,inputs=inp,spec=spec),mp);pp.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(pp,row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=np.full(len(ev),.5))
            write_json(rp,dict(**run.expected_record(frame,spec,m,'validation',fp),prediction_sha256=digest(pp),model_sha256=digest(mp),
                feature_metadata=inp.metadata(),warnings=[],iterations=1,solver_params=run.newton.solver_parameters(spec),
                numerical=numerical,transformed_dimension=len(inp.names),matrix_nnz=z.nnz,matrix_density=z.nnz/(z.shape[0]*z.shape[1]),
                fit_seconds=0.,peak_process_rss_bytes=0,resources=storage(z),reused=False))
        for i,spec in enumerate(c['references']+c['presets']):
            rp,pp=run.paths(c,'validation',m,spec['id']);p=np.linspace(.01,.99,len(ev)) if i%2 else np.linspace(.9,.1,len(ev))
            np.savez_compressed(pp,row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=p)
            write_json(rp,dict(candidate=spec['id'],fingerprint=fp,prediction_sha256=digest(pp)))
        for spec in scores.strategies():
            if spec['kind']!='native':run.derived(c,fp,spec,m)
    generated,main,pairs=run.tables(c,run.crossfit(c),frame,x);out=Path(c['output_dir'])
    for name,t in generated.items():t.to_csv(out/(name+'.csv'),index=False)
    write_json(out/'feature_blocks.json',main);write_json(out/'interaction_blocks.json',pairs)
    return c,fp,frame,x

def test_complete_analytical_export_and_independent_audit(analytical_grid):
    c,fp,frame,x=analytical_grid
    result=report.audit_tables(c,frame,x,fp)
    assert result['scores']==132 and result['operating_rows']==240 and result['contrasts']==1199
    assert result['feature_rows']==522 and result['pair_rows']==18

@pytest.mark.parametrize('table,column',[tuple(s.split(':')) for s in ['crossfit:f2','crossfit:parameters','summary:f2','pooled:tp',
    'contrasts:reference','score_diagnostics:score_mean','clipping:changed_count','diversity:fraud_both',
    'coefficients:coefficient','solver:iterations','feature_diagnostics:missing','pair_diagnostics:n']])
def test_independent_audit_rejects_corrupted_export(analytical_grid,table,column):
    c,fp,frame,x=analytical_grid;p=Path(c['output_dir'])/(table+'.csv');original=p.read_bytes()
    t=pd.read_csv(p);t.loc[0,column]='{}' if column=='parameters' else 'invalid' if column=='reference' else t.loc[0,column]+1
    try:
        t.to_csv(p,index=False)
        with pytest.raises((ValueError,KeyError)):report.audit_tables(c,frame,x,fp)
    finally:p.write_bytes(original)

@pytest.mark.parametrize('winner',[scores.FUSION,scores.ANCHORS['interaction'],'logit_baseline_lgbm',scores.ALIAS,scores.strategy_id('latest','unweighted',.5)])
def test_paired_uncertainty_retains_every_role_and_alias(winner):
    from baf.direct_ranking_report import development,intervals
    from baf.score_fusion_report import audit_intervals
    roles=scores.choices_for(winner,'pooled');h={}
    for n,k in roles.values():h[n]={m:data([.1,.9,.2,.8,.3,.7],m*6) for m in [4,5,6]}
    rows=[]
    for m in [5,6]:
        for n,k in set(map(tuple,roles.values())):
            rule=run.learn({t:d for t,d in h[n].items() if t<m},k,m)
            rows.append(dict(candidate=n,policy=k,eval_month=m,parameters=json.dumps(rule),**run.measure(h[n][m],rule)))
    dint,ddraws=development(dict(bootstrap_seed=1027,bootstrap_replicates=5),{'choices':roles},pd.DataFrame(rows),h)
    final={a:data([.1,.9,.2,.8,.3,.7]) for a in roles};masks={a:report.independent_mask(final[a],report.independent_rule(h[n],k,7)) for a,(n,k) in roles.items()}
    fint,fdraws=intervals(final,masks,np.random.default_rng(1027),5,'7_consumed')
    audit_intervals(dint,ddraws,fint,fdraws,roles,5)
    assert len(dint)==3*(len(roles)-1) and len(fint)==len(roles)-1
    assert len(ddraws)==10 and len(fdraws)==5
    np.testing.assert_array_equal(fdraws['same_strategy'],0.)

@pytest.mark.parametrize('name',[scores.FUSION,scores.ANCHORS['interaction'],'logit_baseline_lgbm',scores.ALIAS,scores.strategy_id('latest','unweighted',.5)])
def test_exact_final_type_bounds_and_score_dag(name):
    c=read_config(run.CONFIG);roles=scores.choices_for(name,'pooled')
    with patch.object(run.tree,'cache_state',return_value=False):bounds=run.final_fit_bounds(c,roles)
    specs=run.required_final_specs(c,roles)
    assert bounds['unique_base_results']==len(specs)<=6
    assert bounds['max_new_interaction_fits']==sum(s.get('representation')=='interaction' for s in specs)<=1
    assert bounds['max_reproduced_additive_fits']==2 and bounds['max_reproduced_tree_fits']==3
    assert all(v is None for v in bounds['original_availability'].values())
    assert set(run.dependencies([n for n,k in roles.values()])['bases'])=={s['id'] for s in specs}


def test_current_parent_precedes_declared_individual_older_origins(tmp_path):
    c,fp,frame,x=tiny_context(tmp_path)
    for kind in ['interaction','additive','lightgbm','xgboost']:
        spec=source_spec(c,kind);origins=run.source_origins(c,spec)
        assert origins[0][0]['output_dir']==read_config(c['fusion_source_config'])['output_dir']
        assert origins[1:]==run.parent.source_origins(c,spec)


@pytest.mark.parametrize('where',['own','origin'])
def test_partial_derived_set_rejected_without_overwrite(tmp_path,where):
    c=read_config(run.CONFIG);c['output_dir']=str(tmp_path/'own');fp={'fixture':True}
    origin=tmp_path/'origin.json';write_json(origin,{**c,'output_dir':str(tmp_path/'origin')})
    c['fusion_source_config']=str(origin);c['fusion_source_fingerprint']={'origin':True}
    spec=scores.lookup(scores.FUSION)
    for n in spec['members']:
        rp,pp=run.paths(c,'validation',4,n);pp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(pp,**data([.1,.9]));write_json(rp,{'source':n})
    target=c if where=='own' else read_config(origin)
    rp,pp=run.paths(target,'validation',4,spec['id']);write_json(rp,{'partial':True});old=rp.read_bytes()
    with pytest.raises(ValueError,match='Partial'):run.derived(c,fp,spec,4)
    assert rp.read_bytes()==old and not pp.exists()


@pytest.mark.parametrize('name',[*scores.ANCHORS.values(),scores.ALIAS])
@pytest.mark.parametrize('corruption',[None,'identity','recipe','epsilon','prediction_hash','parent_hash','arithmetic'])
def test_checked_historical_derived_reuse_and_corruption(tmp_path,name,corruption):
    c=read_config(run.CONFIG);c['output_dir']=str(tmp_path/'own');fp={'fixture':True}
    key='source_config' if name==scores.ALIAS else 'fusion_source_config'
    identity_key='source_fingerprint' if name==scores.ALIAS else 'fusion_source_fingerprint'
    origin=read_config(c[key]);origin['output_dir']=str(tmp_path/'origin')
    source_path=tmp_path/'origin.json';write_json(source_path,origin)
    c[key]=str(source_path);c[identity_key]={'original':True}
    spec=scores.lookup(name)
    for member in spec['members']:
        d=data([.1,.9,.5] if member in scores.BASES else [-3.,4.,0.])
        for cfg in [c,origin]:
            rp,pp=run.paths(cfg,'validation',4,member);pp.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(pp,**d);write_json(rp,dict(candidate=member,prediction_sha256=digest(pp)))
    items={n:run.prediction(c,'validation',4,n) for n in spec['members']}
    composed=scores.compose(spec,items);sr,sp=run.paths(origin,'validation',4,name)
    np.savez_compressed(sp,**composed)
    old_spec=origin['fusion_spec'] if name==scores.FUSION else next(s for s in origin['strategies'] if s['id']==name)
    record=dict(fingerprint=c[identity_key],epsilon=1e-6,spec=old_spec,prediction_sha256=digest(sp),
        dependencies={n:{str(p):digest(p) for p in run.paths(origin,'validation',4,n)} for n in spec['members']},origin_artifacts={})
    if name in [scores.FUSION,scores.ALIAS]:
        record['dependencies']={n:{'record':digest(run.paths(origin,'validation',4,n)[0]),
            'prediction':digest(run.paths(origin,'validation',4,n)[1])} for n in spec['members']}
    if corruption=='identity':record['fingerprint']={}
    elif corruption=='recipe':record['spec']={}
    elif corruption=='epsilon':record['epsilon']=.01
    elif corruption=='prediction_hash':record['prediction_sha256']='changed'
    elif corruption=='parent_hash':
        parents=record['dependencies'][spec['members'][0]]
        parents[next(iter(parents))]='changed'
    elif corruption=='arithmetic':
        composed['scores'][0]+=.01;np.savez_compressed(sp,**composed);record['prediction_sha256']=digest(sp)
    write_json(sr,record)
    if corruption:
        with pytest.raises(ValueError):run.derived(c,fp,spec,4)
        assert not run.paths(c,'validation',4,name)[0].exists()
    else:
        r=run.derived(c,fp,spec,4)
        assert r['reused'] and r['origin_artifacts'][str(sr)]==digest(sr)
        np.testing.assert_array_equal(run.prediction(c,'validation',4,name)['scores'],composed['scores'])
        assert run.derived(c,fp,spec,4,allow_missing=False)==r


def test_source_miniature_gate_checks_two_frozen_ancestry_levels(tmp_path):
    c=read_config(run.CONFIG);c['output_dir']=str(tmp_path/'own')
    source=tmp_path/'source.json';parent_reports=tmp_path/'parent_reports'
    write_json(source,dict(report_dir=str(parent_reports)))
    c['fusion_source_config']=str(source);c['fusion_source_fingerprint']={'parent':True}
    c['interaction_source_fingerprint']={'interaction':True}
    model=tmp_path/'optional_model';mini_path=tmp_path/'miniature.json'
    write_json(mini_path,dict(fingerprint={'interaction':True},fit_count=8,
        results=[dict(artifact=str(model),sha256='declared') for _ in range(8)]))
    receipt=parent_reports/'source_miniature_gate.json'
    write_json(receipt,dict(fingerprint={'parent':True},preserved_fits=8,source=str(mini_path),sha256=digest(mini_path)))
    run.source_miniature_gate(c,{'own':True})
    actual=read_config(Path(c['output_dir'])/'source_miniature_gate.json')
    assert actual['preserved_fits']==8 and actual['new_production_miniature_fits']==0
    model.write_text('changed')
    with pytest.raises(ValueError,match='Preserved artifact'):run.source_miniature_gate(c,{'own':True})
    model.unlink();changed=read_config(receipt);changed['fingerprint']={};write_json(receipt,changed)
    with pytest.raises(ValueError,match='Parent miniature'):run.source_miniature_gate(c,{'own':True})


def test_resource_preflight_retains_failure_and_prevents_retry(tmp_path):
    c=read_config(run.CONFIG);c['output_dir']=str(tmp_path/'own');c['report_dir']=str(tmp_path/'report')
    fp={'fixture':True};out=Path(c['output_dir']);xml=Path(c['report_dir'])/'tests.xml'
    xml.parent.mkdir(parents=True);xml.write_text('<testsuites/>')
    write_json(out/'checks.json',dict(fingerprint=fp,junit_sha256=digest(xml),test_source_sha256=digest('tests/test_unweighted_fusion.py')))
    x,_=run.newton.miniature_frame();frame=x.assign(month=6,fraud_bool=0)
    with patch.object(run,'initialize',return_value=(c,fp)),patch.object(run,'source_miniature_gate'), \
         patch.object(run,'load_data',return_value=frame),patch.object(run,'features',return_value=x), \
         patch.object(run.sys,'platform','darwin'),patch.object(run.subprocess,'check_output',return_value='1'):
        with pytest.raises(ValueError,match='70%'):run.preflight()
        assert (out/'failures'/'resource-preflight.json').exists()
        with patch.object(run.InteractionInputs,'fit',side_effect=AssertionError('no retry')):
            with pytest.raises(ValueError,match='Prior resource failure'):run.preflight()


@pytest.mark.parametrize('spec',[s for s in scores.strategies()])
def test_independent_leaf_coefficients_and_probability_sum(spec):
    actual=scores.leaf_coefficients(spec['id'])
    assert actual==report.independent_leaf_coefficients(spec['id'])
    assert set(actual)<=set(scores.BASES)
    assert all(v>0 for v in actual.values()) and sum(actual.values())==pytest.approx(1.)


@pytest.mark.parametrize('name',[*scores.ANCHORS.values(),scores.ALIAS])
@pytest.mark.parametrize('month',[4,5,6])
def test_real_preserved_derived_dependency_schemas(name,month):
    c=read_config(run.CONFIG);spec=scores.lookup(name)
    source,identity=run.derived_origins(c,spec)[0]
    rp,pp=run.paths(source,'validation',month,name);record=read_config(rp)
    assert record['fingerprint']==identity
    checked=run.origin_dependency_hashes(source,spec,month,'validation',record)
    assert checked=={str(p):digest(p) for member in spec['members'] for p in run.paths(source,'validation',month,member)}
    corrupted=copy.deepcopy(record)
    member=spec['members'][0]
    corrupted['dependencies'][member]['unexpected-key']='not-an-artifact'
    with pytest.raises(ValueError,match='dependency schema'):
        run.origin_dependency_hashes(source,spec,month,'validation',corrupted)


@pytest.mark.parametrize('case',['active','archived','both','missing'])
def test_planning_evidence_resolution_survives_archive(tmp_path,case):
    active=tmp_path/'test-unweighted-reference-fusion-f2'/'planning-evidence.json'
    archived=tmp_path/'archive'/'2026-10-11-test-unweighted-reference-fusion-f2'/'planning-evidence.json'
    if case in ['active','both']:write_json(active,{'location':'active'})
    if case in ['archived','both']:write_json(archived,{'location':'archived'})
    if case=='missing':
        with pytest.raises(FileNotFoundError):planning_evidence_path(tmp_path)
    else:
        expected=active if case in ['active','both'] else archived
        assert planning_evidence_path(tmp_path)==expected
