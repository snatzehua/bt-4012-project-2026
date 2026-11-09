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
from baf import interaction_fusion_study as run
from baf import interaction_fusion_report as report
from baf import interaction_fusion_scores as scores
from baf.interaction_inputs import InteractionInputs, storage
from baf.spline_inputs import AdditiveInputs


def data(values, offset=0):
    return dict(row_ids=np.arange(len(values))+offset, labels=np.arange(len(values))%2,
                scores=np.asarray(values,dtype=float))


def test_declared_counts_ancestry_and_independent_factors():
    c=read_config(run.CONFIG);run.guard(c)
    assert len(run.SOURCES)==len(set(run.SOURCES))==46
    assert len(c['preserve_sha256'])==451
    assert len(scores.strategies())==51
    assert set(scores.contrast_factors())==set(report.expected_factors())
    assert Counter(r[0] for r in scores.contrast_factors())=={
        'original operating reference':612,'policy minus own pooled':102,
        'blend minus native':96,'blend minus transformed':96,'blend minus tree':96,
        'interaction minus additive':72,'balanced minus unweighted':72,
        'larger C minus smaller C':72,'larger alpha minus smaller alpha':144,'transformed minus native':24}


@pytest.mark.parametrize('spec',[s for s in scores.strategies() if s['kind'] in ['logit','blend']])
def test_composition_arithmetic_domains_and_label_batch_invariance(spec):
    p=np.array([0.,1.,.2,.8,.5]);g=np.array([-100.,100.,-2.,3.,0.])
    items={spec['members'][0]:data(p),scores.FUSION:data(g)}
    actual=scores.compose(spec,items)
    clipped=np.maximum(1e-6,np.minimum(1-1e-6,p));u=np.log(clipped)-np.log(1-clipped)
    expected=u if spec['kind']=='logit' else spec['alpha']*u+(1-spec['alpha'])*g
    np.testing.assert_allclose(actual['scores'],expected,atol=1e-12,rtol=1e-12)
    changed=copy.deepcopy(items)
    for d in changed.values():
        d['labels']=1-d['labels'];d['scores'][1:]=.4
    assert scores.compose(spec,changed)['scores'][0]==actual['scores'][0]
    single={n:{k:v[:1] for k,v in d.items()} for n,d in items.items()}
    assert scores.compose(spec,single)['scores'][0]==actual['scores'][0]


@pytest.mark.parametrize('change',['probability','nonfinite','shape','ids','labels','weight','recipe'])
def test_invalid_compositions_rejected(change):
    spec=next(s for s in scores.strategies() if s['kind']=='blend');items={spec['members'][0]:data([.1,.9]),scores.FUSION:data([-5.,6.])}
    if change=='probability':items[spec['members'][0]]['scores'][0]=-.01
    elif change=='nonfinite':items[scores.FUSION]['scores'][0]=np.inf
    elif change=='shape':items[scores.FUSION]['scores']=np.array([[1.,2.]])
    elif change=='ids':items[scores.FUSION]['row_ids']=np.array([8,9])
    elif change=='labels':items[scores.FUSION]['labels']=np.array([1,0])
    elif change=='weight':spec={**spec,'alpha':1.}
    else:spec={**spec,'id':'renamed'}
    with pytest.raises(ValueError):scores.compose(spec,items)


@pytest.mark.parametrize('name',[s['id'] for s in scores.strategies()])
@pytest.mark.parametrize('policy',['pooled','latest','rank_fraction'])
def test_all_winner_policy_roles(name,policy):
    roles=scores.choices_for(name,policy);spec=scores.lookup(name)
    assert roles['recent_fusion']==[scores.RECENT_FUSION,'latest']
    assert roles['same_strategy']==[name,'pooled']
    if name in scores.ORIGINAL_POLICIES:
        assert len(roles)==6
    else:
        assert len(roles)==(14 if spec['kind']=='blend' else 11)
        member=spec['members'][0];source=next(s for s in run.recipes() if s['id']==member)
        assert roles['original_member']==[member,policy]
        assert roles['transformed_member']==['logit_'+member,policy]
        for role,factor in [('matched_representation','representation'),('matched_weight','class_weight'),('matched_C','C')]:
            target=scores.lookup(roles[role][0]);other=next(s for s in run.recipes() if s['id']==target['members'][0])
            assert roles[role][1]==policy and target['form']==spec['form']
            assert {k for k in ['representation','class_weight','C'] if source[k]!=other[k]}=={factor}
        if spec['kind']=='blend':
            assert {scores.lookup(n)['alpha'] for role,(n,k) in roles.items() if role.startswith('other_alpha_')}==set(scores.ALPHAS)-{spec['alpha']}
    dag=run.dependencies([n for n,k in roles.values()])
    assert set(dag['derived'])==set(dag['scores'])-set(dag['bases'])
    assert len(dag['bases'])<=8


def test_canonical_alias_single_strategy_and_reference_precedence():
    assert scores.strategy_id('spline_c0p1_unweighted',.25)==scores.RECENT_FUSION
    assert sum(s['id']==scores.RECENT_FUSION for s in scores.strategies())==1
    assert len(scores.choices_for(scores.RECENT_FUSION,'rank_fraction'))==6
    roles=scores.choices_for(scores.strategy_id('interaction_c0p1_unweighted',.25),'pooled')
    assert roles['matched_representation']==[scores.RECENT_FUSION,'pooled']
    assert roles['recent_fusion']==[scores.RECENT_FUSION,'latest']


def tiny_context(tmp_path):
    x,y=run.newton.miniature_frame()
    frame=x.assign(month=np.where(np.arange(len(x))<120,3,4),fraud_bool=y)
    c=read_config(run.CONFIG);c.update(output_dir=str(tmp_path/'own'),threads=1)
    for key,identity in [('interaction_source_config','interaction_source_fingerprint'),
                         ('newton_source_config','newton_fit_fingerprint'),
                         ('recent_source_config','recent_source_fingerprint'),('source_config','source_fingerprint')]:
        p=tmp_path/(key+'.json');write_json(p,{**c,'output_dir':str(tmp_path/key)})
        c[key]=str(p);c[identity]={'fixture_origin':key}
    return c,{'fixture':True},frame,x


@pytest.mark.parametrize('kind',['interaction','additive','lightgbm','xgboost'])
def test_four_missing_cache_formats_fit_once(tmp_path,kind):
    c,fp,frame,x=tiny_context(tmp_path)
    spec=(next(s for s in run.recipes() if s['representation']==('spline' if kind=='additive' else 'interaction')
               and s['C']==1. and s['class_weight'] is None) if kind in ['interaction','additive']
          else c['references'][0] if kind=='lightgbm' else c['presets'][0])
    r=run.prepare_base(c,fp,frame,x,spec,4)
    assert not r['reused'] and r['fingerprint']==fp
    assert all(p.is_file() for p in run.cache_paths(c,spec,4,'validation'))
    for key in ['interaction_source_config','newton_source_config','recent_source_config','source_config']:
        assert not Path(read_config(c[key])['output_dir']).exists()
    with patch.object(run.parent,'fit_interaction',side_effect=AssertionError('no refit')), \
         patch.object(run.newton,'fit_new',side_effect=AssertionError('no refit')), \
         patch.object(run.tree.old,'fit_model',side_effect=AssertionError('no refit')), \
         patch.object(run.additive,'reproduce_reference',side_effect=AssertionError('no refit')):
        again=run.prepare_base(c,fp,frame,x,spec,4)
    assert again['prediction_sha256']==r['prediction_sha256']


@pytest.mark.parametrize('kind',['interaction','additive','lightgbm','xgboost'])
@pytest.mark.parametrize('where',['own','first_origin','fallback_origin'])
def test_partial_required_cache_never_falls_back_or_fits(tmp_path,kind,where):
    c,fp,frame,x=tiny_context(tmp_path)
    spec=(run.recipes()[4] if kind=='interaction' else run.recipes()[0] if kind=='additive'
          else c['references'][0] if kind=='lightgbm' else c['presets'][0])
    origins=run.source_origins(c,spec)
    target=c if where=='own' else origins[0 if where=='first_origin' else -1][0]
    p=run.cache_paths(target,spec,4,'validation')[0];write_json(p,{'partial':True})
    with patch.object(run.parent,'fit_interaction',side_effect=AssertionError('no fit')), \
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


def test_preserved_miniature_gate_allows_absent_ignored_history_but_rejects_changed_present_model(tmp_path):
    fp={'own':True};source_fp={'source':True};report_dir=tmp_path/'source_reports'
    source=tmp_path/'source.json';write_json(source,dict(report_dir=str(report_dir)))
    model=tmp_path/'ignored_model';mini=dict(fingerprint=source_fp,fit_count=8,
        results=[dict(artifact=str(model),sha256='declared') for _ in range(8)])
    write_json(report_dir/'miniature.json',mini)
    c=dict(output_dir=str(tmp_path/'own'),interaction_source_config=str(source),interaction_source_fingerprint=source_fp)
    run.source_miniature_gate(c,fp)
    r=read_config(Path(c['output_dir'])/'source_miniature_gate.json')
    assert r['preserved_fits']==8 and r['new_production_miniature_fits']==0 and r['optional_source_artifacts_present']==0
    model.write_text('changed')
    with pytest.raises(ValueError,match='Preserved artifact'):run.source_miniature_gate(c,fp)


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
    origin=root/'absent_interaction.json';write_json(origin,{**c,'output_dir':str(root/'absent')})
    c['interaction_source_config']=str(origin);c['interaction_source_fingerprint']={'absent_fixture':True}
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
        run.tree.ensure_score(c,fp,c['fusion_spec'],m)
        for spec in scores.strategies():
            if spec['kind'] in ['logit','blend']:run.derived(c,fp,spec,m)
    generated,main,pairs=run.tables(c,run.crossfit(c),frame,x);out=Path(c['output_dir'])
    for name,t in generated.items():t.to_csv(out/(name+'.csv'),index=False)
    write_json(out/'feature_blocks.json',main);write_json(out/'interaction_blocks.json',pairs)
    return c,fp,frame,x


def test_complete_analytical_export_and_independent_audit(analytical_grid):
    c,fp,frame,x=analytical_grid
    result=report.audit_tables(c,frame,x,fp)
    assert result['scores']==156 and result['operating_rows']==306 and result['contrasts']==1386
    assert result['feature_rows']==1392 and result['pair_rows']==72


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


@pytest.mark.parametrize('winner',[scores.RECENT_FUSION,'logit_interaction_c1p0_balanced','interaction_blend_interaction_c1p0_balanced_a75'])
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


@pytest.mark.parametrize('name',[scores.RECENT_FUSION,'spline_c1p0_balanced','interaction_blend_interaction_c1p0_balanced_a75'])
def test_exact_final_type_bounds_and_score_dag(name):
    c=read_config(run.CONFIG);roles=scores.choices_for(name,'pooled')
    with patch.object(run.tree,'cache_state',return_value=False):bounds=run.final_fit_bounds(c,roles)
    specs=run.required_final_specs(c,roles)
    assert bounds['unique_base_results']==len(specs)<=8
    assert bounds['max_new_interaction_fits']==sum(s.get('representation')=='interaction' for s in specs)
    assert bounds['max_reproduced_additive_fits']==sum(s.get('representation')=='spline' for s in specs)
    assert bounds['max_reproduced_tree_fits']==3
    assert all(value is None for value in bounds['original_availability'].values())
    dag=run.dependencies([n for n,k in roles.values()])
    assert set(dag['bases'])=={s['id'] for s in specs}
    # Blends use raw members directly; only an explicitly selected logit role adds its artifact.
    assert run.dependencies([scores.strategy_id('interaction_c1p0_balanced',.75)])['derived']==[
        scores.strategy_id('interaction_c1p0_balanced',.75),scores.FUSION]
