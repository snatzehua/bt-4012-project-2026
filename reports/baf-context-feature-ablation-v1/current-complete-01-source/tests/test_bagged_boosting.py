"""Analytical grid/corruption checks plus exactly two genuine tiny source fits."""
import copy,json
from itertools import product
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import pytest
from baf.common import read_config,write_json,digest
from baf import bagged_boosting as run
from baf import bagged_boosting_scores as scores
from baf import bagged_boosting_report as report


def planning_evidence():
    root=Path('openspec/changes');active=root/'test-bagged-lightgbm-seed-ensembles-f2/planning-evidence.json'
    matches=[active] if active.is_file() else list((root/'archive').glob('*-test-bagged-lightgbm-seed-ensembles-f2/planning-evidence.json'))
    if len(matches)!=1:raise ValueError('Exactly one planning evidence artifact required')
    return read_config(matches[0])


def data(p):
    return dict(row_ids=np.arange(len(p)),labels=np.arange(len(p))%2,scores=np.asarray(p,dtype=float))


def test_declared_counts_and_preserved_contract():
    c=read_config(run.CONFIG);e=planning_evidence();run.guard(c)
    assert len(run.SOURCES)==len(set(run.SOURCES))==52
    assert len(c['preserve_sha256'])==634 and len(scores.strategies())==15
    assert c['preserve_sha256']==e['mandatory_preservation_sha256']
    assert len(e['historical_ignored_file_sha256'])==3522
    assert len(scores.recipes())==6 and len(scores.contrast_factors())==342
    assert set(scores.contrast_factors())==report.expected_factors()
    assert len(report.expected_factors())==342


@pytest.mark.parametrize('name,policy',list(product([s['id'] for s in scores.strategies()],['pooled','latest','rank_fraction'])))
def test_every_winner_policy(name,policy):
    e=next(x for x in planning_evidence()['winner_policy_cases'] if x['winner']==name and x['policy']==policy)
    choices=scores.choices_for(name,policy);dag=scores.dependencies([n for n,k in choices.values()])
    assert choices==e['roles']
    assert dag['bases']==e['dependencies']['bases'] and dag['scores']==e['dependencies']['scores']
    assert dag['bagged_bases']==e['dependencies']['new_bagged_bases']
    assert len(choices)==(7 if name in scores.REFERENCE_POLICIES else 14)
    assert len(dag['bases'])<=11 and len(dag['scores'])<=17 and len(dag['bagged_bases'])<=6


@pytest.mark.parametrize('spec',[s for s in scores.strategies() if s['kind']=='aggregate'])
def test_fixed_aggregate_arithmetic_and_label_row_invariance(spec):
    arrays=[[0.,.1,.8,1.],[.2,.2,.7,.99],[.3,.3,.6,.98]]
    items={n:data(p) for n,p in zip(spec['members'],arrays)};actual=scores.compose(spec,items)
    values=[report.logit(x) if spec['form']=='logit' else np.array(x) for x in arrays]
    assert np.allclose(actual['scores'],(values[0]+values[1]+values[2])/3.,atol=1e-12,rtol=0)
    changed=copy.deepcopy(items)
    for d in changed.values():d['labels']=1-d['labels'];d['scores'][-1]=.5
    assert np.array_equal(actual['scores'][:-1],scores.compose(spec,changed)['scores'][:-1])
    singleton={n:dict(row_ids=d['row_ids'][:1],labels=d['labels'][:1],scores=d['scores'][:1]) for n,d in items.items()}
    assert actual['scores'][0]==scores.compose(spec,singleton)['scores'][0]
    permuted={n:items[n] for n in reversed(spec['members'])}
    assert np.array_equal(actual['scores'],scores.compose(spec,permuted)['scores'])


@pytest.mark.parametrize('bad',['missing','extra','labels','ids','shape','nan','inf','negative','above_one','spec','empty'])
def test_invalid_aggregates_rejected(bad):
    spec=next(s for s in scores.strategies() if s['kind']=='aggregate');items={n:data([.2,.8]) for n in spec['members']};n=spec['members'][0]
    if bad=='missing':items.pop(n)
    elif bad=='extra':items['undeclared']=data([.2,.8])
    elif bad=='labels':items[n]['labels']=1-items[n]['labels']
    elif bad=='ids':items[n]['row_ids']=np.array([1,0])
    elif bad=='shape':items[n]['scores']=np.array([[.2,.8]])
    elif bad=='nan':items[n]['scores'][0]=np.nan
    elif bad=='inf':items[n]['scores'][0]=np.inf
    elif bad=='negative':items[n]['scores'][0]=-.01
    elif bad=='above_one':items[n]['scores'][0]=1.01
    elif bad=='spec':spec={**spec,'members':list(reversed(spec['members']))}
    else:items[n]=data([])
    with pytest.raises(ValueError):scores.compose(spec,items)


@pytest.mark.parametrize('spec',scores.recipes())
def test_sampling_factory_changes_only_effective_declared_parameters(spec):
    c=read_config(run.CONFIG);_,base=run.synthetic_data(300)
    pipeline=run.factory(c,spec,base.columns);p=pipeline.named_steps['model'].get_params()
    assert p['subsample']==.8 and p['subsample_freq']==1 and p['bagging_seed']==spec['bagging_seed']
    assert p['random_state']==2026 and p['n_estimators']==200 and p['num_leaves']==15 and p['colsample_bytree']==1.
    assert p['class_weight']==('balanced' if spec['weighting']=='balanced' else None)
    assert 'bagging_fraction' not in p and 'bagging_freq' not in p


@pytest.mark.parametrize('weighting',scores.WEIGHTINGS)
def test_new_bagged_cache_format(tmp_path,weighting):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path),threads=1);fp={'fixture':'tiny-exact-format'}
    frame,base=run.synthetic_data();spec=next(s for s in scores.recipes() if s['weighting']==weighting)
    record=run.fit_bagged(c,fp,frame,base,spec,2)
    assert record['fitted_tree_count']<=200 and record['class_weights']['1']>=1
    ps=run.bagged_paths(c,'validation',2,spec['id']);original=ps[0].read_bytes();model_bytes=ps[2].read_bytes()
    with patch('lightgbm.LGBMClassifier.fit',side_effect=AssertionError('Unexpected refit')):
        assert run.fit_bagged(c,fp,frame,base,spec,2)==record
    for key in ['train_rows','train_fraud','train_month_max']:
        broken=copy.deepcopy(record);broken[key]+=1;write_json(ps[0],broken)
        with pytest.raises(ValueError):run.audit_bagged(c,fp,frame,base,spec,2)
        ps[0].write_bytes(original)
    for key in ['fingerprint','preprocessing','booster_params','class_weights']:
        broken=copy.deepcopy(record);broken[key]={'corrupt':True};write_json(ps[0],broken)
        with pytest.raises(ValueError):run.audit_bagged(c,fp,frame,base,spec,2)
        ps[0].write_bytes(original)
    artifact=joblib.load(ps[2]);artifact['model'].named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute'].statistics_[0]+=1
    joblib.dump(artifact,ps[2]);broken=copy.deepcopy(record);broken['model_sha256']=digest(ps[2]);write_json(ps[0],broken)
    with pytest.raises(ValueError,match='numeric statistics'):run.audit_bagged(c,fp,frame,base,spec,2)
    ps[2].write_bytes(model_bytes);ps[0].write_bytes(original)
    artifact=joblib.load(ps[2]);artifact['model'].named_steps['model'].booster_.params['subsample']=1.
    joblib.dump(artifact,ps[2]);broken=copy.deepcopy(record);broken['model_sha256']=digest(ps[2]);broken['booster_params']['subsample']=1.;write_json(ps[0],broken)
    with pytest.raises(ValueError,match='booster recipe'):run.audit_bagged(c,fp,frame,base,spec,2)
    ps[2].write_bytes(model_bytes);ps[0].write_bytes(original)
    ps[1].unlink()
    with patch('lightgbm.LGBMClassifier.fit',side_effect=AssertionError('Unexpected fit')):
        with pytest.raises(ValueError,match='Partial'):run.fit_bagged(c,fp,frame,base,spec,2)


def test_bagged_failure_evidence_blocks_retry(tmp_path):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path));frame,base=run.synthetic_data(300);spec=scores.recipes()[0]
    with patch.object(run,'factory',side_effect=MemoryError('fixture allocation gate')):
        with pytest.raises(MemoryError):run.fit_bagged(c,{},frame,base,spec,2)
    failure=tmp_path/'failures'/f'validation-m2-{spec["id"]}.json'
    assert failure.is_file() and 'allocation gate' in read_config(failure)['error']
    with patch.object(run,'factory',side_effect=AssertionError('Unexpected retry')):
        with pytest.raises(ValueError,match='Prior failed'):run.fit_bagged(c,{},frame,base,spec,2)


@pytest.mark.parametrize('where',['own','first_origin'])
def test_partial_reference_never_falls_back(tmp_path,where):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path/'own'));spec=c['references'][0];frame,base=run.synthetic_data(300)
    source=dict(c,output_dir=str(tmp_path/'origin'));target=c if where=='own' else source
    ps=run.previous.cache_paths(target,spec,2,'validation');write_json(ps[0],{})
    with patch.object(run,'source_origins',return_value=[(source,{})]),patch.object(run.previous,'audit_base',side_effect=AssertionError('Partial origin audited')),patch.object(run.previous.additive,'reproduce_reference',side_effect=AssertionError('Unexpected fit')):
        with pytest.raises(ValueError,match='Partial'):run.prepare_reference(c,{},frame,base,spec,2)


@pytest.mark.parametrize('kind',['spline','xgboost','baseline'])
def test_absent_reference_delegates_exact_individual_recipe(tmp_path,kind):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path));frame,base=run.synthetic_data(300)
    spec=next(s for s in run.reference_specs(c) if (s.get('representation')=='spline' if kind=='spline' else s['id']=='xgb_d4_w1' if kind=='xgboost' else s['id']=='baseline_lgbm'))
    target,method=(run.previous.newton,'fit_new') if kind=='spline' else (run.previous.tree.old,'fit_model') if kind=='xgboost' else (run.previous.additive,'reproduce_reference')
    with patch.object(run,'source_origins',return_value=[]),patch.object(target,method,return_value=None) as fit,patch.object(run.previous,'audit_base',return_value={'checked':True}) as audit:
        assert run.prepare_reference(c,{},frame,base,spec,2)=={'checked':True}
        fit.assert_called_once_with(c,{},frame,base,spec,2,'validation');audit.assert_called_once_with(c,{},frame,base,spec,2,'validation')


def test_constant_correlations_and_complete_overlap():
    a,b=data([.5]*4),data([.1,.2,.3,.4]);r=scores.correlations(a['scores'],b['scores'])
    assert r['pearson'] is r['spearman'] is None and r['correlation_reason']=='constant member'
    r=scores.overlap(a,b,[1,1,0,0],[1,0,1,0])
    assert [r[x] for x in ['both','first_only','second_only','neither']]==[1]*4
    assert sum(v for k,v in r.items() if k.startswith('fraud_'))==2


@pytest.fixture
def analytical_grid(tmp_path):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path));fp={'analytical':'no estimators fitted'};rng=np.random.default_rng(92)
    n=42;frame=pd.DataFrame({'month':np.repeat(np.arange(7),n),'fraud_bool':np.tile((np.arange(n)%7==0).astype(int),7)})
    names=[s['id'] for s in scores.strategies()];dag=scores.dependencies(names);bagged={s['id']:s for s in scores.recipes()}
    for m in [4,5,6]:
        ev=frame.index[frame.month==m].to_numpy();labels=frame.loc[ev,'fraud_bool'].to_numpy()
        for name in dag['bases']:
            values=np.clip(.03+.35*labels+rng.uniform(0,.5,n),.001,.999);rp,pp=run.paths(c,'validation',m,name);pp.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(pp,row_ids=ev,labels=labels,scores=values)
            if name in bagged:
                s=bagged[name];record=dict(candidate=name,spec=s,preprocessing={'transformed_dimension':42},
                    class_weights={'0':1.,'1':3.},warnings=[],requested_model_params={'bagging_seed':str(s['bagging_seed'])},
                    booster_params={'subsample':.8},fitted_tree_count=200,fit_seconds=2.2,peak_process_rss_bytes=12345,train_rows=m*n,train_fraud=m*6)
            else:record=dict(candidate=name,fixture='analytical reference evidence')
            write_json(rp,record)
        old=[s for s in run.previous.strategies() if s['id'] in dag['derived']]
        new=[s for s in scores.strategies() if s['kind']=='aggregate']
        for spec in old+new:
            name=spec['id'];items={n:run.prediction(c,'validation',m,n) for n in spec['members']}
            values=run.old_compose(spec,items) if spec in old else scores.compose(spec,items)
            rp,pp=run.paths(c,'validation',m,name);np.savez_compressed(pp,**values)
            deps={n:{str(p):digest(p) for p in run.paths(c,'validation',m,n)} for n in spec['members']}
            write_json(rp,dict(candidate=name,phase='validation',eval_month=m,fingerprint=fp,spec=spec,dependencies=deps,epsilon=1e-6,
                arithmetic='preserved reference' if spec in old else 'ordered float64 sum / 3',prediction_sha256=digest(pp),reused=False,origin_artifacts={}))
    cf=run.crossfit(c);tables,pre,refs=run.tables(c,cf)
    for name,t in tables.items():t.to_csv(tmp_path/(name+'.csv'),index=False)
    write_json(tmp_path/'preprocessing.json',pre);write_json(tmp_path/'reference_diagnostics.json',refs)
    return c,fp,frame


def test_complete_analytical_grid_and_independent_audit(analytical_grid):
    c,fp,frame=analytical_grid;r=report.audit_tables(c,fp,frame)
    assert r['scores']==54 and r['contrasts']==342 and r['operating_rows']==90


@pytest.mark.parametrize('table,column',[
    ('crossfit','f2'),('crossfit','tp'),('crossfit','flagged_fraction'),('summary','f2'),('summary','recall'),
    ('pooled','f2'),('pooled','tp'),('contrasts','f2_difference'),('contrasts','precision_difference'),
    ('score_diagnostics','score_mean'),('score_diagnostics','unique_scores'),('clipping','changed_count'),
    ('seed_diversity','pearson'),('seed_diversity','unequal_scores'),('decision_overlap','fraud_both'),
    ('decision_overlap','first_only'),('model_diagnostics','bagging_seed'),('model_diagnostics','train_fraud'),
    ('model_diagnostics','fitted_tree_count'),('model_diagnostics','transformed_dimension')])
def test_independent_audit_rejects_numeric_corruption(analytical_grid,table,column):
    c,fp,frame=analytical_grid;p=Path(c['output_dir'])/(table+'.csv');t=pd.read_csv(p);t.loc[0,column]+=1;t.to_csv(p,index=False)
    with pytest.raises(ValueError):report.audit_tables(c,fp,frame)


@pytest.mark.parametrize('table',[n[:-4] for n in run.TABLES if n.endswith('.csv')])
@pytest.mark.parametrize('change',['missing','duplicate'])
def test_independent_audit_checks_exact_table_keys(analytical_grid,table,change):
    c,fp,frame=analytical_grid;p=Path(c['output_dir'])/(table+'.csv');t=pd.read_csv(p)
    t=t.iloc[1:] if change=='missing' else pd.concat([t,t.iloc[:1]],ignore_index=True);t.to_csv(p,index=False)
    with pytest.raises(ValueError):report.audit_tables(c,fp,frame)


@pytest.mark.parametrize('field',['candidate','fingerprint','spec','dependencies','prediction_sha256'])
def test_independent_composition_provenance_corruption(analytical_grid,field):
    c,fp,frame=analytical_grid;name='bagged_balanced_logit_mean';p=run.paths(c,'validation',4,name)[0];r=read_config(p);r[field]={'corrupt':True};write_json(p,r)
    with pytest.raises(ValueError):report.audit_tables(c,fp,frame)


@pytest.mark.parametrize('filename',['preprocessing.json','reference_diagnostics.json'])
def test_independent_json_evidence_corruption(analytical_grid,filename):
    c,fp,frame=analytical_grid;p=Path(c['output_dir'])/filename;r=read_config(p);r.pop(next(iter(r)));write_json(p,r)
    with pytest.raises(ValueError):report.audit_tables(c,fp,frame)


@pytest.mark.parametrize('name',['bagged_balanced_prob_mean','logit_mean_xgb_d4_w1_baseline_balanced_a50'])
def test_partial_derived_cache_is_not_overwritten(analytical_grid,name):
    c,fp,frame=analytical_grid;rp,pp=run.paths(c,'validation',4,name);before=rp.read_bytes();pp.unlink()
    with pytest.raises(ValueError,match='Partial'):run.derived(c,fp,name,4)
    assert rp.read_bytes()==before and not pp.exists()


@pytest.mark.parametrize('name',['bagged_balanced_prob_mean','logit_mean_xgb_d4_w1_baseline_balanced_a50'])
def test_wholly_absent_derived_score_rebuilds_exactly(analytical_grid,name):
    c,fp,frame=analytical_grid;rp,pp=run.paths(c,'validation',4,name);original=run.prediction(c,'validation',4,name)['scores'];rp.unlink();pp.unlink()
    source=Path(c['output_dir'])/'absent-parent.json';write_json(source,{'output_dir':str(Path(c['output_dir'])/'absent-parent')})
    c=dict(c,parent_config=str(source));r=run.derived(c,fp,name,4)
    assert not r['reused'] and np.array_equal(run.prediction(c,'validation',4,name)['scores'],original)


def test_partial_original_reference_margin_is_rejected(analytical_grid):
    c,fp,frame=analytical_grid;name='logit_mean_xgb_d4_w1_baseline_balanced_a50';rp,pp=run.paths(c,'validation',4,name);rp.unlink();pp.unlink()
    source=dict(c,output_dir=str(Path(c['output_dir'])/'partial-parent'));path=Path(c['output_dir'])/'partial-parent.json';write_json(path,source)
    sr,sp=run.paths(source,'validation',4,name);write_json(sr,{'partial':True})
    with pytest.raises(ValueError,match='Partial'):run.derived(dict(c,parent_config=str(path)),fp,name,4)
    assert not rp.exists() and not pp.exists()


@pytest.mark.parametrize('winner',['baseline_balanced','bagged_balanced_prob_mean'])
def test_conditional_draws_keep_all_roles_and_aliases(analytical_grid,winner):
    c,fp,frame=analytical_grid;h=run.history(c);cf=pd.read_csv(Path(c['output_dir'])/'crossfit.csv');choices=scores.choices_for(winner,'pooled');s={'choices':choices}
    dint,draws=report.development({**c,'bootstrap_replicates':5},s,cf,h)
    data={a:h[n][6] for a,(n,k) in choices.items()};masks={a:report.independent_mask(d,report.independent_rule(h[choices[a][0]],choices[a][1],6)) for a,d in data.items()}
    fint,fdraws=report.intervals(data,masks,np.random.default_rng(1027),5,'7_consumed')
    report.audit_intervals(dint,draws,fint,fdraws,choices,5)
    assert len(dint)==3*(len(choices)-1) and len(fint)==len(choices)-1
    if winner=='baseline_balanced':assert np.array_equal(fdraws['balanced'],np.zeros(5))
