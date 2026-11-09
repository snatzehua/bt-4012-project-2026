"""Methodological checks for fixed fusion, chronology, reuse and provenance."""
import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score
from baf.common import read_config,write_json,digest
from baf import xgboost_ensemble as old
from baf import score_fusion_study as study
from baf.score_fusion import checked,log_odds,fuse,compose,clipping,choices_for
from baf.score_fusion_report import audit_tables
from baf.recall_weighted_metrics import decisions,learn
from baf.direct_ranking_report import intervals,development


def data(scores, labels=None, offset=0):
    x=np.asarray(scores,dtype=float)
    return dict(scores=x,labels=np.asarray(labels if labels is not None else np.arange(len(x))%2),
                row_ids=np.arange(len(x))+offset)


def test_hand_arithmetic_weight_orientation_and_endpoints():
    a=np.array([.25,.5,.75]);b=np.array([.75,.25,.5])
    np.testing.assert_allclose(log_odds(a),[-np.log(3),0,np.log(3)],atol=1e-15)
    np.testing.assert_allclose(fuse(a,b,.25,'raw_mean'),[.625,.3125,.5625])
    np.testing.assert_allclose(fuse(a,b,.25,'logit_mean'),.25*np.log(a/(1-a))+.75*np.log(b/(1-b)))
    for mode in ['raw_mean','logit_mean']:
        np.testing.assert_allclose(fuse(a,b,.25,mode),fuse(b,a,.75,mode))
    assert np.isfinite(log_odds([0,1])).all()
    np.testing.assert_allclose(log_odds([0,1]),np.log([1e-6/(1-1e-6),(1-1e-6)/1e-6]))


def test_clipping_boundaries_and_monotonicity():
    x=np.array([0,1e-7,1e-6,.01,.5,.99,1-1e-6,1-1e-7,1])
    z=log_odds(x);assert np.all(np.diff(z)>=0) and np.all(np.diff(z[2:7])>0)
    assert z[0]==z[1]==z[2] and z[-1]==z[-2]==z[-3]
    c=clipping(x);assert (c['below_epsilon'],c['above_one_minus_epsilon'],c['changed_count'])==(2,2,4)


@pytest.mark.parametrize('x',[[],[[.2,.4]],[np.nan],[np.inf],[-.01],[1.01]])
def test_invalid_base_scores(x):
    with pytest.raises(ValueError):checked(x)
    with pytest.raises(ValueError):log_odds(x)


@pytest.mark.parametrize('alpha',[0,1,-1,2,np.nan,np.inf])
def test_invalid_weights(alpha):
    with pytest.raises(ValueError):fuse([.2],[.4],alpha,'raw_mean')


def test_invalid_modes_epsilon_and_shape():
    with pytest.raises(ValueError):fuse([.2],[.4,.5],.5,'raw_mean')
    with pytest.raises(ValueError):fuse([.2],[.4],.5,'percentile')
    for e in [0,.5,1,-1,np.nan]:
        with pytest.raises(ValueError):log_odds([.2],e)


def test_label_batch_independence_and_alignment():
    spec=dict(id='example',kind='logit_mean',members=['a','b'],alpha=.25)
    a=data([.1,.2,.3,.4]);b=data([.4,.3,.2,.1]);actual=compose(spec,dict(a=a,b=b))
    changed={k:{**v,'labels':1-v['labels']} for k,v in dict(a=a,b=b).items()}
    np.testing.assert_array_equal(compose(spec,changed)['scores'],actual['scores'])
    one={k:{key:value[:1] for key,value in v.items()} for k,v in dict(a=a,b=b).items()}
    assert compose(spec,one)['scores'][0]==actual['scores'][0]
    with pytest.raises(ValueError):compose(spec,dict(a=a,b={**b,'row_ids':b['row_ids'][::-1]}))
    with pytest.raises(ValueError):compose(spec,dict(a=a,b=changed['b']))
    with pytest.raises(ValueError):compose(spec,dict(a=a))
    with pytest.raises(ValueError):compose({**spec,'members':['a','a']},dict(a=a))


def test_complete_grid_and_all_control_branches():
    c=read_config(study.CONFIG);grid=c['strategies']
    assert grid==study.declared_strategies(c) and len(grid)==len({s['id'] for s in grid})==61
    assert {k:sum(s['kind']==k for s in grid) for k in ['raw','logit','raw_mean','logit_mean']}==dict(raw=7,logit=6,raw_mean=24,logit_mean=24)
    names={s['id'] for s in grid}
    for s in grid:
        roles=choices_for(s,'latest');assert all(n in names for n,k in roles.values())
        assert roles['selected']==[s['id'],'latest'] and roles['same_model']==[s['id'],'pooled']
        assert roles['balanced']==['baseline_balanced','pooled'] and roles['unweighted']==['baseline_lgbm','pooled']
        expected=8 if s['kind']=='logit_mean' else 6 if s['kind']=='raw_mean' else 4 if s['id']=='native31' else 5
        assert len(roles)==expected
        if s['kind']=='logit_mean':
            assert roles['xgb_member']==['logit_'+s['members'][0],'latest']
            assert roles['raw_anchor']==[s['members'][1],'latest']
    assert len(study.SOURCES)==30 and len(c['preserve_sha256'])==22


@pytest.mark.parametrize('mutation',['source','beta','grid','epsilon','destination','overlap','data','reference','protection','quota'])
def test_guard_rejects_changed_contract(mutation):
    c=copy.deepcopy(read_config(study.CONFIG))
    if mutation=='source':c['source_fingerprint']['software']['numpy']='wrong'
    elif mutation=='beta':c['beta']=1
    elif mutation=='grid':c['strategies'][-1]['alpha']=.8
    elif mutation=='epsilon':c['epsilon']=1e-5
    elif mutation=='destination':c['output_dir']='outputs/baf-xgboost-rank-ensembles-v1/nested'
    elif mutation=='overlap':c['report_dir']=c['output_dir']+'/nested'
    elif mutation=='data':c['data_sha256']='wrong'
    elif mutation=='reference':c['references'][0]['weighted']=False
    elif mutation=='protection':c['preserve_sha256'].pop(next(iter(c['preserve_sha256'])))
    else:c['primary_fraction']=.01
    with pytest.raises(ValueError):study.guard(c)


def test_preserved_guard():
    study.guard(read_config(study.CONFIG))


@pytest.fixture
def miniature(tmp_path):
    n=512;rng=np.random.default_rng(42)
    frame=pd.DataFrame(dict(month=np.r_[np.zeros(480,dtype=int),np.ones(32,dtype=int)],fraud_bool=np.arange(n)%3==0))
    base=pd.DataFrame(dict(payment_type=np.where(np.arange(n)%2,'AA','AB'),value=rng.normal(size=n)))
    base.loc[480,'payment_type']='UNKNOWN';base.loc[481,'value']=np.nan
    source={**read_config(old.CONFIG),'output_dir':str(tmp_path/'source'),'tree_params':{**read_config(old.CONFIG)['tree_params'],'n_estimators':12}}
    sp=tmp_path/'source.json';write_json(sp,source);identity={'miniature':'source'};spec=source['presets'][0]
    old.fit_model(source,identity,frame,base,spec,1)
    c={**source,'output_dir':str(tmp_path/'new'),'source_config':str(sp),'source_fingerprint':identity}
    return c,{'miniature':'new'},frame,base,spec


def test_verified_model_copy_and_origin_hash(miniature):
    c,i,f,b,s=miniature;r=study.fit_model(c,i,f,b,s,1)
    assert r['reused'] and r['fingerprint']==i and len(r['origin_artifacts'])==4
    assert study.audit_model(c,i,f,b,s,1)==r
    origin=next(iter(r['origin_artifacts']));Path(origin).write_text('changed')
    with pytest.raises(ValueError,match='origin changed'):study.audit_model(c,i,f,b,s,1)


def test_absent_model_exact_refit_and_partial_rejection(miniature):
    c,i,f,b,s=miniature;source=read_config(c['source_config'])
    for p in [*study.paths(source,'validation',1,s['id']),*study.model_paths(source,'validation',1,s['id'])]:p.unlink()
    r=study.fit_model(c,i,f,b,s,1);assert not r['reused']
    study.model_paths(c,'validation',1,s['id'])[1].unlink()
    with pytest.raises(ValueError,match='Partial'):study.fit_model(c,i,f,b,s,1)


def test_partial_or_incompatible_source_rejected(miniature):
    c,i,f,b,s=miniature;source=read_config(c['source_config']);rp,pp=study.paths(source,'validation',1,s['id'])
    r=read_config(rp);r['fingerprint']={'wrong':True};write_json(rp,r)
    with pytest.raises(ValueError,match='identity'):study.fit_model(c,i,f,b,s,1)
    pp.unlink()
    with pytest.raises(ValueError,match='Partial'):study.fit_model(c,i,f,b,s,1)


def test_score_cache_dependencies_partial_and_tampering(tmp_path):
    c={**read_config(study.CONFIG),'output_dir':str(tmp_path)};i={'fixture':1}
    spec=next(s for s in c['strategies'] if s['kind']=='logit_mean')
    for j,n in enumerate(spec['members']):
        rp,pp=study.paths(c,'validation',4,n);pp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(pp,**data([.1,.2,.3] if j==0 else [.4,.5,.6]));write_json(rp,{'fixture':n})
    r=study.ensure_score(c,i,spec,4);assert r['epsilon']==1e-6 and len(r['dependencies'])==2
    assert study.ensure_score(c,i,spec,4,allow_missing=False)==r
    rp,pp=study.paths(c,'validation',4,spec['members'][0]);write_json(rp,{'tampered':True})
    with pytest.raises(ValueError,match='identity'):study.ensure_score(c,i,spec,4)
    study.paths(c,'validation',4,spec['id'])[1].unlink()
    with pytest.raises(ValueError,match='Partial'):study.ensure_score(c,i,spec,4)


def test_chronological_grid_tables_and_independent_audit(tmp_path,monkeypatch):
    c={**read_config(study.CONFIG),'output_dir':str(tmp_path)};rng=np.random.default_rng(9);h={s['id']:{} for s in c['strategies']}
    for m in [4,5,6]:
        base={s['id']:data(rng.uniform(0,1,16),offset=m*100) for s in c['strategies'] if s['kind']=='raw'}
        for s in c['strategies']:h[s['id']][m]=compose(s,base)
    monkeypatch.setattr(study,'history',lambda _:h)
    cf=study.crossfit(c);summary=study.export_comparisons(c,cf)
    cf.to_csv(tmp_path/'crossfit.csv',index=False);summary.to_csv(tmp_path/'summary.csv',index=False)
    assert len(cf)==366 and len(summary)==183
    audit_tables(c,cf,h)
    for r in cf.itertuples():
        rule=json.loads(r.parameters);assert max(rule['source_months'])<r.eval_month
        d=h[r.candidate][r.eval_month];mask=decisions(d,rule)
        assert np.isclose(r.f2,fbeta_score(d['labels'],mask,beta=2))
    t=pd.read_csv(tmp_path/'contrasts.csv');assert len(t)==866
    t.loc[0,'reference_policy']='latest';t.to_csv(tmp_path/'contrasts.csv',index=False)
    with pytest.raises(ValueError):audit_tables(c,cf,h)


def test_dynamic_role_bootstrap_pairing():
    d=data([.1,.2,.3,.4],[0,1,0,1]);masks={'selected':np.array([0,1,0,1],bool),'anchor_member':np.array([0,0,0,1],bool),'raw_anchor':np.array([0,1,0,0],bool)}
    table,draws=intervals({n:d for n in masks},masks,np.random.default_rng(42),4,'test')
    rng=np.random.default_rng(42)
    for rep in range(4):
        ix=rng.integers(0,4,4)
        values={n:fbeta_score(d['labels'][ix],m[ix],beta=2,zero_division=0) for n,m in masks.items()}
        for n in ['anchor_member','raw_anchor']:assert np.isclose(draws.loc[rep,n],values['selected']-values[n])
    for r in table.itertuples():
        np.testing.assert_allclose([r.ci_low,r.ci_high],np.quantile(draws[r.reference],[.025,.975]))


def test_reference_absent_model_reproduction_and_tampering(tmp_path):
    from baf.engineering import make_estimator
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(17);n=512
    f=pd.DataFrame(dict(month=np.r_[np.zeros(480,dtype=int),np.ones(32,dtype=int)],fraud_bool=np.arange(n)%3==0))
    b=pd.DataFrame(dict(value=rng.normal(size=n),other=rng.normal(size=n)))
    for name in ['payment_type','employment_status','housing_status','source','device_os']:
        b[name]=np.where(np.arange(n)%2,'AA','AB')
    c={**read_config(study.CONFIG),'output_dir':str(tmp_path)};spec=c['references'][0]
    model,adapter=make_estimator(spec,study.reference_config(c),b.iloc[:480])
    with threadpool_limits(limits=6):
        model.fit(b.iloc[:480],f.fraud_bool.iloc[:480]);p=model.predict_proba(b.iloc[480:])[:,1]
    rp,pp=study.paths(c,'validation',1,spec['id']);pp.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(pp,row_ids=np.arange(480,512),labels=f.fraud_bool.iloc[480:].to_numpy(),scores=p)
    record=dict(candidate=spec['id'],fingerprint={'fixture':1},spec=spec)
    write_json(rp,record)
    r=study.ensure_reference_model(c,record['fingerprint'],f,b,spec,1,'validation',record,True)
    assert r['reference_model_new_fit'] and r['reference_model_warnings']==[]
    assert study.ensure_reference_model(c,record['fingerprint'],f,b,spec,1,'validation',r,False)==r
    mp=study.reference_model_path(c,'validation',1,spec['id']);mp.write_bytes(b'changed')
    with pytest.raises(ValueError,match='model changed'):study.reference_inference(c,f,b,spec,1,'validation',r)


def test_missing_reference_model_is_not_silently_skipped(tmp_path):
    c={**read_config(study.CONFIG),'output_dir':str(tmp_path)};spec=c['references'][0]
    with pytest.raises(ValueError,match='missing at audit'):
        study.ensure_reference_model(c,{},None,None,spec,4,'validation',{},False)
    mp=study.reference_model_path(c,'validation',4,spec['id']);mp.parent.mkdir(parents=True);mp.write_bytes(b'partial')
    with pytest.raises(ValueError,match='Partial'):
        study.ensure_reference_model(c,{},None,None,spec,4,'validation',{},True)


def test_wholly_absent_reference_fits_prediction_and_model_once(tmp_path,monkeypatch):
    from sklearn.pipeline import Pipeline
    rng=np.random.default_rng(31);n=512
    f=pd.DataFrame(dict(month=np.r_[np.zeros(480,dtype=int),np.ones(32,dtype=int)],fraud_bool=np.arange(n)%3==0))
    b=pd.DataFrame(dict(value=rng.normal(size=n)))
    for name in ['payment_type','employment_status','housing_status','source','device_os']:
        b[name]=np.where(np.arange(n)%2,'AA','AB')
    ref=read_config(read_config(study.CONFIG)['reference_source_config'])
    for key in ['stack_config','previous_f2_config','prior_config']:
        prior=read_config(ref[key]);prior['output_dir']=str(tmp_path/key)
        path=tmp_path/(key+'.json');write_json(path,prior);ref[key]=str(path)
    path=tmp_path/'reference.json';write_json(path,ref)
    c={**read_config(study.CONFIG),'output_dir':str(tmp_path/'new'),'reference_source_config':str(path),'validation_months':[1]}
    calls=[];fit=Pipeline.fit
    def counted(self,*args,**kwargs):
        calls.append(1);return fit(self,*args,**kwargs)
    monkeypatch.setattr(Pipeline,'fit',counted)
    rows=study.prepare_references(c,{'fixture':1},f,b,specs=[c['references'][0]])
    assert len(calls)==1 and len(rows)==1 and not rows[0]['reused'] and rows[0]['reference_model_new_fit']
    study.prepare_references(c,{'fixture':1},f,b,specs=[c['references'][0]],allow_missing=False)
    assert len(calls)==1
