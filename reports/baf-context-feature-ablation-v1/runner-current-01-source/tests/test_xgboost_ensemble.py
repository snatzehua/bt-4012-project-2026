"""Methodological contracts: label-free composition, chronology and cache identity."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score
from baf.common import read_config
from baf.rank_ensemble_scores import percentile,blend_scores,strategies,compose,choices_for
from baf.xgboost_inputs import tree_parameters,presets,fit_checked,integration_gate
from baf.xgboost_ensemble import CONFIG,guard,declared_strategies,fit_model,audit_model,model_paths,paths
from baf.direct_ranking_report import intervals,development


def data(scores,labels=None):
    scores=np.asarray(scores);return dict(scores=scores,labels=np.asarray(labels if labels is not None else np.arange(len(scores))%2),row_ids=np.arange(len(scores)))


def test_percentile_ties_and_invariance():
    x=np.array([3.,1.,1.,7.]);expected=np.array([.75,.375,.375,1.])
    np.testing.assert_array_equal(percentile(x),expected)
    np.testing.assert_array_equal(percentile(4*x+10),expected)
    order=np.array([3,0,2,1]);np.testing.assert_array_equal(percentile(x[order])[np.argsort(order)],expected)
    np.testing.assert_array_equal(percentile([5]),[1]);np.testing.assert_array_equal(percentile([4,4,4]),[2/3]*3)


@pytest.mark.parametrize('x',[[],[np.nan],[np.inf],[[1,2]]])
def test_percentile_invalid_batches(x):
    with pytest.raises(ValueError):percentile(x)


def test_composition_label_independence_and_alignment():
    spec=dict(id='example',kind='blend',members=['a','b'],alpha=.25)
    a=data([0,2,1,2],[0,1,0,1]);b=data([3,2,1,0],[0,1,0,1])
    actual=compose(spec,{'a':a,'b':b});np.testing.assert_allclose(actual['scores'],.25*percentile(a['scores'])+.75*percentile(b['scores']))
    a2={**a,'labels':1-a['labels']};b2={**b,'labels':1-b['labels']}
    np.testing.assert_array_equal(compose(spec,{'a':a2,'b':b2})['scores'],actual['scores'])
    with pytest.raises(ValueError):compose(spec,{'a':a,'b':{**b,'row_ids':b['row_ids'][::-1]}})
    with pytest.raises(ValueError):compose(spec,{'a':a,'b':b2})
    for alpha in [np.nan,-1,0,1,2]:
        with pytest.raises(ValueError):blend_scores(a['scores'],b['scores'],alpha)


def test_grid_and_required_controls():
    c=read_config(CONFIG);grid=declared_strategies(c)
    assert len(grid)==25 and len({x['id'] for x in grid})==25
    assert {k:sum(x['kind']==k for x in grid) for k in ['raw','percentile','blend']}==dict(raw=7,percentile=6,blend=12)
    for s in grid:
        choice=choices_for(s,'latest');assert 4<=len(choice)<=7
        assert choice['selected']==[s['id'],'latest'] and choice['same_model']==[s['id'],'pooled']
        assert choice['balanced']==['baseline_balanced','pooled'] and choice['unweighted']==['baseline_lgbm','pooled']
        if s['kind']=='blend':assert choice['xgb_member']==['pct_'+s['members'][0],'latest'] and choice['raw_xgb']==[s['members'][0],'latest']
        if s['kind']=='percentile':assert choice['raw_member']==[s['members'][0],'latest']
        if s['id']=='native31':assert len(choice)==4
    assert sum(len(grid) for _ in [4,5,6])==75


@pytest.mark.parametrize('mutation',['source','quota','grid','destination','data','reference'])
def test_guards_reject_changed_study(mutation):
    c=copy.deepcopy(read_config(CONFIG))
    if mutation=='source':c['source_fingerprint']['config']='wrong'
    elif mutation=='quota':c['beta']=1
    elif mutation=='grid':c['strategies'][-1]['alpha']=.8
    elif mutation=='destination':c['output_dir']='outputs/baf-study-v1/nested'
    elif mutation=='data':c['data_sha256']='wrong'
    else:c['references'][0]['weighted']=False
    with pytest.raises(ValueError):guard(c)


def test_frozen_prior_guard_and_pins():
    c=read_config(CONFIG);guard(c);assert len(c['preserve_sha256'])==19
    assert c['tree_params']==tree_parameters() and c['presets']==presets()


@pytest.fixture
def miniature(tmp_path):
    rng=np.random.default_rng(42);n=512
    frame=pd.DataFrame({'month':np.r_[np.zeros(480,dtype=int),np.ones(32,dtype=int)],'fraud_bool':np.arange(n)%3==0})
    base=pd.DataFrame({'payment_type':np.where(np.arange(n)%2,'AA','AB'),'value':rng.normal(size=n)})
    base.loc[480,'payment_type']='UNKNOWN';base.loc[481,'value']=np.nan
    c={**read_config(CONFIG),'output_dir':str(tmp_path),'tree_params':{**tree_parameters(),'n_estimators':12}}
    spec=presets()[0];identity={'miniature':True}
    r=fit_model(c,identity,frame,base,spec,1)
    return c,identity,frame,base,spec,r


def test_model_persistence_and_tampered_prediction(miniature):
    c,i,f,b,s,r=miniature;assert r['fitted_rounds']==12 and r['train_month_max']==0
    assert audit_model(c,i,f,b,s,1)==r
    rp,pp=paths(c,'validation',1,s['id'])
    with np.load(pp) as z:d={k:z[k] for k in z.files}
    d['scores'][0]+=.01;np.savez_compressed(pp,**d)
    with pytest.raises(ValueError,match='artifact changed'):audit_model(c,i,f,b,s,1)


def test_partial_cache_and_adapter_identity(miniature):
    c,i,f,b,s,r=miniature;mp,ap=model_paths(c,'validation',1,s['id']);ap.unlink()
    with pytest.raises(ValueError,match='Partial'):fit_model(c,i,f,b,s,1)


def test_installed_native_gate():
    rows=integration_gate(read_config(CONFIG));assert len(rows)==4 and sum(r['fits'] for r in rows)==8
    assert all(r['repeat_max_difference']==0 and r['reload_checked'] and r['unknown_as_missing'] for r in rows)


def test_binary_training_rejects_invalid_or_leaking_predictors():
    c=read_config(CONFIG);x=pd.DataFrame({'value':[1.,2.,3.]})
    with pytest.raises(ValueError):fit_checked(x,[0,0,0],presets()[0],c)
    with pytest.raises(ValueError):fit_checked(x.assign(month=[0,0,0]),[0,1,0],presets()[0],c)


def test_paired_intervals_independent_draws():
    y=np.array([1,0,1,0,0]);d=data(np.arange(5),y);masks={'selected':np.array([1,0,0,1,0],dtype=bool),'control':np.array([0,0,1,0,0],dtype=bool)}
    table,draws=intervals({'selected':d,'control':d},masks,np.random.default_rng(31),17,'test')
    rng=np.random.default_rng(31);expected=[]
    for _ in range(17):
        ids=rng.integers(0,len(y),len(y));expected.append(fbeta_score(y[ids],masks['selected'][ids],beta=2,zero_division=0)-fbeta_score(y[ids],masks['control'][ids],beta=2,zero_division=0))
    np.testing.assert_allclose(draws.control,expected,atol=1e-14)
    np.testing.assert_allclose(table[['ci_low','ci_high']].iloc[0],np.quantile(expected,[.025,.975]),atol=1e-14)


def test_composed_cache_reconstruction_and_tampering(tmp_path):
    from baf.xgboost_ensemble import ensure_score
    from baf.common import write_json
    c={'output_dir':str(tmp_path)};identity={'unit':1};member=data([.1,.4,.3,.4]);spec=dict(id='pct_a',kind='percentile',members=['a'])
    rp,pp=paths(c,'validation',4,'a');pp.parent.mkdir(parents=True);np.savez_compressed(pp,**member);write_json(rp,{'unit':'member'})
    record=ensure_score(c,identity,spec,4);assert ensure_score(c,identity,spec,4,allow_missing=False)==record
    rp,pp=paths(c,'validation',4,spec['id']);write_json(rp,{**record,'transform':'wrong'})
    with pytest.raises(ValueError,match='identity changed'):ensure_score(c,identity,spec,4)
    rp.unlink();pp.unlink()
    with pytest.raises(ValueError,match='missing'):ensure_score(c,identity,spec,4,allow_missing=False)


def test_reference_route_preserves_frozen_recipe_and_new_destination(tmp_path,monkeypatch):
    from baf.xgboost_ensemble import prepare_references
    import baf.focal_study as focal
    c={**read_config(CONFIG),'output_dir':str(tmp_path)};calls=[]
    def capture(config,*args,**kwargs):calls.append(config);return [{'reused':False}]
    monkeypatch.setattr(focal,'prepare_references',capture)
    assert prepare_references(c,{},None,None)==[{'reused':False}]
    assert calls[0]['output_dir']==str(tmp_path) and calls[0]['source_config']==c['reference_source_config']
