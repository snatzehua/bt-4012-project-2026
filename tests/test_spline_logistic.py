"""Methodological spline, chronology, solver and immutable-cache checks."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.interpolate import BSpline
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import warnings

from baf.common import read_config, write_json, digest
from baf.improvements import paths, prediction
from baf.spline_inputs import AdditiveInputs, SMOOTH, DISCRETE, NUMERIC, recipes, solver_parameters, fit_checked, checked_scores
from baf import spline_study as study
from baf.spline_report import direct_draw_audit
from baf.recall_weighted_metrics import learn, decisions
from baf.direct_ranking_report import intervals


def test_roles_and_finite_recipe_grid():
    assert len(SMOOTH)==17 and len(DISCRETE)==7 and len(set(NUMERIC))==24
    specs=recipes();assert len(specs)==12 and len({s['id'] for s in specs})==12
    assert {(s['representation'],s['C'],s['class_weight']) for s in specs}=={
        (r,C,w) for r in ['linear','spline'] for C in [.01,.1,1.] for w in [None,'balanced']}
    assert solver_parameters(specs[0])['max_iter']==2000
    assert 'penalty' not in solver_parameters(specs[0]) and 'n_jobs' not in solver_parameters(specs[0])


@pytest.mark.parametrize('rep',['linear','spline'])
def test_train_only_schema_missingness_serialization_and_batch_independence(rep,tmp_path):
    x,y=study.miniature_frame();inp=AdditiveInputs(rep).fit(x);before=copy.deepcopy(inp.metadata())
    future=x.iloc[:9].copy();future.loc[future.index[0],'income']=np.nan
    future.loc[future.index[1],'device_os']='UNSEEN';future.loc[future.index[2],'name_email_similarity']=1e8
    z=inp.transform(future);assert sparse.isspmatrix_csr(z) and np.isfinite(z.data).all()
    for i in range(len(future)):np.testing.assert_allclose(z[i].toarray(),inp.transform(future.iloc[[i]]).toarray(),rtol=0,atol=1e-12)
    perm=np.arange(len(future))[::-1];np.testing.assert_allclose(inp.transform(future.iloc[perm]).toarray(),z[perm].toarray())
    assert inp.metadata()==before
    mp=tmp_path/'inputs.joblib';joblib.dump(inp,mp);loaded=joblib.load(mp)
    assert loaded.metadata()==before;np.testing.assert_array_equal(loaded.transform(future).toarray(),z.toarray())
    assert len([n for n in inp.names if n.startswith('missing__')])==24
    idx=inp.names.index('missing__income');assert z[0,idx]>0
    cats=inp.diagnostics(future);assert next(d for d in cats if d['feature']=='device_os')['unseen_category']==1
    assert next(d for d in cats if d['feature']=='name_email_similarity')['above_training']==1


def test_observed_quantile_dedup_and_degenerate_fallbacks():
    x,_=study.miniature_frame();x.loc[:130,'prev_address_months_count']=np.nan
    inp=AdditiveInputs('spline').fit(x);b={r['feature']:r for r in inp.blocks}
    expected=np.unique(np.quantile(x.prev_address_months_count.dropna(),[0,.25,.5,.75,1],method='linear'))
    np.testing.assert_array_equal(b['prev_address_months_count']['knots'],expected)
    assert b['customer_age']['knots']==[20.,25.,40.]
    assert b['zip_count_4w']['mode']=='constant' and b['intended_balcon_amount']['mode']=='all_missing'
    assert b['intended_balcon_amount']['median']==0
    assert len(inp.splines['customer_age'].bsplines_[0].t)==9


def test_basis_agrees_with_direct_scipy_and_constant_extrapolation():
    x,_=study.miniature_frame();inp=AdditiveInputs('spline').fit(x);col='name_email_similarity'
    transformer=inp.splines[col];bs=transformer.bsplines_[0];knots=next(b for b in inp.blocks if b['feature']==col)['knots']
    values=np.array([knots[0]-50,knots[0],np.mean(knots),knots[-1],knots[-1]+50])
    actual=transformer.transform(pd.DataFrame({col:values})).toarray()
    manual=BSpline(bs.t,bs.c,bs.k)(np.clip(values,knots[0],knots[-1]))[:,:-1]
    np.testing.assert_allclose(actual,manual,atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(actual[0],actual[1]);np.testing.assert_array_equal(actual[-1],actual[-2])


@pytest.mark.parametrize('change',['excluded','missing','order','infinity','empty','extra'])
def test_invalid_feature_inputs_are_rejected(change):
    x,_=study.miniature_frame();inp=AdditiveInputs('spline').fit(x);z=x.iloc[:3].copy()
    if change=='excluded':z['fraud_bool']=0
    elif change=='missing':z=z.drop(columns='income')
    elif change=='order':z=z[z.columns[::-1]]
    elif change=='infinity':z.loc[z.index[0],'income']=np.inf
    elif change=='empty':z=z.iloc[:0]
    else:z['undeclared']=1
    with pytest.raises(ValueError):inp.transform(z)


def test_training_statistics_and_representation_are_unweighted_and_no_label_api():
    x,y=study.miniature_frame();a=AdditiveInputs('spline').fit(x);b=AdditiveInputs('spline').fit(x.copy())
    assert a.metadata()==b.metadata()
    with pytest.raises(TypeError):AdditiveInputs('spline').fit(x,y)
    x2=x.copy();x2['income']+=100
    assert AdditiveInputs('spline').fit(x2).metadata()!=a.metadata()


def test_numerical_gate_rejects_warnings_and_preserves_fixed_solver(monkeypatch):
    matrix=sparse.eye(4,format='csr');spec=recipes()[0];seen={}
    def bad_fit(self,x,y):
        seen.update(self.get_params());warnings.warn('deliberate nonconvergence',ConvergenceWarning)
        self.coef_=np.zeros((1,4));self.intercept_=np.zeros(1);self.classes_=np.array([0,1]);self.n_iter_=np.array([2000])
        return self
    monkeypatch.setattr(LogisticRegression,'fit',bad_fit)
    with pytest.raises(ValueError,match='convergence'):fit_checked(matrix,[0,1,0,1],spec)
    assert seen['solver']=='lbfgs' and seen['max_iter']==2000 and seen['C']==.01


@pytest.mark.parametrize('labels',[[0,0,0,0],[1,1,1,1],[0,1]])
def test_solver_population_rejection(labels):
    with pytest.raises(ValueError):fit_checked(sparse.eye(4,format='csr'),labels,recipes()[0])


def cache_fixture(tmp_path):
    x,y=study.miniature_frame();frame=x.copy();frame['month']=np.where(np.arange(len(x))<120,3,4);frame['fraud_bool']=y
    spec=recipes()[0];inp=AdditiveInputs('linear').fit(x.iloc[:120]);dim=len(inp.names)
    # A saved analytic zero-coefficient model checks artifacts without adding fits.
    model=LogisticRegression(**solver_parameters(spec));model.classes_=np.array([0,1]);model.n_features_in_=dim
    model.coef_=np.zeros((1,dim));model.intercept_=np.zeros(1);model.n_iter_=np.array([1])
    c=dict(output_dir=str(tmp_path),threads=1,seed=2026,solver=solver_parameters(spec));identity={'fixture':True}
    rp,pp=paths(c,'validation',4,spec['id']);mp=study.model_path(c,'validation',4,spec['id'])
    for p in [rp,pp,mp]:p.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model=model,inputs=inp,spec=spec),mp)
    np.savez_compressed(pp,row_ids=np.arange(120,160),labels=y[120:],scores=np.full(40,.5))
    r={**study.expected_record(frame,spec,4,'validation',identity),'warnings':[],'iterations':1,
       'feature_metadata':inp.metadata(),'prediction_sha256':digest(pp),'model_sha256':digest(mp)}
    write_json(rp,r);return c,identity,frame,x,spec,rp,pp,mp


def test_complete_model_cache_and_independent_margin(tmp_path):
    c,identity,frame,x,spec,rp,pp,mp=cache_fixture(tmp_path)
    r=study.audit_new(c,identity,frame,x,spec,4);assert r['iterations']==1
    a=joblib.load(mp);np.testing.assert_array_equal(checked_scores(a['model'],a['inputs'].transform(x.iloc[120:])),.5)
    with patch.object(study,'fit_checked',side_effect=AssertionError('must not refit')):
        assert study.fit_new(c,identity,frame,x,spec,4)==r


@pytest.mark.parametrize('change',['partial','prediction','model','identity','population','metadata','coefficient'])
def test_partial_incompatible_or_changed_models_rejected(tmp_path,change):
    c,identity,frame,x,spec,rp,pp,mp=cache_fixture(tmp_path)
    if change=='partial':mp.unlink()
    elif change=='prediction':pp.write_bytes(pp.read_bytes()+b'changed')
    elif change=='model':mp.write_bytes(mp.read_bytes()+b'changed')
    elif change=='identity':identity={'fixture':False}
    elif change=='population':frame.loc[120,'fraud_bool']=1-frame.loc[120,'fraud_bool']
    elif change=='metadata':
        r=read_config(rp);r['feature_metadata']['medians']['income']+=1;write_json(rp,r)
    else:
        a=joblib.load(mp);a['model'].coef_[0,0]=1;joblib.dump(a,mp)
        r=read_config(rp);r['model_sha256']=digest(mp);write_json(rp,r)
    with pytest.raises(ValueError):study.fit_new(c,identity,frame,x,spec,4)


def test_absent_cache_audit_never_fits(tmp_path):
    x,y=study.miniature_frame();frame=x.assign(month=np.where(np.arange(len(x))<120,3,4),fraud_bool=y)
    c=dict(output_dir=str(tmp_path),threads=1,seed=2026)
    with pytest.raises(ValueError,match='missing'):study.fit_new(c,{},frame,x,recipes()[0],4,allow_missing=False)


def test_reference_roles_and_matched_factor_grid():
    c=read_config(study.CONFIG);assert len(study.contrast_factors(c))==237
    from collections import Counter
    assert sorted(Counter(row[0] for row in study.contrast_factors(c)).values())==[18,18,30,36,135]
    for spec in c['candidates']:
        for policy in c['policies']:
            choices=study.choices_for(c,spec['id'],policy);assert len(choices)==7
            lookup={s['id']:s for s in c['candidates']}
            a=lookup[choices['matched_representation'][0]];b=lookup[choices['matched_weight'][0]]
            assert a['representation']!=spec['representation'] and a['C']==spec['C'] and a['class_weight']==spec['class_weight']
            assert b['representation']==spec['representation'] and b['C']==spec['C'] and b['class_weight']!=spec['class_weight']
    for name in study.REFERENCE_IDS:assert len(study.choices_for(c,name,'pooled'))==5


@pytest.mark.parametrize('change',['grid','quota','identity','destination','preservation','solver'])
def test_study_guard_rejects_changed_contracts(change,tmp_path):
    c=read_config(study.CONFIG)
    if change=='grid':c['candidates'][0]['C']=7
    elif change=='quota':c['review_quota']=.01
    elif change=='identity':c['source_fingerprint']['config']='changed'
    elif change=='destination':c['output_dir']='outputs/baf-xgboost-score-fusion-v1/new'
    elif change=='preservation':
        p=tmp_path/'protected';p.write_text('old');c['preserve_sha256'][str(p)]=digest(p);p.write_text('new')
    else:c['solver']['max_iter']=100
    with pytest.raises(ValueError):study.guard(c)


def test_chronological_policy_and_independent_paired_draws():
    data={n:dict(labels=np.array([0,1,0,1,0]),scores=np.array(v),row_ids=np.arange(5))
          for n,v in [('selected',[.1,.9,.2,.8,.5]),('control',[.9,.2,.1,.7,.5])]}
    rule=learn({4:data['selected']},'pooled',5);assert rule['source_months']==[4]
    with pytest.raises(ValueError):learn({5:data['selected']},'pooled',5)
    masks={n:decisions(d,rule) for n,d in data.items()}
    table,draws=intervals(data,masks,np.random.default_rng(1027),30,'fixture')
    direct_draw_audit(data,masks,np.random.default_rng(1027),draws)
    broken=draws.copy();broken.loc[0,'control']+=.1
    with pytest.raises(ValueError):direct_draw_audit(data,masks,np.random.default_rng(1027),broken)


def reference_fixture(tmp_path,monkeypatch):
    x,y=study.miniature_frame();frame=x.assign(month=np.where(np.arange(len(x))<120,3,4),fraud_bool=y)
    spec=dict(id='reference',family='baseline',model='lightgbm',engineered=False,weighted=False)
    source=dict(output_dir=str(tmp_path/'old'));source_path=tmp_path/'old-config.json';write_json(source_path,source)
    identity={'fixture':'new'};source_identity={'fixture':'old'}
    c=dict(output_dir=str(tmp_path/'new'),source_config=str(source_path),source_fingerprint=source_identity,
           references=[spec],presets=[],validation_months=[4],fusion_spec=dict(id='reference',kind='raw'))
    sr,sp=paths(source,'validation',4,spec['id']);sm=study.old.reference_model_path(source,'validation',4,spec['id'])
    for p in [sr,sp,sm]:p.parent.mkdir(parents=True,exist_ok=True)
    sm.write_bytes(b'checked model fixture')
    np.savez_compressed(sp,row_ids=np.arange(120,160),labels=y[120:],scores=np.full(40,.5))
    record={**study.expected_record(frame,spec,4,'validation',source_identity),
            'prediction_sha256':digest(sp),'reference_model_sha256':digest(sm),'reused':False}
    write_json(sr,record)
    def inference(config,frame,base,spec,month,phase,record):
        mp=study.old.reference_model_path(config,phase,month,spec['id'])
        if digest(mp)!=record['reference_model_sha256']:raise ValueError('Reference model changed')
        study.old.verify_origin(record);return record
    monkeypatch.setattr(study.old,'reference_inference',inference)
    return c,identity,frame,x,spec,source,sr,sp,sm


def test_reference_reuse_keeps_origins_and_rejects_changed_origin(tmp_path,monkeypatch):
    c,identity,frame,x,spec,source,sr,sp,sm=reference_fixture(tmp_path,monkeypatch)
    before={str(p):digest(p) for p in [sr,sp,sm]}
    rows=study.reference_bases(c,identity,frame,x);assert len(rows)==1 and rows[0]['reused']
    assert rows[0]['origin_artifacts']==before and all(digest(p)==h for p,h in before.items())
    sm.write_bytes(b'changed')
    with pytest.raises(ValueError,match='origin'):study.reference_bases(c,identity,frame,x,allow_missing=False)


@pytest.mark.parametrize('change',['source_partial','source_population','own_partial','own_prediction','own_identity'])
def test_reference_partial_incompatible_and_changed_cache_rejected(tmp_path,monkeypatch,change):
    c,identity,frame,x,spec,source,sr,sp,sm=reference_fixture(tmp_path,monkeypatch)
    if change=='source_partial':sm.unlink()
    elif change=='source_population':
        d=prediction(source,'validation',4,spec['id']);d['row_ids'][0]=999;np.savez_compressed(sp,**d)
        r=read_config(sr);r['prediction_sha256']=digest(sp);write_json(sr,r)
    else:
        study.reference_bases(c,identity,frame,x)
        rp,pp=paths(c,'validation',4,spec['id']);mp=study.old.reference_model_path(c,'validation',4,spec['id'])
        if change=='own_partial':mp.unlink()
        elif change=='own_prediction':pp.write_bytes(pp.read_bytes()+b'changed')
        else:
            r=read_config(rp);r['fingerprint']={};write_json(rp,r)
    with pytest.raises(ValueError):study.reference_bases(c,identity,frame,x)


def test_genuinely_absent_reference_calls_only_exact_recipe_destination(tmp_path,monkeypatch):
    c,identity,frame,x,spec,source,sr,sp,sm=reference_fixture(tmp_path,monkeypatch)
    for p in [sr,sp,sm]:p.unlink()
    called=[]
    def reproduce(config,fp,frame,base,phase,specs,allow_missing):
        assert config['output_dir']==c['output_dir'] and fp==identity and specs==[spec] and allow_missing
        called.append(spec['id']);rp,pp=paths(config,phase,4,spec['id']);mp=study.old.reference_model_path(config,phase,4,spec['id'])
        for p in [rp,pp,mp]:p.parent.mkdir(parents=True,exist_ok=True)
        mp.write_bytes(b'checked model fixture');np.savez_compressed(pp,row_ids=np.arange(120,160),labels=frame.fraud_bool.to_numpy()[120:],scores=np.full(40,.5))
        write_json(rp,{**study.expected_record(frame,spec,4,phase,fp),'reused':False,
                      'prediction_sha256':digest(pp),'reference_model_sha256':digest(mp)})
    monkeypatch.setattr(study.old,'prepare_references',reproduce)
    rows=study.reference_bases(c,identity,frame,x)
    assert called==['reference'] and not rows[0]['reused'] and not sr.exists() and not sp.exists() and not sm.exists()
