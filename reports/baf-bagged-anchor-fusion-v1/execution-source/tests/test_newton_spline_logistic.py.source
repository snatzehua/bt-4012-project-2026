"""Independent numerical gates, preserved studies and Newton cache integration."""
import copy
import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn._loss.loss import HalfBinomialLoss
from baf import newton_spline_study as study
from baf.common import read_config,write_json,digest
from baf.spline_inputs import AdditiveInputs,recipes,checked_scores
from baf.improvements import paths
from test_spline_logistic import reference_fixture


@pytest.mark.parametrize('weighted',[False,True])
@pytest.mark.parametrize('as_sparse',[False,True])
def test_objective_gradient_matches_finite_differences_and_pinned_loss(weighted,as_sparse):
    x=np.array([[1.,2.],[-.3,.6],[.2,-1.],[3.,.2],[1.,-.8]])
    y=np.array([0.,0.,1.,0.,1.]);x=sparse.csr_matrix(x) if as_sparse else x
    spec=dict(C=.1,class_weight='balanced' if weighted else None)
    model=SimpleNamespace(coef_=np.array([[.13,-.31]]),intercept_=np.array([.25]))
    actual=study.objective_gradient(model,x,y,spec);v=np.r_[model.coef_[0],model.intercept_[0]]
    numerical=[]
    for k in range(3):
        delta=np.zeros(3);delta[k]=1e-6
        def loss(a):return study.objective_gradient(SimpleNamespace(coef_=a[:2][None,:],intercept_=a[2:]),x,y,spec)['objective']
        numerical.append((loss(v+delta)-loss(v-delta))/(2e-6))
    np.testing.assert_allclose(actual['gradient'],numerical,atol=1e-9,rtol=1e-7)
    w=len(y)/(2*np.bincount(y.astype(int))[y.astype(int)]) if weighted else np.ones(len(y))
    expected,grad=LinearModelLoss(HalfBinomialLoss(),fit_intercept=True).loss_gradient(v,x,y,sample_weight=w,l2_reg_strength=1/(spec['C']*sum(w)))
    np.testing.assert_allclose(actual['objective'],expected,atol=1e-14)
    np.testing.assert_allclose(actual['gradient'],grad,atol=1e-14)
    if weighted:np.testing.assert_array_equal(w,[5/6,5/6,5/4,5/6,5/4])


def test_observer_detects_real_fallback_entry_without_warning_and_restores():
    assert sys.getprofile() is None
    with study.observe_fallback() as observed:
        # Enter the actual pinned function; a deliberately incomplete solver
        # raises before it could solve or emit a warning.
        with pytest.raises(AttributeError):study.NewtonSolver.fallback_lbfgs_solve(SimpleNamespace(),None,None,None)
    assert observed['fallback_count']==1 and sys.getprofile() is None
    with pytest.raises(RuntimeError):
        with study.observe_fallback():raise RuntimeError('fixture')
    assert sys.getprofile() is None


def test_competing_profiler_is_not_replaced():
    def profiler(*args):pass
    sys.setprofile(profiler)
    try:
        with pytest.raises(ValueError,match='profiler'):
            with study.observe_fallback():pass
        assert sys.getprofile() is profiler
    finally:sys.setprofile(None)


@pytest.mark.parametrize('spec',recipes(),ids=lambda s:s['id'])
def test_newton_recipe_converges_and_persists(spec,tmp_path):
    x,y=study.miniature_frame();inp=AdditiveInputs(spec['representation']).fit(x);z=inp.transform(x)
    model,messages,num=study.fit_checked(z,y,spec,threads=1)
    assert not messages and num['fallback_count']==0 and num['gradient_max_abs']<=1.1e-6
    assert 0<model.n_iter_.max()<100 and model.solver=='newton-cholesky'
    mp=tmp_path/'model';joblib.dump((inp,model),mp);a,b=joblib.load(mp)
    np.testing.assert_allclose(checked_scores(model,z),checked_scores(b,a.transform(x)),rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('bad',['fallback','warning','gradient','iterations','identity'])
def test_gate_rejects_numerically_invalid_records(bad):
    spec=recipes()[0];x,y=study.miniature_frame();z=AdditiveInputs('linear').fit(x).transform(x)
    model,messages,num=study.fit_checked(z,y,spec,threads=1);computed=study.objective_gradient(model,z,y,spec)
    if bad=='fallback':num['fallback_count']=1
    elif bad=='warning':messages=['deliberate warning']
    elif bad=='gradient':computed['gradient_max_abs']=.01
    elif bad=='iterations':model.n_iter_=np.array([100])
    else:num['fallback_function_sha256']='changed'
    with pytest.raises(ValueError):study.check_numerical(model,messages,num,computed)


@pytest.mark.parametrize('change',['grid','quota','identity','destination','solver','gradient','optional'])
def test_prospective_contract_rejects_changes(change):
    c=read_config(study.CONFIG)
    if change=='grid':c['candidates'][0]['C']=7
    elif change=='quota':c['review_quota']=.01
    elif change=='identity':c['source_fingerprint']['config']='changed'
    elif change=='destination':c['output_dir']='outputs/baf-spline-logistic-v1'
    elif change=='solver':c['solver']['solver']='lbfgs'
    elif change=='gradient':c['gradient_limit']=.1
    else:c['stopped_optional_hashes']={}
    with pytest.raises(ValueError):study.guard(c)


def fixture(tmp_path):
    x,y=study.miniature_frame();frame=x.assign(month=np.where(np.arange(len(x))<120,3,4),fraud_bool=y)
    c=dict(output_dir=str(tmp_path),threads=1,seed=2026,solver=study.solver_parameters(recipes()[0]));identity={'fixture':True}
    spec=recipes()[0];study.fit_new(c,identity,frame,x,spec,4)
    rp,pp=paths(c,'validation',4,spec['id']);mp=study.model_path(c,'validation',4,spec['id'])
    return c,identity,frame,x,spec,rp,pp,mp


def test_checked_cache_is_reused_without_fit(tmp_path):
    c,identity,frame,x,spec,*_=fixture(tmp_path)
    with patch.object(study,'fit_checked',side_effect=AssertionError('must not refit')):
        r=study.fit_new(c,identity,frame,x,spec,4)
    assert r['numerical']['fallback_count']==0


@pytest.mark.parametrize('change',['partial','prediction','model','population','metadata','gradient','fallback'])
def test_invalid_cache_rejected(tmp_path,change):
    c,identity,frame,x,spec,rp,pp,mp=fixture(tmp_path)
    if change=='partial':mp.unlink()
    elif change=='prediction':pp.write_bytes(pp.read_bytes()+b'changed')
    elif change=='model':mp.write_bytes(mp.read_bytes()+b'changed')
    elif change=='population':frame.loc[120,'fraud_bool']=1-frame.loc[120,'fraud_bool']
    else:
        r=read_config(rp)
        if change=='metadata':r['feature_metadata']['medians']['income']+=1
        elif change=='gradient':r['numerical']['gradient'][0]+=.01
        else:r['numerical']['fallback_count']=1
        write_json(rp,r)
    with pytest.raises(ValueError):study.fit_new(c,identity,frame,x,spec,4)


def test_failed_fit_retains_numerical_evidence_and_no_valid_cache(tmp_path):
    x,y=study.miniature_frame();frame=x.assign(month=np.where(np.arange(len(x))<120,3,4),fraud_bool=y)
    c=dict(output_dir=str(tmp_path),threads=1,seed=2026);spec=recipes()[0]
    with patch.object(study,'fit_checked',side_effect=study.NumericalGate('fallback',{'fallback_count':1})):
        with pytest.raises(study.NumericalGate):study.fit_new(c,{},frame,x,spec,4)
    failure=read_config(next((tmp_path/'failures').glob('*.json')))
    assert failure['numerical']['fallback_count']==1 and failure['train_rows']==120
    assert not list((tmp_path/'models').glob('*'))


def test_reference_interface_preserves_origins(tmp_path,monkeypatch):
    c,identity,frame,x,spec,source,sr,sp,sm=reference_fixture(tmp_path,monkeypatch)
    before={str(p):digest(p) for p in [sr,sp,sm]}
    rows=study.reference_bases(c,identity,frame,x)
    assert rows[0]['origin_artifacts']==before
    assert all(digest(p)==h for p,h in before.items())
    sm.unlink()
    with pytest.raises(ValueError):study.reference_bases(c,identity,frame,x,allow_missing=False)


def test_optional_absence_is_allowed_but_captured_deletion_is_rejected(tmp_path):
    optional=tmp_path/'old-model';optional.write_bytes(b'preserved')
    expected={str(optional):digest(optional),str(tmp_path/'absent'):'a'*64}
    snapshot=tmp_path/'presence.json';mandatory={'fixed':'hash'}
    study.verify_stopped_presence(expected,snapshot,mandatory)
    write_json(snapshot,dict(mandatory=mandatory,optional_present={str(optional):digest(optional)},optional_absent=[str(tmp_path/'absent')]))
    study.verify_stopped_presence(expected,snapshot,mandatory)
    optional.unlink()
    with pytest.raises(ValueError,match='Captured'):study.verify_stopped_presence(expected,snapshot,mandatory)
    snapshot.unlink()
    study.verify_stopped_presence(expected,snapshot,mandatory)


def test_optional_modified_or_incomplete_presence_is_rejected(tmp_path):
    optional=tmp_path/'old-model';optional.write_bytes(b'preserved');expected={str(optional):digest(optional)}
    snapshot=tmp_path/'presence.json'
    write_json(snapshot,dict(mandatory={},optional_present={},optional_absent=[]))
    with pytest.raises(ValueError,match='incomplete'):study.verify_stopped_presence(expected,snapshot,{})
    snapshot.unlink();optional.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):study.verify_stopped_presence(expected,snapshot,{})


def test_sorted_summary_verifier_resets_index_without_changing_values(tmp_path):
    from baf import newton_spline_audit as audit
    table=pd.DataFrame(dict(candidate=['first','second'],f2=[.2,.4]),index=[7,3])
    p=tmp_path/'summary.csv';table.to_csv(p,index=False);before=digest(p)
    with pytest.raises(ValueError,match='Provenance'):study.verify_table(table,p)
    audit.verify_table(table.reset_index(drop=True),p)
    assert digest(p)==before


def test_analysis_repair_identity_retains_original_fit_sources():
    from baf import newton_spline_audit as audit
    original=study.fingerprint(study.CONFIG);repaired=audit.analysis_fingerprint(study.CONFIG)
    assert len(repaired['sources'])==len(original['sources'])+2
    assert all(repaired['sources'][p]==h for p,h in original['sources'].items())
    assert repaired['config']==original['config'] and repaired['software']==original['software']
