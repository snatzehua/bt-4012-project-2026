"""Real native/source formats and independent chronological corruption checks."""
import copy
from itertools import product
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from baf import shallow_catboost as models
from baf import shallow_catboost_study as run
from baf import shallow_catboost_report as audit
from baf.common import read_config,write_json,digest,features
from baf.improvements import prediction
from baf.recall_weighted_metrics import POLICIES


@pytest.mark.parametrize('spec',models.recipes(),ids=lambda s:s['id'])
def test_twelve_native_formats_fit_audit_reuse_and_reject_corruption(tmp_path,spec):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path/'own'));fp={'fixture':'native tiny format'}
    columns=pd.read_csv(c['data_path'],nrows=0).drop(columns=['fraud_bool','month','device_fraud_count']).columns
    frame,base=run.miniature_data(columns);r=models.fit_model(c,fp,frame,base,spec,2);ps=models.model_paths(c,'validation',2,spec['id'])
    audit.audit_native(c,fp,frame,base,r)
    model=CatBoostClassifier();model.load_model(str(ps[2]));adapter=joblib.load(ps[3])['adapter'];x=base.loc[frame.month==2];saved=prediction(c,'validation',2,spec['id'])
    assert adapter.medians['prev_address_months_count']==0
    assert adapter.transform(x).loc[480,models.CATEGORICAL].eq('__MISSING__').all()
    assert adapter.transform(x).loc[481,models.CATEGORICAL].eq('unseen').all()
    checks=[models.margin(model,adapter,x.iloc[::-1])[::-1],np.concatenate([models.margin(model,adapter,x.iloc[:17]),models.margin(model,adapter,x.iloc[17:])]),
        models.margin(model,adapter,x.iloc[:1]),models.margin(model,adapter,pd.concat([x,x.iloc[:1].assign(name_email_similarity=999)]))[:len(x)]]
    for actual in checks:np.testing.assert_allclose(actual,saved['scores'][:len(actual)],atol=1e-12,rtol=0)
    changed=frame.copy();changed.loc[frame.month==2,'fraud_bool']=1-changed.loc[frame.month==2,'fraud_bool']
    np.testing.assert_allclose(models.margin(model,adapter,features(changed).loc[x.index]),saved['scores'],atol=1e-12,rtol=0)
    snapshot={p:p.read_bytes() for p in ps}
    with patch.object(CatBoostClassifier,'fit',side_effect=AssertionError('No refit')):
        assert models.fit_model(c,fp,frame,base,spec,2)==r
    for field,value in [('train_rows',r['train_rows']+1),('class_costs',{'0':1,'1':999}),('warnings',['unapproved warning']),('train_row_rounds',0)]:
        bad=copy.deepcopy(r);bad[field]=value;write_json(ps[0],bad)
        with pytest.raises(ValueError):models.audit_model(c,fp,frame,base,spec,2)
        ps[0].write_bytes(snapshot[ps[0]])
    detached=models.detached(r['effective_model_params']);detached['class_weights'][0]=999
    assert r['effective_model_params']['class_weights'][0]==1
    ps[2].write_bytes(snapshot[ps[2]]+b'corrupt')
    with pytest.raises(ValueError):models.audit_model(c,fp,frame,base,spec,2)
    ps[2].write_bytes(snapshot[ps[2]])
    ps[1].unlink()
    with patch.object(CatBoostClassifier,'fit',side_effect=AssertionError('No fallback fit')):
        with pytest.raises(ValueError,match='Partial'):models.fit_model(c,fp,frame,base,spec,2)


@pytest.mark.parametrize('kind',['bagged_balanced','bagged_unweighted','spline','xgboost','lightgbm_balanced','lightgbm_unweighted'])
def test_six_reference_formats_through_new_destination(tmp_path,kind):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path/'own'),threads=1);fp={'fixture':'new destination source'};old=run.parent.parent
    if kind.startswith('bagged_'):
        frame,base=old.parent.synthetic_data();spec=next(s for s in old.scores.recipes() if s['weighting']==kind[7:]);month=2
    else:
        base,y=old.parent.previous.newton.miniature_frame();frame=base.assign(month=np.where(np.arange(len(base))<120,3,4),fraud_bool=y);month=4
        name={'spline':'spline_c1p0_balanced','xgboost':'xgb_d4_w1','lightgbm_balanced':'baseline_balanced','lightgbm_unweighted':'baseline_lgbm'}[kind]
        spec=next(s for s in old.reference_specs(c) if s['id']==name)
    with patch.object(run,'required_sources',return_value=[spec]),patch.object(old,'source_origins',return_value=[]):r=run.prepare_source(c,fp,frame,base,spec,month)
    assert not r.get('reused');ps=old.source_paths(c,spec,month,'validation');before={p:p.read_bytes() for p in ps}
    bad=copy.deepcopy(r);bad['train_rows']+=1;write_json(ps[0],bad)
    with patch.object(run,'required_sources',return_value=[spec]):
        with pytest.raises(ValueError):run.prepare_source(c,fp,frame,base,spec,month)
    ps[0].write_bytes(before[ps[0]])
    other=dict(c,output_dir=str(tmp_path/'copy'));otherfp={'fixture':'copied reference'}
    with patch.object(run,'required_sources',return_value=[spec]),patch.object(old,'source_origins',return_value=[(c,fp)]),patch('lightgbm.LGBMClassifier.fit',side_effect=AssertionError('No refit')):
        copied=run.prepare_source(other,otherfp,frame,base,spec,month);assert copied['reused']
        assert copied['origin_artifacts']=={str(p):digest(p) for p in ps}
        assert run.prepare_source(other,otherfp,frame,base,spec,month)==copied
    assert all(p.read_bytes()==v for p,v in before.items());ps[1].unlink()
    with patch.object(run,'required_sources',return_value=[spec]),patch.object(old,'source_origins',side_effect=AssertionError('No fallback origin')):
        with pytest.raises(ValueError,match='Partial'):run.prepare_source(c,fp,frame,base,spec,month)


CASES=read_config(run.planning_evidence_path())['winner_policy_cases']
@pytest.mark.parametrize('case',CASES,ids=lambda c:c['winner']+'-'+c['policy'])
def test_all_winner_policy_cases_preserve_roles_and_bounds(case):
    roles=models.choices_for(case['winner'],case['policy']);dag=models.dependencies([n for n,k in roles.values()])
    assert roles==case['roles'] and dag==case['dependencies']
    kind=models.lookup(case['winner'])['kind'];assert len(roles)==(8 if kind=='reference' else 22)
    assert len(dag['catboost_bases'])==(0 if kind=='reference' else 5)
    assert len(dag['full_scores'])=={'reference':13,'native':21,'blend':25}[kind]
    assert len(dag['old_bases'])==8 and len(dag['old_derived'])==5
    assert len(dag['full_scores'])==case['physical_score_bound']
    if kind!='reference':
        assert roles['alpha_a000'][0]==models.ANCHOR
        assert roles['alpha_a100']==roles['pure_catboost']
    assert roles['same_strategy']==[case['winner'],'pooled']


@pytest.fixture(scope='module')
def analytical():
    c=read_config(run.CONFIG);n=35;rng=np.random.default_rng(2326)
    frame=pd.DataFrame(dict(month=np.repeat(np.arange(7),n),fraud_bool=np.tile((np.arange(n)%7==0).astype(int),7)))
    h={name:{} for name in run.physical_names()};old=models.dependencies([]);parent=run.parent.parent
    for m in [4,5,6]:
        ids=frame.index[frame.month==m].to_numpy();y=frame.loc[ids,'fraud_bool'].to_numpy()
        for name in old['old_bases']:h[name][m]=dict(row_ids=ids,labels=y,scores=rng.uniform(.01,.99,n))
        for name in parent.derived_order([n for n,k in models.REFERENCE_ROLES.values()]):
            spec=parent.derived_spec(name);items={v:h[v][m] for v in spec['members']}
            h[name][m]=parent.parent.old_compose(spec,items) if spec['kind']=='anchor' else parent.scores.compose(spec,items)
        for spec in models.recipes():h[spec['id']][m]=dict(row_ids=ids,labels=y,scores=rng.normal(size=n))
        # Degenerate native scores retain diversity rows and tied-score diagnostics.
        h['cb_d3_r200_w1'][m]['scores'][:]=.7
        for spec in models.strategies():
            if spec['kind']=='blend':h[spec['id']][m]=models.blend(h[models.ANCHOR][m],h[spec['members'][1]][m],spec['alpha_numerator']/100)
    tables=run.tables(c,frame,h);return c,frame,h,tables


def test_complete_analytical_tables_independently_reconstruct(analytical):
    c,frame,h,t=analytical;result=audit.audit_tables(c,frame,h,t)
    assert result['counts']==dict(crossfit=324,summary=162,pooled=162,contrasts=1728,physical_score_diagnostics=183,score_diagnostics=162,diversity=36,decision_overlap=504)
    g=t['diversity'].query("candidate == 'cb_d3_r200_w1'");assert g.correlation_undefined.all() and g.pearson.isna().all()
    assert any(t['contrasts'].f2_difference<0) and any(t['contrasts'].f2_difference==0)


@pytest.mark.parametrize('table,column', [('crossfit','f2'),('summary','recall'),('pooled','tp'),('contrasts','f2_difference'),
    ('physical_score_diagnostics','q0.25'),('score_diagnostics','outside_unit'),('diversity','spearman'),('decision_overlap','fraud_first_only')])
def test_independent_audit_rejects_numeric_corruption(analytical,table,column):
    c,frame,h,t=analytical;bad={k:v.copy(deep=True) for k,v in t.items()};index=bad[table][column].first_valid_index();bad[table].loc[index,column]+=1
    with pytest.raises(ValueError):audit.audit_tables(c,frame,h,bad)


@pytest.mark.parametrize('table',run.TABLES)
def test_independent_audit_rejects_missing_and_duplicate_keys(analytical,table):
    c,frame,h,t=analytical;name=table[:-4]
    for mode in ['missing','duplicate']:
        bad=dict(t);bad[name]=t[name].iloc[1:].copy() if mode=='missing' else pd.concat([t[name],t[name].iloc[:1]],ignore_index=True)
        with pytest.raises(ValueError):audit.audit_tables(c,frame,h,bad)


@pytest.mark.parametrize('damage',['domain','ids','labels','nonfinite','month','rule'])
def test_audit_rejects_domain_population_and_policy_corruption(analytical,damage):
    c,frame,h,t=analytical;h=copy.deepcopy(h);t={k:v.copy(deep=True) for k,v in t.items()}
    if damage=='domain':h['baseline_balanced'][5]['scores'][0]=2
    elif damage=='ids':h[models.ANCHOR][5]['row_ids'][0]=-100
    elif damage=='labels':h[models.ANCHOR][5]['labels'][0]^=1
    elif damage=='nonfinite':h['cb_d3_r200_w1'][6]['scores'][0]=np.inf
    elif damage=='month':h[models.ANCHOR][7]=h[models.ANCHOR].pop(4)
    else:
        rule=json.loads(t['crossfit'].loc[0,'parameters']);rule['source_months']=[5];t['crossfit'].loc[0,'parameters']=json.dumps(rule)
    with pytest.raises(ValueError):audit.audit_tables(c,frame,h,t)


@pytest.mark.parametrize('field,value',[('sample_weight',[1,2]),('auto_class_weights','Balanced'),('scale_pos_weight',100),('class_weights',[1,200]),('early_stopping_rounds',10)])
def test_duplicate_costs_and_undeclared_settings_rejected_before_fit(field,value):
    spec=copy.deepcopy(models.recipes()[0]);spec['params'][field]=value
    with patch.object(CatBoostClassifier,'fit',side_effect=AssertionError('Fit forbidden')):
        with pytest.raises(ValueError):models.validate_recipe(spec)


@pytest.mark.parametrize('a',[0,1,-.1,.75,True,np.nan])
def test_only_declared_interior_composition_weights(a):
    d=dict(row_ids=np.arange(2),labels=np.array([0,1]),scores=np.array([-3.,2.]))
    with pytest.raises(ValueError):models.blend(d,d,a)


def test_preservation_and_prospective_enumeration_without_fits():
    c=read_config(run.CONFIG);run.guard(c);p=read_config(run.planning_evidence_path())
    assert models.recipes()==p['recipes'] and models.strategies()==p['strategies'] and models.contrast_factors()==p['contrasts']
    assert c['preserve_sha256']==p['mandatory_preservation_sha256'] and len(run.SOURCES)==61
    assert audit.independent_names()==[s['id'] for s in models.strategies()]
    assert audit.independent_physical_names()==run.physical_names()
    view=run.adapter_view(c);assert view['parent_config']==read_config(run.parent.CONFIG)['parent_config']
    assert view['output_dir']==c['output_dir'] and view['parent_fingerprint']==c['reference_adapter_fingerprint']
    assert c['parent_config']!=view['parent_config'] and not any(k in c for k in ['review_quota','precision_floor','primary_fraction'])


def test_only_required_individual_sources_can_execute():
    c=read_config(run.CONFIG)
    with patch.object(run.parent.parent,'prepare_source',side_effect=AssertionError('Unrelated source forbidden')):
        with pytest.raises(ValueError):run.prepare_source(c,{},None,None,{'id':'unrelated'},4)


def test_source_gate_uses_reference_view_not_conditional_gate():
    c=read_config(run.CONFIG);fp={'fixture':'gate'}
    with patch.object(run.parent,'source_miniature_gate',return_value='preserved') as gate:
        assert run.source_miniature_gate(c,fp)=='preserved';gate.assert_called_once_with(run.adapter_view(c),fp)


def test_canonical_figure_curves_and_checked_rendering(analytical,tmp_path):
    c,frame,h,t=analytical;di=pd.DataFrame([dict(scope='macro',reference=a,difference=0.,ci_low=-.1,ci_high=.1,replicates=500) for a in audit.REFS]);fi=di.assign(scope='7_consumed')
    result=audit.operating_figure(t['summary'],t['contrasts'],di,fi,tmp_path)
    assert result['operating_points']==162 and result['curves']==36 and result['endpoint_rows']==180
    curves=pd.read_csv(tmp_path/'plotted_alpha_curves.csv');assert len(curves[curves.alpha==0])==36 and curves[curves.alpha==0].candidate.eq(models.ANCHOR).all()
    curves.loc[0,'candidate']='false_alias';curves.to_csv(tmp_path/'plotted_alpha_curves.csv',index=False)
    with pytest.raises(AssertionError):audit.audit_figure(t['summary'],t['contrasts'],di,fi,tmp_path)
