"""Conditional-model methodology; four current and six reference tiny fits."""
import copy
from itertools import product
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

from baf import conditional_experts as models
from baf import conditional_expert_study as run
from baf import conditional_expert_report as report
from baf.common import read_config, write_json, digest, make_model
from baf.improvements import prediction


def synthetic():
    rng = np.random.default_rng(881)
    n = 25200; index = np.arange(n)
    a = index % 2 == 0; x = rng.normal(size=n); z = rng.normal(size=n)
    base = pd.DataFrame({'signal': x, 'noise': z})
    base.loc[index % 23 == 0, 'noise'] = np.nan
    for col in ['employment_status', 'source', 'device_os']:
        base[col] = np.where(index % 3 == 0, 'a', 'b')
    base['payment_type'] = np.where(a, 'AA', 'AB')
    base['housing_status'] = np.where(index % 4 < 2, 'BA', 'BB')
    y = (x + .5*z + rng.normal(size=n) > np.where(a, .8, 2.1)).astype(int)
    frame = base.assign(month=np.where(index < 24000, index // 12000, 2), fraud_bool=y)
    frame.loc[24000, ['payment_type', 'housing_status']] = ['AE', 'BF']
    frame.loc[24001, ['payment_type', 'housing_status']] = [None, None]
    frame.loc[24002, ['payment_type', 'housing_status']] = ['unknown', 'unknown']
    base.loc[frame.index, ['payment_type', 'housing_status']] = frame[['payment_type', 'housing_status']]
    return frame, base


@pytest.fixture(scope='module')
def synthetic_context():
    frame, base = synthetic()
    original = read_config('configs/study.json')
    pre = make_model('lightgbm', False, original, base.columns).named_steps['preprocess']
    pre.fit(base.loc[frame.month < 2])
    return frame, base, pre


def test_prospective_declaration_matches_independent_enumeration():
    c = read_config(run.CONFIG); p = read_config(run.planning_evidence_path())
    assert models.strategies() == p['strategies']
    assert [s['id'] for s in models.strategies()] == report.independent_strategy_ids()
    assert models.contrast_factors() == p['contrasts']
    assert len(models.contrast_factors()) == 738
    assert len({str(v) for v in models.contrast_factors()}) == 738
    assert c['preserve_sha256'] == p['mandatory_preservation_sha256']
    assert len(c['source_paths']) == 58 and len(c['preserve_sha256']) == 1053
    assert c['parent_config'] == run.parent.CONFIG
    assert not any(k in c for k in ['review_quota', 'precision_floor', 'primary_fraction'])


@pytest.mark.parametrize('archived',[False,True])
def test_planning_evidence_resolves_after_archive_without_config_changes(tmp_path,archived):
    root=tmp_path/'changes'
    folder='archive/2026-10-12-test-conditional-lightgbm-experts-f2' if archived else 'test-conditional-lightgbm-experts-f2'
    path=root/folder/'planning-evidence.json';write_json(path,{'fixture':True})
    assert run.planning_evidence_path(root)==path


@pytest.mark.parametrize('name,policy', list(product(report.independent_strategy_ids(), ['pooled','latest','rank_fraction'])))
def test_all_winner_policy_cases_preserve_roles_and_bounds(name, policy):
    p = read_config(run.planning_evidence_path())
    original = next(c for c in p['winner_policy_cases'] if c['winner'] == name and c['policy'] == policy)
    choices = models.choices_for(name, policy)
    assert choices == original['roles']
    kind = models.lookup(name)['kind']
    assert len(choices) == (8 if kind == 'reference' else 10 if kind == 'capacity' else 17)
    assert choices['same_strategy'] == [name, 'pooled']
    assert all(choices[role] == pair for role, pair in models.REFERENCE_ROLES.items())


@pytest.mark.parametrize('weighting', models.WEIGHTINGS)
def test_global_costs_are_independently_reconstructible(synthetic_context, weighting):
    frame, base, pre = synthetic_context
    y = frame.loc[frame.month < 2, 'fraud_bool'].to_numpy()
    actual = models.class_costs(y, weighting)
    assert actual == report.direct_class_costs(y, weighting == 'balanced')
    if weighting == 'balanced':
        assert actual != models.class_costs(frame.loc[(frame.month < 2) & frame.payment_type.eq('AA'), 'fraud_bool'], weighting)
    assert models.weights_digest(models.weights_for(y, actual)) == models.weights_digest(models.weights_for(y.copy(), actual))


@pytest.mark.parametrize('partition', models.PARTITIONS)
def test_eligibility_only_uses_earlier_training(synthetic_context, partition):
    frame, base, pre = synthetic_context; original = models.eligibility(frame, 2, partition)
    changed = frame.copy(); changed.loc[changed.month == 2, 'fraud_bool'] = 1
    changed.loc[changed.month == 2, partition] = 'future-new'
    assert models.eligibility(changed, 2, partition) == original
    assert sum(g['eligible'] for g in original) == 2
    for g in original:
        y = frame.loc[(frame.month < 2) & frame[partition].eq(g['level']), 'fraud_bool']
        assert g['train_rows'] == len(y) and g['train_fraud'] == int(y.sum())


@pytest.mark.parametrize('kind,weighting', list(product(['local','capacity'], models.WEIGHTINGS)))
def test_four_new_formats_fit_audit_reuse_and_reject_corruption(tmp_path, synthetic_context, kind, weighting):
    frame, base, pre = synthetic_context
    c = dict(read_config(run.CONFIG), output_dir=str(tmp_path), threads=1)
    fp = {'fixture': 'conditional tiny model; not a production fit'}
    spec = next(s for s in models.model_specs(frame, 2) if s['kind'] == kind and s['weighting'] == weighting)
    calls = []; original = models.estimator(c, spec['rounds']).__class__.fit
    def observe(self, x, y, **kwargs):
        calls.append((y.copy(), kwargs['sample_weight'].copy()))
        return original(self, x, y, **kwargs)
    with patch('lightgbm.LGBMClassifier.fit', observe):
        record = models.fit_model(c, fp, frame, base, spec, 2, pre)
    assert len(calls) == 1
    assert np.array_equal(calls[0][1], models.weights_for(calls[0][0], models.class_costs(
        frame.loc[frame.month < 2, 'fraud_bool'], weighting)))
    ps = models.model_paths(c, 'validation', 2, spec['id'])
    original_record = ps[0].read_bytes(); original_model = ps[2].read_bytes()
    original_prediction = ps[1].read_bytes()
    with patch('lightgbm.LGBMClassifier.fit', side_effect=AssertionError('No refit')):
        assert models.fit_model(c, fp, frame, base, spec, 2, pre) == record
    pipe = joblib.load(ps[2])['model']; _, ev = models.membership(frame, spec, 2)
    saved = prediction(c, 'validation', 2, spec['id'])
    assert np.allclose(pipe.predict_proba(base.loc[ev[::-1]])[:,1][::-1], saved['scores'], atol=1e-12, rtol=0)
    extended=pd.concat([base.loc[ev],base.loc[ev[:1]].assign(signal=999)],ignore_index=True)
    assert np.allclose(pipe.predict_proba(extended)[:len(ev),1],saved['scores'],atol=1e-12,rtol=0)
    for field in ['local_train_rows', 'global_train_fraud', 'local_train_ids_sha256', 'global_weight_sha256', 'class_costs', 'fitted_tree_count']:
        bad = copy.deepcopy(record); bad[field] = 'changed'; write_json(ps[0], bad)
        with pytest.raises(ValueError): models.audit_model(c, fp, frame, base, spec, 2, pre)
        ps[0].write_bytes(original_record)
    saved['scores'] = 1 - saved['scores']; np.savez_compressed(ps[1], **saved)
    bad = copy.deepcopy(record); bad['prediction_sha256'] = digest(ps[1]); write_json(ps[0], bad)
    with pytest.raises(ValueError, match='inference'): models.audit_model(c, fp, frame, base, spec, 2, pre)
    ps[0].write_bytes(original_record); ps[1].write_bytes(original_prediction)
    artifact = joblib.load(ps[2]); artifact['model'].named_steps['model'].set_params(class_weight='balanced')
    joblib.dump(artifact, ps[2]); bad = copy.deepcopy(record); bad['model_sha256'] = digest(ps[2]); write_json(ps[0], bad)
    with pytest.raises(ValueError, match='recipe'): models.audit_model(c, fp, frame, base, spec, 2, pre)
    ps[2].write_bytes(original_model); ps[0].write_bytes(original_record)
    ps[1].unlink()
    with patch('lightgbm.LGBMClassifier.fit', side_effect=AssertionError('No partial retry')):
        with pytest.raises(ValueError, match='Partial'): models.fit_model(c, fp, frame, base, spec, 2, pre)


def test_failed_eligible_fit_retains_failure_and_never_falls_back(tmp_path, synthetic_context):
    frame, base, pre = synthetic_context; c = dict(read_config(run.CONFIG), output_dir=str(tmp_path))
    spec = next(s for s in models.model_specs(frame, 2) if s['kind'] == 'local')
    with patch('lightgbm.LGBMClassifier.fit', side_effect=ValueError('numerical fixture failure')):
        with pytest.raises(ValueError, match='numerical'): models.fit_model(c, {}, frame, base, spec, 2, pre)
    failure = tmp_path/'failures'/('validation-m2-'+spec['id']+'.json')
    assert 'numerical fixture failure' in read_config(failure)['error']
    with patch('lightgbm.LGBMClassifier.fit', side_effect=AssertionError('No retry')):
        with pytest.raises(ValueError, match='Prior conditional failure'): models.fit_model(c, {}, frame, base, spec, 2, pre)


@pytest.mark.parametrize('partition', models.PARTITIONS)
def test_exact_stitching_fallbacks_membership_and_label_invariance(synthetic_context, partition):
    frame, base, pre = synthetic_context; ev = frame.index[frame.month == 2].to_numpy()
    global_data = dict(row_ids=ev, labels=frame.loc[ev,'fraud_bool'].to_numpy(), scores=np.full(len(ev),.2))
    levels = [g['level'] for g in models.eligibility(frame,2,partition) if g['eligible']]
    subsets = {}
    for i, level in enumerate(levels):
        mask = frame.loc[ev,partition].eq(level).to_numpy()
        subsets[level] = dict(row_ids=ev[mask], labels=global_data['labels'][mask], scores=np.full(int(mask.sum()),.5+i*.1))
    d, routing = models.stitch(frame,2,partition,global_data,subsets,levels)
    assert len(routing) == len(models.PARTITIONS[partition])+2
    assert set(r['route'] for r in routing)=={'expert','missing','unseen','ineligible'}
    assert np.all(d['scores'][:3] == .2) and sum(r['rows'] for r in routing) == len(ev)
    changed = frame.copy(); changed.loc[ev,'fraud_bool'] = 1-changed.loc[ev,'fraud_bool']
    gd = dict(global_data,labels=1-global_data['labels'])
    sd = {k:dict(v,labels=1-v['labels']) for k,v in subsets.items()}
    changed_d, _ = models.stitch(changed,2,partition,gd,sd,levels)
    assert np.array_equal(d['scores'],changed_d['scores'])
    with pytest.raises(ValueError,match='exact earlier'): models.stitch(frame,2,partition,global_data,{},[])
    broken = copy.deepcopy(subsets); broken[levels[0]]['row_ids'] = broken[levels[0]]['row_ids'][::-1]
    with pytest.raises(ValueError,match='membership'): models.stitch(frame,2,partition,global_data,broken,levels)


@pytest.mark.parametrize('alpha',[0,.25,.5,.75,1])
def test_probability_blends_and_endpoints(alpha):
    g=dict(row_ids=np.array([1,2]),labels=np.array([0,1]),scores=np.array([.1,.8]))
    e=dict(g,scores=np.array([.3,.9]))
    d=models.blend(g,e,alpha)
    assert np.array_equal(d['scores'],(1-alpha)*g['scores']+alpha*e['scores'])
    if alpha==0: assert np.array_equal(d['scores'],g['scores'])
    if alpha==1: assert np.array_equal(d['scores'],e['scores'])


@pytest.mark.parametrize('values',[[np.nan],[np.inf],[-.01],[1.01],[[.5]]])
def test_invalid_probability_domains_fail(values):
    with pytest.raises(ValueError): models.probability(values)


@pytest.mark.parametrize('kind',['bagged_balanced','bagged_unweighted','spline_balanced','spline_unweighted','xgboost','lightgbm'])
def test_six_reference_formats_through_new_destination(tmp_path, kind):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path/'own'),threads=1)
    fp={'fixture':'conditional destination reference format'}
    old=run.parent
    if kind.startswith('bagged_'):
        frame,base=old.parent.synthetic_data()
        spec=next(s for s in old.scores.recipes() if s['weighting']==kind[7:]); month=2
    else:
        base,y=old.parent.previous.newton.miniature_frame()
        frame=base.assign(month=np.where(np.arange(len(base))<120,3,4),fraud_bool=y);month=4
        name={'spline_balanced':'spline_c1p0_balanced','spline_unweighted':'spline_c0p1_unweighted','xgboost':'xgb_d4_w1','lightgbm':'baseline_lgbm'}[kind]
        spec=next(s for s in old.reference_specs(c) if s['id']==name)
    # The unweighted bagged format is a synthetic format check, never a production dependency.
    with patch.object(run,'required_sources',return_value=[spec]),patch.object(old,'source_origins',return_value=[]):
        record=run.prepare_source(c,fp,frame,base,spec,month)
    assert not record.get('reused')
    ps=old.source_paths(c,spec,month,'validation');before={p:p.read_bytes() for p in ps}
    corrupted=copy.deepcopy(record);corrupted['train_rows']+=1;write_json(ps[0],corrupted)
    with patch.object(run,'required_sources',return_value=[spec]):
        with pytest.raises(ValueError):run.prepare_source(c,fp,frame,base,spec,month)
    ps[0].write_bytes(before[ps[0]])
    other=dict(c,output_dir=str(tmp_path/'copied'));otherfp={'fixture':'new-destination copy'}
    with patch.object(run,'required_sources',return_value=[spec]),patch.object(old,'source_origins',return_value=[(c,fp)]),patch('lightgbm.LGBMClassifier.fit',side_effect=AssertionError('No refit')):
        copied=run.prepare_source(other,otherfp,frame,base,spec,month)
        assert copied['reused'] and copied['origin_artifacts']=={str(p):digest(p) for p in ps}
        assert run.prepare_source(other,otherfp,frame,base,spec,month)==copied
    assert all(p.read_bytes()==v for p,v in before.items())
    ps[1].unlink()
    with patch.object(run,'required_sources',return_value=[spec]),patch.object(old,'source_origins',side_effect=AssertionError('No fallback')):
        with pytest.raises(ValueError,match='Partial'):run.prepare_source(c,fp,frame,base,spec,month)


@pytest.fixture(scope='module')
def analytical():
    n=42; rng=np.random.default_rng(2001)
    frame=pd.DataFrame({'month':np.repeat(np.arange(7),n),
        'fraud_bool':np.tile((np.arange(n)%7==0).astype(int),7),
        'payment_type':np.tile(np.where(np.arange(n)%2==0,'AA','AB'),7),
        'housing_status':np.tile(np.where(np.arange(n)%2==0,'BA','BB'),7)})
    frame.loc[frame.index%42==0,'payment_type']=None
    frame.loc[frame.index%42==1,'housing_status']='unknown'
    c=read_config(run.CONFIG);h={name:{} for name in run.physical_names()}
    old_names=[n for n,k in models.REFERENCE_ROLES.values()]
    dag=run.parent.scores.dependencies(old_names)
    for m in [4,5,6]:
        ids=frame.index[frame.month==m].to_numpy();y=frame.loc[ids,'fraud_bool'].to_numpy()
        for name in dag['bases']:
            h[name][m]=dict(row_ids=ids,labels=y,scores=rng.uniform(.01,.99,n))
        for name in run.parent.derived_order(old_names):
            spec=run.parent.derived_spec(name);items={key:h[key][m] for key in spec['members']}
            h[name][m]=(run.parent.parent.old_compose(spec,items) if spec['kind']=='anchor' else run.parent.scores.compose(spec,items))
        for w in models.WEIGHTINGS:
            h[models.capacity(w)][m]=dict(row_ids=ids,labels=y,scores=rng.uniform(.01,.99,n))
        for p,w in product(models.PARTITIONS,models.WEIGHTINGS):
            # All synthetic groups are ineligible, so the pure score is global fallback.
            g=h[models.global200(w)][m];h[models.expert(p,w)][m]=dict(g,scores=g['scores'].copy())
            for a in [25,50,75]:h[models.candidate(p,w,a)][m]=models.blend(g,h[models.expert(p,w)][m],a/100)
    generated=run.tables(c,frame,h)
    return c,frame,h,generated


def test_complete_analytical_tables_independently_reconstructed(analytical):
    c,frame,h,generated=analytical
    receipt=report.audit_tables(c,frame,h,generated)
    assert receipt['counts']=={'crossfit':144,'summary':72,'pooled':72,'contrasts':738,
        'eligibility':72,'routing':96,'stratum_errors':2304,'physical_score_diagnostics':93,'score_diagnostics':72}
    assert not receipt['actual_source_models_audited']
    assert (generated['stratum_errors'].query('n == 0').accuracy==0).all()


@pytest.mark.parametrize('table',['crossfit','summary','pooled','contrasts','eligibility','routing','stratum_errors','physical_score_diagnostics','score_diagnostics'])
@pytest.mark.parametrize('change',['missing','duplicate','identity','numeric'])
def test_all_analytical_table_corruptions_fail(analytical,table,change):
    c,frame,h,original=analytical; generated={k:v.copy(deep=True) for k,v in original.items()};t=generated[table]
    if change=='missing':generated[table]=t.iloc[1:]
    elif change=='duplicate':generated[table]=pd.concat([t,t.iloc[:1]],ignore_index=True)
    elif change=='identity':
        key='partition' if table in ['eligibility','routing'] else 'candidate';t.loc[0,key]='undeclared'
    else:
        key={'crossfit':'f2','summary':'f2','pooled':'f2','contrasts':'f2_difference','eligibility':'train_rows',
             'routing':'rows','stratum_errors':'fp','physical_score_diagnostics':'score_mean','score_diagnostics':'score_mean'}[table]
        t.loc[0,key]+=1
    with pytest.raises(ValueError):report.audit_tables(c,frame,h,generated)


def test_policy_uses_earlier_labels_and_selection_uses_actual_f2(analytical):
    c,frame,h,generated=analytical
    changed=copy.deepcopy(h)
    for n in changed:changed[n][6]['labels']=1-changed[n][6]['labels']
    actual=run.crossfit(c,changed)
    assert actual.parameters.tolist()==generated['crossfit'].parameters.tolist()
    # Month-6 outcomes change metrics but do not alter its already-fitted rules.
    assert not np.array_equal(actual.f2.to_numpy(),generated['crossfit'].f2.to_numpy())


@pytest.mark.parametrize('field',['parameters','source_months'])
def test_forward_rule_or_chronology_corruption_fails(analytical,field):
    c,frame,h,original=analytical;generated={k:v.copy(deep=True) for k,v in original.items()}
    generated['crossfit'].loc[0,field]='{}' if field=='parameters' else '5'
    with pytest.raises(ValueError):report.audit_tables(c,frame,h,generated)


def test_unrelated_production_recipe_rejected_before_adapter(synthetic_context,tmp_path):
    frame,base,pre=synthetic_context;c=read_config(run.CONFIG)
    spec=next(s for s in run.parent.scores.recipes() if s['weighting']=='unweighted')
    with patch.object(run.parent,'prepare_source',side_effect=AssertionError('No unrelated preparation')):
        with pytest.raises(ValueError,match='Unrelated'):run.prepare_source(c,{},frame,base,spec,2)


@pytest.fixture(scope='module')
def bound_population():
    # Synthetic eligibility only: same 4/5 counts as the observed final training population.
    cells=list(product(['AA','AB','AC','AD'],['BA','BB','BC','BD','BE']))
    p=[x for x,y in cells for _ in range(3000)];h=[y for x,y in cells for _ in range(3000)]
    return pd.DataFrame(dict(month=np.zeros(60000,dtype=int),payment_type=p,housing_status=h,
                             fraud_bool=(np.arange(60000)%9==0).astype(int)))


@pytest.mark.parametrize('name,policy',list(product(report.independent_strategy_ids(),['pooled','latest','rank_fraction'])))
def test_all_exact_winner_dags_and_observed_type_bounds(bound_population,name,policy):
    p=read_config(run.planning_evidence_path())
    case=next(v for v in p['winner_policy_cases'] if v['winner']==name and v['policy']==policy)
    choices=models.choices_for(name,policy);dag=models.dependencies([n for n,k in choices.values()],bound_population)
    assert dag['full_scores']==case['dependencies']['full_scores']
    assert dag['old_bases']==case['dependencies']['old_bases']
    assert sorted(s['id'] for s in dag['new_models'] if s['kind']=='local')==case['dependencies']['expert_models']
    assert sorted(s['id'] for s in dag['new_models'] if s['kind']=='capacity')==case['dependencies']['capacity_bases']
    assert len(dag['new_models'])==case['current_new_capacity_fit_bound']+case['current_new_expert_fit_bound']


@pytest.mark.parametrize('n,fraud,expected',[(9999,100,False),(10000,99,False),(10000,9901,False),(10000,100,True),(10000,9900,True)])
def test_original_eligibility_boundaries(n,fraud,expected):
    frame=pd.DataFrame(dict(month=np.zeros(n,dtype=int),payment_type=['AA']*n,
        fraud_bool=np.r_[np.ones(fraud,dtype=int),np.zeros(n-fraud,dtype=int)]))
    g=models.eligibility(frame,1,'payment_type')[0]
    assert g['eligible']==expected


@pytest.mark.parametrize('partition,weighting',list(product(models.PARTITIONS,models.WEIGHTINGS)))
def test_saved_conditional_compositions_and_subset_corruptions(tmp_path,synthetic_context,partition,weighting):
    frame,base,pre=synthetic_context;c=dict(read_config(run.CONFIG),output_dir=str(tmp_path))
    fp={'fixture':'scores only, no source-model audit claim'};ev=frame.index[frame.month==2].to_numpy()
    y=frame.loc[ev,'fraud_bool'].to_numpy()
    def save(name,ids,labels,scores):
        rp,pp=run.paths(c,'validation',2,name);pp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(pp,row_ids=ids,labels=labels,scores=scores)
        write_json(rp,{'fixture':'synthetic member score metadata, no fitted source claim'})
    global_name=models.global200(weighting);save(global_name,ev,y,np.full(len(ev),.2))
    levels=[g['level'] for g in models.eligibility(frame,2,partition) if g['eligible']]
    for i,level in enumerate(levels):
        take=frame.loc[ev,partition].eq(level).to_numpy();save(f'expert_model_{partition}_{weighting}_{level}',ev[take],y[take],np.full(int(take.sum()),.6+i*.1))
    names=[models.expert(partition,weighting)]+[models.candidate(partition,weighting,a) for a in [25,50,75]]
    for name in names:run.new_derived(c,fp,frame,name,2)
    receipt=report.audit_new_compositions(c,fp,frame,[2],names=names)
    assert receipt['new_full_compositions']==4 and receipt['member_subsets']==2
    assert not receipt['actual_source_models_audited']
    for name in names:
        rp,pp=run.paths(c,'validation',2,name);rb,pb=rp.read_bytes(),pp.read_bytes()
        d=prediction(c,'validation',2,name);d['scores'][0]=.99;np.savez_compressed(pp,**d)
        r=read_config(rp);r['prediction_sha256']=digest(pp);write_json(rp,r)
        with pytest.raises(ValueError,match='arithmetic'):report.audit_new_compositions(c,fp,frame,[2],names=names)
        with pytest.raises(ValueError):run.new_derived(c,fp,frame,name,2)
        rp.write_bytes(rb);pp.write_bytes(pb)
    local_name=f'expert_model_{partition}_{weighting}_{levels[0]}'
    rp,pp=run.paths(c,'validation',2,local_name);d=prediction(c,'validation',2,local_name)
    d['row_ids']=d['row_ids'][::-1];np.savez_compressed(pp,**d)
    with pytest.raises(ValueError,match='membership'):report.audit_new_compositions(c,fp,frame,[2],names=names)


def test_production_miniature_full_schema_original_minima_and_prevalence():
    columns=pd.read_csv('data/raw/baf/Base.csv',nrows=0).drop(columns=['fraud_bool','month','device_fraud_count']).columns
    frame,base=run.miniature_data(columns)
    assert list(base)==list(columns) and len(base.columns)==29
    assert frame.groupby('month').size().to_dict()=={0:12000,1:12000,2:12000}
    for p in models.PARTITIONS:
        groups=models.eligibility(frame,2,p)
        valid=[g for g in groups if g['eligible']]
        assert len(valid)==2 and all(g['train_rows']==12000 and g['train_fraud']>=100 and g['train_legitimate']>=100 for g in valid)
        assert set(frame.loc[frame.month==2,p].dropna())>=set([models.PARTITIONS[p][0],'unseen'])
    assert models.class_costs(frame.loc[frame.month<2,'fraud_bool'],'balanced')!=models.class_costs(
        frame.loc[(frame.month<2)&frame.payment_type.eq('AA'),'fraud_bool'],'balanced')


@pytest.mark.parametrize('name,policy',list(product(report.independent_strategy_ids(),['pooled','latest','rank_fraction'])))
def test_final_bounds_all_cases_with_wholly_absent_original_caches(bound_population,name,policy):
    c=read_config(run.CONFIG);choices=models.choices_for(name,policy)
    with patch.object(run.parent,'source_origins',return_value=[]):
        bounds=run.final_fit_bounds(c,choices,bound_population)
    p=read_config(run.planning_evidence_path())
    case=next(v for v in p['winner_policy_cases'] if v['winner']==name and v['policy']==policy)
    assert bounds['physical_scores']==len(case['dependencies']['full_scores'])
    assert bounds['max_new_local_fits']==case['current_new_expert_fit_bound']
    assert bounds['max_new_capacity_fits']==case['current_new_capacity_fit_bound']
    assert bounds['max_reproduced_bagged_fits']==3 and bounds['max_reproduced_spline_fits']==2 and bounds['max_reproduced_tree_fits']==3
    assert set(bounds['original_availability'].values())=={None}
    assert len(bounds['eligibility'])==12 and all(g['eval_month']==7 for g in bounds['eligibility'])


@pytest.fixture
def freeze_context(tmp_path,analytical,monkeypatch):
    c,frame,h,tables=analytical
    c=dict(c,output_dir=str(tmp_path/'own'),report_dir=str(tmp_path/'report'))
    out=Path(c['output_dir']);out.mkdir(parents=True);fp={'fixture':'immutable execution identity'}
    monkeypatch.setattr(run,'initialize',lambda config_path=run.CONFIG:(c,fp))
    monkeypatch.setattr(run,'execution_gate',lambda *a:None)
    monkeypatch.setattr(run,'load_data',lambda *a:frame.copy())
    monkeypatch.setattr(run,'history',lambda *a,**kw:h)
    monkeypatch.setattr(run.parent,'source_origins',lambda *a:[])
    def prepare_winner(name,policy):
        cf=tables['crossfit'].copy()
        cf.loc[:,'f2']=0.
        cf.loc[(cf.candidate==name)&(cf.policy==policy),'f2']=.9
        cf.to_csv(out/'crossfit.csv',index=False)
        summary=run.order_summary(cf).reset_index(drop=True)
        summary.to_csv(out/'summary.csv',index=False)
        done=dict(fingerprint=fp,files={p:digest(out/p) for p in ['crossfit.csv','summary.csv']})
        write_json(out/'validation_complete.json',done)
        write_json(out/'independent_development_audit.json',dict(fingerprint=fp,
            validation_sha256=digest(out/'validation_complete.json'),actual_source_models_audited=True,
            saved_model_checks=84,winner=name,policy=policy,development_f2=.9))
    return c,fp,out,frame,h,prepare_winner


@pytest.mark.parametrize('name,policy',list(product(report.independent_strategy_ids(),['pooled','latest','rank_fraction'])))
def test_freeze_and_reload_every_winner_policy_contract(freeze_context,name,policy):
    c,fp,out,frame,h,prepare_winner=freeze_context;prepare_winner(name,policy)
    s=run.freeze();cc,restored=run.load_selection()
    assert cc==c and restored==s and s['choices']==models.choices_for(name,policy)
    assert s['score_dependencies']==models.dependencies([n for n,k in s['choices'].values()],frame)
    assert not (out/'exploratory_access.json').exists()
    assert s['month7_previously_consumed'] and s['exploratory']
    with pytest.raises(ValueError,match='immutable'):run.freeze()


@pytest.mark.parametrize('field',['fingerprint','config','choices','score_dependencies','final_fit_bounds',
    'rules','selected_development_f2','selected_candidate','selected_policy','independent_audit_sha256'])
def test_selection_corruption_rejected(freeze_context,field):
    c,fp,out,frame,h,prepare_winner=freeze_context
    prepare_winner('expert_blend_payment_type_balanced_a25','latest')
    s=run.freeze();s[field]='changed';write_json(out/'selection.json',s)
    with pytest.raises((ValueError,TypeError)):run.load_selection()


def test_independent_audit_required_before_freeze(freeze_context):
    c,fp,out,frame,h,prepare_winner=freeze_context;prepare_winner('baseline_balanced','pooled')
    path=out/'independent_development_audit.json';r=read_config(path)
    r['actual_source_models_audited']=False;write_json(path,r)
    with pytest.raises(ValueError,match='84'):run.freeze()
    assert not (out/'selection.json').exists()


@pytest.mark.parametrize('change',['identity','subset','attempts','incomplete','evidence'])
def test_execution_gates_reject_changed_or_incomplete_evidence(tmp_path,monkeypatch,change):
    c={'output_dir':str(tmp_path)};fp={'fixture':'gates'}
    monkeypatch.setattr(run,'check_receipt',lambda *a:{})
    artifact=tmp_path/'evidence.txt';artifact.write_text('original')
    for n in ['resource_preflight.json','source_miniature_gate.json','older_miniature_audit.json']:
        write_json(tmp_path/n,dict(fingerprint=fp,artifacts={str(artifact):digest(artifact)}))
    mini=dict(fingerprint=fp,fit_count=16,attempts=[{'status':'completed'} for _ in range(16)],
        artifacts={str(artifact):digest(artifact)},original_explicit_equivalent=True,
        same_seed_repeatable=True,shared_global_preprocessing=True)
    write_json(tmp_path/'miniature_gate.json',mini);run.execution_gate(c,fp)
    if change=='identity':mini['fingerprint']={}
    elif change=='subset':mini['fit_count']=15
    elif change=='attempts':mini['attempts']=mini['attempts'][:-1]
    elif change=='incomplete':mini['attempts'][0]['status']='failed'
    else:artifact.write_text('changed')
    write_json(tmp_path/'miniature_gate.json',mini)
    with pytest.raises(ValueError):run.execution_gate(c,fp)


def test_initialization_freezes_sources_protocol_and_preservation(tmp_path,monkeypatch):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path));cfg=tmp_path/'config.json';write_json(cfg,c)
    write_json(tmp_path/'preimplementation_preservation.json',{})
    fp={'sources':{'fixture':'original'}}
    monkeypatch.setattr(run,'guard',lambda *a:None)
    monkeypatch.setattr(run,'fingerprint',lambda *a:copy.deepcopy(fp))
    assert run.initialize(cfg)==(c,fp)
    declaration=(tmp_path/'declaration.json').read_bytes()
    run.initialize(cfg);assert (tmp_path/'declaration.json').read_bytes()==declaration
    fp['sources']['fixture']='changed'
    with pytest.raises(ValueError,match='execution declaration'):run.initialize(cfg)
    assert (tmp_path/'declaration.json').read_bytes()==declaration


def test_artifact_grid_distinguishes_objects_native_files_and_subsets(tmp_path,bound_population):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path));frame=bound_population
    names=[s['id'] for s in models.strategies()]
    for m in [4,5,6]:
        dag=models.dependencies(names,frame,m)
        old={s['id']:s for s in run.required_sources(c)}
        paths=[]
        for n in dag['old_bases']:paths.extend(run.parent.source_paths(c,old[n],m,'validation'))
        for spec in dag['new_models']:paths.extend(models.model_paths(c,'validation',m,spec['id']))
        for n in set(dag['full_scores'])-set(dag['old_bases'])-{s['id'] for s in dag['new_models']}:
            paths.extend(run.paths(c,'validation',m,n))
        for path in paths:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture only')
    assert run.exact_artifact_grid(c,frame,'validation')=={'model_artifacts':84,'model_files':87,'prediction_artifacts':147}
    extra=tmp_path/'models'/'validation-m4-unrelated.joblib';extra.write_bytes(b'not allowed')
    with pytest.raises(ValueError,match='model artifact'):run.exact_artifact_grid(c,frame,'validation')
    extra.unlink();native=next((tmp_path/'models').glob('*.ubj'));native.unlink()
    with pytest.raises(ValueError,match='model artifact'):run.exact_artifact_grid(c,frame,'validation')


@pytest.mark.parametrize('name,policy',[
    ('baseline_balanced','pooled'),('capacity400_unweighted','latest'),
    ('expert_blend_payment_type_balanced_a25','rank_fraction')])
def test_all_role_type_uncertainty_reconstructs_1500_draws(analytical,name,policy):
    c,frame,h,t=analytical;choices=models.choices_for(name,policy)
    s={'choices':choices};data={a:h[n][6] for a,(n,k) in choices.items()}
    masks={a:report.independent_mask(data[a],report.independent_rule(h[n],k,6)) for a,(n,k) in choices.items()}
    dint,ddraws,fint,fdraws=report.uncertainty(c,s,t['crossfit'],h,data,masks)
    assert len(dint)==3*(len(choices)-1) and len(fint)==len(choices)-1
    assert len(ddraws)==1000 and len(fdraws)==500
    assert set(fint.reference)==set(choices)-{'selected'}


def test_figure_values_endpoints_legends_and_regeneration(tmp_path,analytical):
    c,frame,h,t=analytical;dest=tmp_path/'one';other=tmp_path/'two'
    report.operating_figure(t['summary'],dest);report.operating_figure(t['summary'],other)
    assert digest(dest/'conditional_experts_operating.png')==digest(other/'conditional_experts_operating.png')
    assert report.audit_figure(t['summary'],dest)=={'operating_points':72,'curve_rows':60,'capacity_lines':6}
    path=dest/'plotted_alpha_curves.csv';curves=pd.read_csv(path)
    curves.loc[0,'candidate']='undeclared';curves.to_csv(path,index=False)
    with pytest.raises(ValueError,match='endpoint'):report.audit_figure(t['summary'],dest)


@pytest.fixture
def complete_check_evidence(tmp_path,monkeypatch):
    import xml.etree.ElementTree as ET
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path/'out'),report_dir=str(tmp_path/'report'))
    config=tmp_path/'config.json';write_json(config,c);dest=Path(c['report_dir']);dest.mkdir()
    names=['test_six_reference_formats_through_new_destination['+v+']' for v in [
        'bagged_balanced','bagged_unweighted','spline_balanced','spline_unweighted','xgboost','lightgbm']]
    names+=['test_four_new_formats_fit_audit_reuse_and_reject_corruption['+a+'-'+b+']' for a,b in product(['local','capacity'],models.WEIGHTINGS)]
    names+=['test_all_winner_policy_cases_preserve_roles_and_bounds['+n+'-'+k+']' for n,k in product(report.independent_strategy_ids(),report.POLICIES)]
    nodes=[run.TEST_SOURCE+'::'+n for n in names]
    nodes += [str(p)+'::test_fixture' for p in sorted(Path('tests').glob('test_*.py')) if str(p)!=run.TEST_SOURCE]
    suite=ET.Element('testsuite')
    for node in nodes:
        path,name=node.split('::');ET.SubElement(suite,'testcase',classname=path.removesuffix('.py').replace('/','.'),name=name)
    ET.ElementTree(suite).write(dest/'tests.xml')
    source_paths=set(run.SOURCES+[run.CONFIG,c['protocol_path'],'scripts/check_conditional_experts.py']+[
        str(p) for p in sorted(Path('tests').glob('test_*.py'))])
    fits=[dict(nodeid=node,status='completed') for node in nodes[:10]]
    fits += [dict(nodeid=nodes[-1],status='completed'),dict(nodeid=nodes[-1],status='failed')]
    ledger=dict(exit_code=0,junit_sha256=digest(dest/'tests.xml'),deselected_nodes=[],collected_nodes=nodes,
        fits=fits,execution_source_sha256={p:digest(p) for p in source_paths})
    write_json(dest/'test_model_fits.json',ledger)
    monkeypatch.setattr(run,'guard',lambda *a:None)
    monkeypatch.setattr(run,'fingerprint',lambda *a:{'fixture':'check evidence'})
    return c,config,dest,ledger


def test_check_receipt_observes_current_and_historical_fits_separately(complete_check_evidence):
    c,config,dest,ledger=complete_check_evidence;r=run.record_checks(config)
    assert r['actual_current_test_fits']==10 and r['actual_historical_test_fit_attempts']==2
    assert r['actual_historical_test_fits']==1 and r['actual_historical_test_fit_failures']==1
    run.record_checks(config)
    assert read_config(dest/'test_fit_counts.json')['total_current_tiny_fits']==10


@pytest.mark.parametrize('change',['selected','missing_fit','unattributed','unfinished','missing_source','changed_source','missing_case','incomplete_module'])
def test_check_receipt_rejects_incomplete_observed_execution(complete_check_evidence,change):
    c,config,dest,ledger=complete_check_evidence
    if change=='selected':ledger['deselected_nodes']=['fixture']
    elif change=='missing_fit':ledger['fits'].pop(0)
    elif change=='unattributed':ledger['fits'][0]['nodeid']='unknown'
    elif change=='unfinished':ledger['fits'][-1]['status']='started'
    elif change=='missing_source':ledger['execution_source_sha256'].pop(run.TEST_SOURCE)
    elif change=='changed_source':ledger['execution_source_sha256'][run.TEST_SOURCE]='changed'
    elif change=='missing_case':ledger['collected_nodes'].pop(0)
    else:
        import xml.etree.ElementTree as ET
        tree=ET.parse(dest/'tests.xml');root=tree.getroot();root.remove(list(root)[-1]);tree.write(dest/'tests.xml')
        ledger['collected_nodes'].pop();ledger['junit_sha256']=digest(dest/'tests.xml')
    write_json(dest/'test_model_fits.json',ledger)
    with pytest.raises(ValueError):run.record_checks(config)


@pytest.mark.parametrize('partition,weighting',list(product(models.PARTITIONS,models.WEIGHTINGS)))
def test_inference_work_counts_shared_global_plus_member_only(synthetic_context,partition,weighting):
    frame,base,pre=synthetic_context;local={g['level']:190 for g in models.eligibility(frame,2,partition) if g['eligible']}
    rows=run.inference_round_work(frame,2,partition,weighting,199,local)
    assert sum(r['rows'] for r in rows)==1200
    assert all(r['computed_rounds_per_row']==389 if r['level'] in local else r['computed_rounds_per_row']==199 for r in rows)
    assert sum(r['total_row_rounds'] for r in rows)==199*1200+190*1197
    with pytest.raises(ValueError,match='eligible'):run.inference_round_work(frame,2,partition,weighting,199,{})


def test_preserved_source_gates_audit_present_models_without_fits(tmp_path):
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path));fp={'fixture':'preserved gate inference only'}
    with (patch('lightgbm.LGBMClassifier.fit',side_effect=AssertionError('No old gate fit')),
            patch('sklearn.linear_model.LogisticRegression.fit',side_effect=AssertionError('No old gate fit'))):
        result=run.source_miniature_gate(c,fp)
    assert result['preserved_sampling_fits']==9 and result['preserved_older_fits']==8
    assert result['new_production_miniature_fits']==0
    old=read_config(tmp_path/'older_miniature_audit.json')
    assert old['saved_models_audited'] in [0,8] and old['new_model_fits']==0


def test_source_gate_failure_retained_and_retry_rejected(tmp_path,monkeypatch):
    c={'output_dir':str(tmp_path)}
    monkeypatch.setattr(run,'_source_miniature_gate',lambda *a:(_ for _ in []).throw(ValueError('fixture numerical failure')))
    with pytest.raises(ValueError,match='numerical'):run.source_miniature_gate(c,{})
    failure=tmp_path/'failures'/'source-miniature.json';original=failure.read_bytes()
    with pytest.raises(ValueError,match='Prior'):run.source_miniature_gate(c,{})
    assert failure.read_bytes()==original


@pytest.mark.parametrize('change',['absent','partial_sampling','partial_older','changed_original_receipt'])
def test_source_gates_use_exports_when_optional_caches_are_wholly_absent(tmp_path,monkeypatch,change):
    root=Path.cwd();c=dict(read_config(run.CONFIG),output_dir=str(tmp_path/'own'))
    (tmp_path/'configs').symlink_to(root/'configs',target_is_directory=True)
    (tmp_path/'reports').symlink_to(root/'reports',target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    if change=='partial_sampling':
        ancestor=read_config(c['parent_config']);seed=read_config(ancestor['parent_config'])
        p=Path(next(iter(read_config(Path(seed['report_dir'])/'miniature_gate.json')['files'])))
        p.parent.mkdir(parents=True);p.write_bytes(b'partial fixture')
    elif change in ['partial_older','changed_original_receipt']:
        fusion=read_config(c['fusion_source_config']);receipt=read_config(Path(fusion['report_dir'])/'source_miniature_gate.json')
        p=Path(receipt['source'])
        if change=='partial_older':
            exported=read_config(str(p).replace('outputs/','reports/',1));p=Path(exported['results'][0]['artifact'])
        else:
            # Simulate an older receipt naming an optional original, while
            # retaining the actual immutable exported receipt and reports.
            inherited=Path(fusion['report_dir'])/'source_miniature_gate.json'
            p=Path(str(p).replace('reports/','outputs/',1));native_read=run.read_config
            alternate=dict(receipt,source=str(p))
            monkeypatch.setattr(run,'read_config',lambda path:alternate if Path(path)==inherited else native_read(path))
        p.parent.mkdir(parents=True);p.write_bytes(b'changed or partial fixture')
    with (patch('lightgbm.LGBMClassifier.fit',side_effect=AssertionError('No new source gate fit')),
            patch('sklearn.linear_model.LogisticRegression.fit',side_effect=AssertionError('No new source gate fit'))):
        if change=='absent':
            r=run.source_miniature_gate(c,{'fixture':'fresh copy'})
            assert r['optional_sampling_artifacts_present']==0 and r['optional_older_artifacts_present']==0
            assert r['new_production_miniature_fits']==0
        else:
            with pytest.raises(ValueError,match='Partial|Present optional'):run.source_miniature_gate(c,{})
            assert (tmp_path/'own/failures/source-miniature.json').exists()


@pytest.mark.parametrize('change',['absent','present','changed','extra'])
def test_historical_snapshot_accepts_whole_absence_and_checks_all_present_files(tmp_path,change):
    folder=tmp_path/'old';p=folder/'record.json';folder.mkdir()
    original={str(p):'not needed when wholly absent'}
    if change!='absent':
        p.write_text('original');original[str(p)]=digest(p)
        if change=='changed':p.write_text('changed')
        if change=='extra':(folder/'extra.json').write_text('undeclared')
    c={'historical_output_roots':[str(folder)]}
    if change in ['changed','extra']:
        with pytest.raises(ValueError):run.present_history(c,original)
    else:
        assert run.present_history(c,original)==({} if change=='absent' else original)
