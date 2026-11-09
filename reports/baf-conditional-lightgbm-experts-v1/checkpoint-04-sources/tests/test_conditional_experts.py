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
    c = read_config(run.CONFIG); p = read_config(run.PLANNING)
    assert models.strategies() == p['strategies']
    assert [s['id'] for s in models.strategies()] == report.independent_strategy_ids()
    assert models.contrast_factors() == p['contrasts']
    assert len(models.contrast_factors()) == 738
    assert len({str(v) for v in models.contrast_factors()}) == 738
    assert c['preserve_sha256'] == p['mandatory_preservation_sha256']
    assert len(c['source_paths']) == 58 and len(c['preserve_sha256']) == 1053
    assert c['parent_config'] == run.parent.CONFIG
    assert not any(k in c for k in ['review_quota', 'precision_floor', 'primary_fraction'])


@pytest.mark.parametrize('name,policy', list(product(report.independent_strategy_ids(), ['pooled','latest','rank_fraction'])))
def test_all_winner_policy_cases_preserve_roles_and_bounds(name, policy):
    p = read_config(run.PLANNING)
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
    p=read_config(run.PLANNING)
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
