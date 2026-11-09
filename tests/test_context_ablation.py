"""Methodological checks for the separate fixed context-feature study."""
import copy
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from lightgbm import LGBMClassifier

from baf import context_ablation as scores
from baf import context_ablation_report as audit
from baf import shallow_catboost_study as predecessor
from baf.common import read_config,write_json,digest,features
from baf.improvements import prediction

PLAN=read_config('openspec/changes/test-context-feature-ablation-f2/planning-evidence.json')


@pytest.mark.parametrize('spec',[s for s in scores.recipes() if s['new_fit']],ids=lambda s:s['id'])
def test_six_reduced_formats_fit_inference_reuse_and_reject_corruption(tmp_path,spec):
    c=dict(output_dir=str(tmp_path/'study'));fp={'fixture':'reduced native format'}
    frame,base=predecessor.miniature_data(scores.RAW_COLUMNS)
    r=scores.fit_model(c,fp,frame,base,spec,2,miniature=True)
    assert audit.audit_reduced(c,fp,frame,base,r)['saved_native_inference_verified'] is True
    from baf import context_ablation_study as run
    work,params,pre=run.model_work(c,[r]);assert len(work)==len(params)==len(pre)==1
    assert int(work.iloc[0].model_bytes)==r['model_bytes'] and int(work.iloc[0].train_row_rounds)==r['train_row_rounds']
    assert params[0]['actual_rounds']==r['parameters']['actual_rounds'] and pre[0]['preprocessing']==r['preprocessing']
    ps=scores.model_paths(c,'validation',2,spec['id']);pipe=joblib.load(ps[2])['model'];x=scores.select(base,spec);ev=frame.index[frame.month==2]
    saved=prediction(c,'validation',2,spec['id']);assert len(x.columns)==spec['raw_columns']
    assert r['preprocessing']['medians']['prev_address_months_count']==0
    assert not set(spec['removed_columns'])&set(x)
    assert not any(removed in name for removed in spec['removed_columns'] for name in r['preprocessing']['transformed_names'])
    for actual in [scores.margins(pipe,x.loc[ev].iloc[::-1])[::-1],
                   np.concatenate([scores.margins(pipe,x.loc[ev].iloc[:17]),scores.margins(pipe,x.loc[ev].iloc[17:])]),
                   scores.margins(pipe,x.loc[ev].iloc[:1]),
                   scores.margins(pipe,__import__('pandas').concat([x.loc[ev],x.loc[ev].iloc[:1].assign(name_email_similarity=999)]))[:len(ev)]]:
        np.testing.assert_allclose(actual,saved['scores'][:len(actual)],atol=1e-12,rtol=0)
    changed_frame=frame.copy();changed_frame.loc[ev,'fraud_bool']=1-changed_frame.loc[ev,'fraud_bool']
    np.testing.assert_allclose(scores.margins(pipe,scores.select(features(changed_frame),spec).loc[ev]),saved['scores'],atol=1e-12,rtol=0)
    changed=base.copy();changed[spec['removed_columns']]=999999
    assert scores.preprocessing(pipe,scores.select(changed,spec).loc[frame.month<2])==r['preprocessing']
    np.testing.assert_allclose(scores.margins(pipe,scores.select(changed,spec).loc[ev]),saved['scores'],atol=1e-12,rtol=0)
    with patch.object(LGBMClassifier,'fit',side_effect=AssertionError('No refit')):
        assert scores.fit_model(c,fp,frame,base,spec,2,miniature=True)==r
    original={p:p.read_bytes() for p in ps}
    for key,value in [('train_rows',r['train_rows']+1),('class_costs',{'0':999.,'1':999.}),('warnings',['invalid']),('train_row_rounds',0)]:
        bad=copy.deepcopy(r);bad[key]=value;write_json(ps[0],bad)
        with pytest.raises(ValueError):scores.audit_model(c,fp,frame,base,spec,2,miniature=True)
        ps[0].write_bytes(original[ps[0]])
    ps[2].write_bytes(original[ps[2]]+b'corrupt')
    with pytest.raises(ValueError):scores.audit_model(c,fp,frame,base,spec,2,miniature=True)
    ps[2].write_bytes(original[ps[2]]);ps[1].unlink()
    with patch.object(LGBMClassifier,'fit',side_effect=AssertionError('No fallback fit')):
        with pytest.raises(ValueError,match='Partial'):scores.fit_model(c,fp,frame,base,spec,2,miniature=True)


@pytest.mark.parametrize('case',PLAN['winner_policy_cases'],ids=lambda c:c['winner']+'-'+c['policy'])
def test_all_winner_policy_cases_preserve_exact_roles_work_and_dependencies(case):
    roles=scores.choices_for(case['winner'],case['policy']);dag=scores.dependencies([n for n,k in roles.values()])
    assert roles==case['roles'] and dag==case['dependencies']
    assert len(dag['new_ablation_bases'])==case['new_fit_bound']
    assert len(dag['full_scores'])==case['physical_score_bound']
    assert 8+len(dag['new_ablation_bases'])==case['model_object_bound']
    assert 9+len(dag['new_ablation_bases'])==case['model_file_bound']
    assert 4*(len(roles)-1)==case['paired_interval_rows']
    assert len({tuple(v) for v in roles.values()})==case['unique_operating_pairs']


def test_prospective_strategy_recipe_factor_and_schema_contract():
    assert scores.recipes()==PLAN['recipes'] and scores.strategies()==PLAN['strategies']
    assert scores.contrast_factors()==PLAN['contrasts']
    assert scores.dependencies([s['id'] for s in scores.strategies()])==PLAN['development_dependencies']
    assert len(scores.RAW_COLUMNS)==29 and not set(scores.REMOVED['drop_velocity'])&set(scores.REMOVED['drop_counts'])
    for case in PLAN['winner_policy_cases']:
        s=scores.lookup(case['winner'])
        if s['kind']=='reference':assert len(case['roles'])==8
        else:
            assert len(case['roles'])==19
            assert case['roles']['alpha_a000'][0]==scores.ANCHOR
            assert case['roles']['alpha_a100'][0]==scores.pure(s['representation'],s['weighting'])


@pytest.mark.parametrize('damage',['reordered','missing','extra','altered_recipe','duplicate_cost'])
def test_undeclared_schema_or_recipe_rejected_before_fitting(damage):
    _,base=predecessor.miniature_data(scores.RAW_COLUMNS);spec=copy.deepcopy(scores.recipes()[2])
    if damage=='reordered':base=base.iloc[:,::-1]
    elif damage=='missing':base=base.drop(columns=['velocity_4w'])
    elif damage=='extra':base['fraud_bool']=1
    elif damage=='altered_recipe':spec['removed_columns']=[]
    else:spec['sample_weight']=[1,2]
    with patch.object(LGBMClassifier,'fit',side_effect=AssertionError('Fit forbidden')):
        with pytest.raises(ValueError):scores.select(base,spec)


@pytest.mark.parametrize('alpha',[0,1,-.1,.75,True,np.nan])
def test_only_declared_blend_interior_weights(alpha):
    d=dict(row_ids=np.arange(3),labels=np.array([0,1,0]),scores=np.array([-2.,1.,3.]))
    with pytest.raises(ValueError):scores.blend(d,d,alpha)


@pytest.mark.parametrize('kind',['bagged_balanced','bagged_unweighted','spline','xgboost','lightgbm_balanced','lightgbm_unweighted'])
def test_six_original_formats_and_full_native_controls(tmp_path,kind):
    import pandas as pd
    from baf import context_ablation_study as run
    c=dict(read_config(run.CONFIG),output_dir=str(tmp_path/'source'),threads=1);fp={'fixture':'original required format'};old=run.OLD
    if kind.startswith('bagged_'):
        frame,base=old.parent.synthetic_data();spec=next(s for s in old.scores.recipes() if s['weighting']==kind[7:]);month=2
    elif kind.startswith('lightgbm_'):
        c['threads']=6;frame,base=predecessor.miniature_data(scores.RAW_COLUMNS)
        for cat in scores.CATEGORICAL:base[cat]=base[cat].fillna('a')
        for cat in scores.CATEGORICAL:frame[cat]=base[cat]
        name='baseline_balanced' if kind=='lightgbm_balanced' else 'baseline_lgbm';spec=next(s for s in old.reference_specs(c) if s['id']==name);month=2
    else:
        base,y=old.parent.previous.newton.miniature_frame();frame=base.assign(month=np.where(np.arange(len(base))<120,3,4),fraud_bool=y);month=4
        name='spline_c1p0_balanced' if kind=='spline' else 'xgb_d4_w1';spec=next(s for s in old.reference_specs(c) if s['id']==name)
    with patch.object(run,'required_sources',return_value=[spec]),patch.object(old,'source_origins',return_value=[]):r=run.prepare_source(c,fp,frame,base,spec,month)
    assert not r.get('reused')
    with patch.object(predecessor,'required_sources',return_value=[spec]):
        audit.audit_reference(c,frame,base,r)
        work,params,pre=run.model_work(c,[r]);assert len(work)==1 and not params and not pre
        assert int(work.iloc[0].model_bytes)>0 and int(work.iloc[0].fitted_tree_count)>=0
    if kind.startswith('lightgbm_'):
        name='context_full_balanced' if kind=='lightgbm_balanced' else 'context_full_unweighted'
        with patch.object(run,'required_sources',return_value=[spec]),patch.object(LGBMClassifier,'fit',side_effect=AssertionError('Full control cannot refit')):
            raw=run.full_margin(c,fp,frame,base,name,month,miniature=True)
            assert audit.audit_full_margin(c,fp,frame,base,raw)['original_model_used'] is True
            assert run.full_margin(c,fp,frame,base,name,month,miniature=True)==raw
            rp,_=run.paths(c,'validation',month,name);oldbytes=rp.read_bytes();bad=copy.deepcopy(raw);bad['source_candidate']='wrong';write_json(rp,bad)
            with pytest.raises(ValueError):run.full_margin(c,fp,frame,base,name,month,miniature=True)
            rp.write_bytes(oldbytes)
    ps=old.source_paths(c,spec,month,'validation');before={p:p.read_bytes() for p in ps}
    bad=copy.deepcopy(r);bad['train_rows']+=1;write_json(ps[0],bad)
    with patch.object(run,'required_sources',return_value=[spec]):
        with pytest.raises(ValueError):run.prepare_source(c,fp,frame,base,spec,month)
    ps[0].write_bytes(before[ps[0]])
    other=dict(c,output_dir=str(tmp_path/'copy'));otherfp={'fixture':'copied reference'}
    with patch.object(run,'required_sources',return_value=[spec]),patch.object(old,'source_origins',return_value=[(c,fp)]),patch.object(LGBMClassifier,'fit',side_effect=AssertionError('No refit')):
        copied=run.prepare_source(other,otherfp,frame,base,spec,month);assert copied['reused']
        assert copied['origin_artifacts']=={str(p):digest(p) for p in ps}
        assert run.prepare_source(other,otherfp,frame,base,spec,month)==copied
    assert all(p.read_bytes()==v for p,v in before.items());ps[1].unlink()
    with patch.object(run,'required_sources',return_value=[spec]),patch.object(old,'source_origins',side_effect=AssertionError('No fallback origin')):
        with pytest.raises(ValueError,match='Partial'):run.prepare_source(c,fp,frame,base,spec,month)


def test_only_declared_individual_source_and_reference_gate_can_execute():
    from baf import context_ablation_study as run
    c=read_config(run.CONFIG);fp={'fixture':'gate routing'}
    with patch.object(run.OLD,'prepare_source',side_effect=AssertionError('Unrelated source forbidden')):
        with pytest.raises(ValueError):run.prepare_source(c,fp,None,None,{'id':'unrelated'},4)
    with patch.object(run.parent,'source_miniature_gate',return_value='preserved') as gate:
        assert run.source_miniature_gate(c,fp)=='preserved';gate.assert_called_once_with(run.adapter_view(c),fp)


def test_only_original_project_context_entries_can_be_versioned():
    from baf import context_ablation_study as run
    c=copy.deepcopy(read_config(run.CONFIG));c['project_context_snapshots']['baf/common.py']={'original_sha256':'invalid','preserved_snapshot':'invalid'}
    with pytest.raises(ValueError,match='Only three'):run.verify_parent_indices(c)
    c=copy.deepcopy(read_config(run.CONFIG));c['project_context_snapshots']['README.md']['original_sha256']='changed'
    with pytest.raises(ValueError,match='Original context hash'):run.verify_parent_indices(c)


@pytest.fixture(scope='module')
def analytical():
    import pandas as pd
    from scipy.special import logit
    from baf import context_ablation_study as run
    c=read_config(run.CONFIG);n=35;rng=np.random.default_rng(2327)
    frame=pd.DataFrame(dict(month=np.repeat(np.arange(7),n),fraud_bool=np.tile((np.arange(n)%7==0).astype(int),7)))
    h={name:{} for name in run.physical_names()};old=run.OLD
    for m in [4,5,6]:
        ids=frame.index[frame.month==m].to_numpy();y=frame.loc[ids,'fraud_bool'].to_numpy()
        for name in scores.dependencies([])['old_bases']:h[name][m]=dict(row_ids=ids,labels=y,scores=rng.uniform(.01,.99,n))
        for name in old.derived_order([v[0] for v in scores.REFERENCE_ROLES.values()]):
            spec=old.derived_spec(name);items={v:h[v][m] for v in spec['members']}
            h[name][m]=old.parent.old_compose(spec,items) if spec['kind']=='anchor' else old.scores.compose(spec,items)
        # Model-free analytical fixture; genuine original native inference is tested separately.
        for spec in scores.recipes():
            values=rng.normal(size=n) if spec['new_fit'] else logit(h[spec['original_full_source']][m]['scores'])
            h[spec['id']][m]=dict(row_ids=ids,labels=y,scores=values)
        h['context_drop_both_balanced'][m]['scores'][:]=.7
        for spec in scores.strategies():
            if spec['kind']=='blend':h[spec['id']][m]=scores.blend(h[scores.ANCHOR][m],h[scores.pure(spec['representation'],spec['weighting'])][m],spec['alpha_numerator']/100)
    t=run.tables(c,frame,h);return c,frame,h,t


def test_analytical_complete_tables_and_csv_roundtrip_independently_reconstruct(analytical,tmp_path):
    c,frame,h,t=analytical;result=audit.audit_tables(c,frame,h,t)
    assert result['counts']=={k:PLAN['expected_counts'][k] for k in t}
    assert result['four_term_interactions']==24
    c=dict(c,output_dir=str(tmp_path))
    for n,df in t.items():df.to_csv(tmp_path/(n+'.csv'),index=False)
    assert audit.audit_tables(c,frame,h)['counts']==result['counts']


@pytest.mark.parametrize('table,column',[
    ('crossfit','f2'),('summary','f2'),('pooled','tp'),('contrasts','f2_difference'),
    ('physical_score_diagnostics','score_mean'),('score_diagnostics','unique_scores'),
    ('diversity','pearson'),('decision_overlap','fraud_both_flagged')])
def test_analytical_numeric_corruption_rejected(analytical,table,column):
    c,frame,h,t=analytical;t={k:v.copy(deep=True) for k,v in t.items()};index=t[table].index[t[table][column].notna()][0]
    t[table].loc[index,column]+=1
    with pytest.raises((ValueError,AssertionError)):audit.audit_tables(c,frame,h,t)


@pytest.mark.parametrize('table',['crossfit','summary','pooled','contrasts','physical_score_diagnostics','score_diagnostics','diversity','decision_overlap'])
@pytest.mark.parametrize('damage',['missing','duplicate'])
def test_analytical_exact_table_keys_rejected_on_corruption(analytical,table,damage):
    import pandas as pd
    c,frame,h,t=analytical;t={k:v.copy(deep=True) for k,v in t.items()}
    t[table]=t[table].iloc[1:].reset_index(drop=True) if damage=='missing' else pd.concat([t[table],t[table].iloc[:1]],ignore_index=True)
    with pytest.raises((ValueError,AssertionError)):audit.audit_tables(c,frame,h,t)


def test_analytical_four_term_removal_interaction_and_alias_overlap(analytical):
    import json
    c,frame,h,t=analytical;index=t['summary'].set_index(['candidate','policy']);g=t['contrasts'].query("contrast == 'removal_interaction'")
    assert len(g)==24
    for row in g.itertuples():
        terms=json.loads(row.term_details);assert [x['coefficient'] for x in terms]==[1,-1,-1,1]
        value=sum(x['coefficient']*index.loc[(x['candidate'],x['policy']),'f2'] for x in terms)
        assert row.f2_difference==pytest.approx(value,abs=1e-12,rel=0)
    same=t['decision_overlap'].query('candidate == reference');assert len(same)==36
    assert (same[['fraud_first_only','fraud_second_only','legitimate_first_only','legitimate_second_only']].to_numpy()==0).all()
    bad={k:v.copy(deep=True) for k,v in t.items()};bad['contrasts'].loc[g.index[0],'term_details']=json.dumps(json.loads(g.iloc[0].term_details)[:2],separators=(',',':'))
    with pytest.raises((ValueError,AssertionError)):audit.audit_tables(c,frame,h,bad)


@pytest.mark.parametrize('damage',['domain','ids','labels','nonfinite','month','policy'])
def test_analytical_domain_population_or_policy_damage_rejected(analytical,damage):
    import json
    c,frame,h,t=analytical;h=copy.deepcopy(h);t={k:v.copy(deep=True) for k,v in t.items()}
    if damage=='domain':h['baseline_balanced'][5]['scores'][0]=2
    elif damage=='ids':h[scores.ANCHOR][5]['row_ids'][0]=-100
    elif damage=='labels':h[scores.ANCHOR][5]['labels'][0]^=1
    elif damage=='nonfinite':h['context_drop_velocity_balanced'][6]['scores'][0]=np.inf
    elif damage=='month':h[scores.ANCHOR][7]=h[scores.ANCHOR].pop(4)
    else:
        rule=json.loads(t['crossfit'].loc[0,'parameters']);rule['source_months']=[5];t['crossfit'].loc[0,'parameters']=json.dumps(rule)
    with pytest.raises((ValueError,AssertionError)):audit.audit_tables(c,frame,h,t)


def test_analytical_parent_verification_allows_absent_optional_caches_only():
    from baf import context_ablation_study as run
    c=read_config(run.CONFIG);parent=read_config(run.parent.CONFIG)
    index=read_config(Path(parent['report_dir'])/'run_manifest.json')
    optional={name for name in index if run.optional_historical_entry(c,name)}
    assert optional and all(name.startswith('outputs/') for name in optional)
    assert not run.optional_historical_entry(c,'baf/common.py')
    assert not run.optional_historical_entry(c,parent['output_dir']+'/selection.json')
    original=Path.is_file
    with patch.object(Path,'is_file',lambda self:False if str(self) in optional else original(self)):
        run.verify_parent_indices(c)
