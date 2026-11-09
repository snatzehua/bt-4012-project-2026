"""Methodological checks for the separate fixed context-feature study."""
import copy
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from lightgbm import LGBMClassifier

from baf import context_ablation as scores
from baf import shallow_catboost_study as predecessor
from baf.common import read_config,write_json,digest
from baf.improvements import prediction

PLAN=read_config('openspec/changes/test-context-feature-ablation-f2/planning-evidence.json')


@pytest.mark.parametrize('spec',[s for s in scores.recipes() if s['new_fit']],ids=lambda s:s['id'])
def test_six_reduced_formats_fit_inference_reuse_and_reject_corruption(tmp_path,spec):
    c=dict(output_dir=str(tmp_path/'study'));fp={'fixture':'reduced native format'}
    frame,base=predecessor.miniature_data(scores.RAW_COLUMNS)
    r=scores.fit_model(c,fp,frame,base,spec,2,miniature=True)
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
