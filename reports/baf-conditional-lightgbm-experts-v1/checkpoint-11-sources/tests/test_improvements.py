import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, matthews_corrcoef, balanced_accuracy_score

from baf.common import features, read_config
from baf.engineering import NativeInputs, correct_prior, engineer, recency_weights, undersample_indices
from baf.improvement_metrics import choose_policies, paired_intervals, threshold_stats, weighted_ap
from baf.improvements import crossfit_policies, fingerprint, guard_preservation, load_selection
from baf.logistic_ablation import run as run_logistic_ablation


@pytest.fixture
def rows():
    return pd.read_csv('data/raw/baf/Base.csv',nrows=10)


def test_engineering_row_local_and_excludes_labels(rows):
    x = features(rows)
    result = engineer(x)
    pd.testing.assert_frame_equal(result.loc[[2]],engineer(x.loc[[2]]))
    assert len(result.columns) > len(x.columns)
    assert result.loc[2,'velocity_6h_to_24h'] == pytest.approx(x.loc[2,'velocity_6h']/(x.loc[2,'velocity_24h']+1))
    assert not {'month','fraud_bool','device_fraud_count'} & set(result)
    with pytest.raises(ValueError,match='Excluded'):
        engineer(rows)


def test_native_training_only_levels_medians_and_unknowns():
    train = pd.DataFrame({'payment_type':['A','B'],'value':[2.,np.nan]})
    future = pd.DataFrame({'payment_type':['C'],'value':[np.nan]})
    adapter=NativeInputs().fit(train)
    assert adapter.transform(future).payment_type.isna().all()
    assert adapter.levels['payment_type'] == ['A','B']
    cat=NativeInputs(catboost=True).fit(train)
    assert cat.transform(future).value.iloc[0] == 2
    assert cat.transform(future).payment_type.iloc[0] == 'C'
    assert cat.medians['value'] == 2


def test_undersampling_eligibility_class_counts_seed():
    y=np.array([1,1]+[0]*30+[1,0])
    eligible=np.arange(32)
    a=undersample_indices(y,eligible,3,2026)
    np.testing.assert_array_equal(a,undersample_indices(y,eligible,3,2026))
    assert len(a)==8 and y[a].sum()==2 and set(a)<=set(eligible)


def test_prior_odds_and_recency_weights():
    # A sample-prevalence prediction maps back to the population prevalence.
    assert correct_prior([.2],.01,.2)[0] == pytest.approx(.01)
    p=correct_prior([.01,.2,.9],.01,.2)
    assert np.all(np.diff(p)>0)
    w=recency_weights([0,2,4],2)
    assert w.mean()==pytest.approx(1)
    assert w[2]/w[1]==pytest.approx(2)
    with pytest.raises(ValueError):
        recency_weights([1,2],0)


def test_operating_metrics_against_sklearn():
    y=np.array([0,0,1,1,1,0])
    scores=np.array([.1,.8,.7,.9,.2,.3])
    pred=scores>=.5
    result=threshold_stats(y,scores,.5)
    assert result['f1']==pytest.approx(f1_score(y,pred))
    assert result['f2']==pytest.approx(fbeta_score(y,pred,beta=2))
    assert result['mcc']==pytest.approx(matthews_corrcoef(y,pred))
    assert result['balanced_accuracy']==pytest.approx(balanced_accuracy_score(y,pred))
    assert result['tp']+result['fn']==y.sum()
    assert result['tp']+result['fp']==result['flagged']


def test_threshold_policies_brute_force():
    y=np.array([1,0,1,0,0,1,0,1])
    scores=np.array([.9,.8,.7,.6,.5,.4,.3,.2])
    result=choose_policies(y,scores)
    for beta in [1,2]:
        candidates=[fbeta_score(y,scores>=t,beta=beta) for t in np.unique(scores)]
        assert fbeta_score(y,scores>=result[f'f{beta}'],beta=beta)==pytest.approx(max(candidates))
    assert threshold_stats(y,scores,result['precision30'])['precision']>=.3
    assert threshold_stats(y,scores,result['recall50'])['recall']>=.5
    impossible=choose_policies([0,0,1],[.9,.8,.7],precision_target=.9)
    assert threshold_stats([0,0,1],[.9,.8,.7],impossible['precision30'])['flagged']==0


def test_weighted_ap_matches_explicit_bootstrap_with_score_ties():
    y=np.array([1,0,1,0,1])
    scores=np.array([.8,.8,.5,.2,.2])
    weights=np.array([0,2,3,0,1])
    order=np.argsort(-scores,kind='stable')
    ends=np.r_[np.flatnonzero(scores[order][:-1]!=scores[order][1:]),len(y)-1]
    repeated=np.repeat(np.arange(len(y)),weights)
    assert weighted_ap(y,order,ends,weights)==pytest.approx(average_precision_score(y[repeated],scores[repeated]))


def test_paired_intervals_fixed_thresholds_and_identical_cases():
    data=dict(labels=np.array([1,0,1,0,1,0]),scores=np.array([.9,.8,.6,.5,.3,.1]),row_ids=np.arange(6))
    thresholds={'base':{'f1':.5},'same':{'f1':.5}}
    result=paired_intervals({'base':data,'same':data},thresholds,'base',20,2026)
    assert {r['metric'] for r in result}=={'average_precision','recall_at_1pct','f1_policy'}
    assert all(r['difference']==r['ci_low']==r['ci_high']==0 for r in result)
    thresholds['same']['f1']=.9
    changed=paired_intervals({'base':data,'same':data},thresholds,'base',20,2026)
    f1=next(r for r in changed if r['metric']=='f1_policy')
    assert f1['difference']==pytest.approx(threshold_stats(data['labels'],data['scores'],.9)['f1']-threshold_stats(data['labels'],data['scores'],.5)['f1'])
    wrong={**data,'row_ids':data['row_ids'][::-1]}
    with pytest.raises(ValueError,match='identical'):
        paired_intervals({'base':data,'same':wrong},thresholds,'base',2,2026)


def test_crossfit_uses_only_prior_prediction_months(monkeypatch):
    calls=[]
    def fake(config,phase,month,name):
        calls.append(month)
        return dict(labels=np.array([0,1]),scores=np.array([.1,.9]),row_ids=np.array([month*2,month*2+1]))
    monkeypatch.setattr('baf.improvements.prediction',fake)
    config=dict(validation_months=[4,5,6],precision_target=.3,recall_target=.5)
    result=crossfit_policies(config,['a'])
    assert calls==[4,5,4,5,6]
    assert set(result[result.eval_month==5].threshold_source_months)=={'4'}
    assert set(result[result.eval_month==6].threshold_source_months)=={'4,5'}


@pytest.mark.parametrize('stale',[False,True])
def test_selection_missing_or_stale_rejected_before_data_access(tmp_path,stale):
    config=dict(output_dir=str(tmp_path),data_sha256='example')
    path=tmp_path/'config.json'
    path.write_text(json.dumps(config))
    if stale:
        (tmp_path/'selection.json').write_text(json.dumps(dict(fingerprint={'invalid':True},data_sha256='example')))
    with pytest.raises(ValueError,match='Stale|Freeze'):
        load_selection(path)


def test_original_preservation_guard(tmp_path):
    config=read_config('configs/improvements.json')
    config['output_dir']=str(tmp_path)
    guard_preservation(config)
    config['preserve_sha256']['baf/common.py']='bad'
    with pytest.raises(ValueError,match='Original-study'):
        guard_preservation(config)


def test_candidate_coverage_and_frozen_source_identity():
    config=read_config('configs/improvements.json')
    assert len(config['candidates'])==17
    assert len({c['id'] for c in config['candidates']})==17
    assert set(config['validation_months'])=={4,5,6}
    assert config['exploratory'] and config['test_month']==7
    assert 'baf/engineering.py' in fingerprint('configs/improvements.json')['sources']


@pytest.mark.parametrize('change',[
    {'C':1.0}, {'family':'lightgbm'}, {'engineered':True}, {'id':'baseline_logistic'},
])
def test_adaptive_diagnostic_scope_guards_before_loading_data(tmp_path,monkeypatch,change):
    main_path='configs/improvements.json'
    main=read_config(main_path)
    identity=fingerprint(main_path)
    fake=[dict(candidate=name,eval_month=m,fingerprint=identity)
          for name in ['baseline_logistic','logistic_engineered'] for m in main['validation_months']]
    monkeypatch.setattr('baf.logistic_ablation.records',lambda *args:fake)
    def forbidden(*args):
        raise AssertionError('Invalid diagnostic reached dataset loading')
    monkeypatch.setattr('baf.logistic_ablation.load_data',forbidden)
    config=read_config('configs/logistic-ablation.json')
    config['candidate'].update(change)
    path=tmp_path/'diagnostic.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        run_logistic_ablation(path)
