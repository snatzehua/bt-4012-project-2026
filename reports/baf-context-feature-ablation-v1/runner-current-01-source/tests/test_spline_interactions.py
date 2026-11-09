"""Pair audits plus two explicitly counted tiny missing-cache reproduction fits."""
import copy
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from baf.common import read_config, write_json, digest
from baf.interaction_inputs import InteractionInputs, PAIRS, PopulationMoments, storage
from baf.interaction_report import (audit_statistics, audit_inference, reconstructed_matrix,
                                    independent_objective, expected_factors)
from baf import interaction_study as study
from baf.spline_inputs import AdditiveInputs, NUMERIC
from baf.newton_spline_study import miniature_frame


def fixture(degenerate=False):
    x, _ = miniature_frame()
    if not degenerate:
        rng = np.random.default_rng(128)
        for col in NUMERIC:
            x[col] = rng.normal(size=len(x))
    else:
        x['credit_risk_score'] = 123.
        x['current_address_months_count'] = np.nan
        x['income'] = np.where(np.arange(len(x)) % 7, 0., 1.)
    return x


@pytest.mark.parametrize('degenerate', [False, True])
@pytest.mark.parametrize('chunk_rows', [1, 7, 8192])
def test_independent_population_moments_basis_and_column_order(degenerate, chunk_rows):
    x = fixture(degenerate)
    inp = InteractionInputs(chunk_rows).fit(x)
    z = inp.transform(x)
    assert sparse.isspmatrix_csr(z)
    assert inp.metadata()['added_dimension'] <= 108
    assert [(b['left'], b['right']) for b in inp.pair_blocks] == list(PAIRS)
    statistics=audit_statistics(inp, x)
    assert statistics['training_rows'] == len(x)
    assert statistics['accumulator_bits'] == np.finfo(np.longdouble).bits
    np.testing.assert_allclose(z.toarray(), reconstructed_matrix(inp, x), atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(z[:, :inp.main_dimension].toarray(),
                                  AdditiveInputs('spline').fit(x).transform(x).toarray())
    for b in inp.pair_blocks:
        actual = z[:, b['start']:b['stop']].toarray()
        inactive = np.asarray(b['inactive'])
        np.testing.assert_array_equal(actual[:, inactive], 0.)
        np.testing.assert_allclose(actual[:, ~inactive].mean(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(actual[:, ~inactive].var(axis=0), 1., atol=1e-12)
    assert storage(z)['csr_bytes'] == z.data.nbytes + z.indices.nbytes + z.indptr.nbytes


@pytest.mark.parametrize('degenerate', [False, True])
def test_saved_features_invariant_to_labels_batch_permutation_and_future_extremes(tmp_path, degenerate):
    x = fixture(degenerate); inp = InteractionInputs(11).fit(x)
    metadata = copy.deepcopy(inp.metadata())
    future = x.iloc[:12].copy()
    future.loc[future.index[0], 'income'] = np.nan
    future.loc[future.index[1], 'device_os'] = 'UNSEEN'
    for col in ['credit_risk_score', 'current_address_months_count', 'customer_age']:
        future.loc[future.index[2], col] = 1e20
    z = inp.transform(future).toarray()
    for i in range(len(future)):
        np.testing.assert_array_equal(inp.transform(future.iloc[[i]]).toarray(), z[[i]])
    changed = future.copy(); changed.iloc[1:] = x.iloc[20:31].to_numpy()
    np.testing.assert_array_equal(inp.transform(changed).toarray()[0], z[0])
    np.testing.assert_array_equal(inp.transform(future.iloc[::-1]).toarray(), z[::-1])
    with pytest.raises(TypeError):
        InteractionInputs().fit(x, np.zeros(len(x)))
    assert inp.metadata() == metadata
    path = tmp_path / 'inputs.joblib'; joblib.dump(inp, path)
    restored = joblib.load(path)
    assert restored.metadata() == metadata
    np.testing.assert_array_equal(restored.transform(future).toarray(), z)
    # A deterministic analytical model checks inference without fitting an estimator.
    model = LogisticRegression(**study.newton.solver_parameters(study.recipes()[4]))
    model.classes_ = np.array([0, 1]); model.n_features_in_ = len(inp.names)
    model.coef_ = np.linspace(-.1, .1, len(inp.names))[None, :]
    model.intercept_ = np.array([-.2])
    p = audit_inference(inp, model, future)
    np.testing.assert_allclose(p, expit(z @ model.coef_[0] - .2), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(p[[0]], audit_inference(inp, model, future.iloc[[0]]), atol=1e-12, rtol=1e-12)
    if degenerate:
        for b in inp.pair_blocks:
            if b['marginal_degenerate']:
                np.testing.assert_array_equal(z[:, b['start']:b['stop']], 0.)
    assert next(r for r in inp.diagnostics(future) if r['feature'] == 'device_os')['unseen_category'] == 1
    assert len(inp.pair_diagnostics(future)) == 3


def test_chunk_bound_is_observed_and_chunk_partition_does_not_change_features():
    x = pd.concat([fixture()] * 55, ignore_index=True)
    seen = []; original = InteractionInputs.centered_product
    def observe(self, part, block):
        seen.append(len(part)); return original(self, part, block)
    with patch.object(InteractionInputs, 'centered_product', observe):
        a = InteractionInputs().fit(x); za = a.transform(x)
    assert max(seen) == 8192 and len(x) > 8192
    b = InteractionInputs(257).fit(x)
    np.testing.assert_allclose(za.toarray(), b.transform(x).toarray(), atol=1e-12, rtol=1e-10)
    assert storage(za)['estimated_hessian_bytes'] == 8 * (za.shape[1] + 1) ** 2


def test_stable_moments_and_no_near_zero_inactivity_threshold():
    a = np.array([[1e12 + j, 1e-14 * j, 3.] for j in range(20)])
    moments = PopulationMoments(3)
    for part in np.array_split(a, 4):
        moments.update(part)
    np.testing.assert_allclose(moments.mean, a.mean(axis=0), atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(moments.variance, a.var(axis=0), atol=1e-30, rtol=1e-10)
    np.testing.assert_array_equal(moments.constant, [False, False, True])
    assert moments.variance[1] > 0


@pytest.mark.parametrize('chunk_rows', [0, 8193, -1, 1.5, True])
def test_invalid_chunk_size_rejected(chunk_rows):
    with pytest.raises(ValueError):
        InteractionInputs(chunk_rows)


@pytest.mark.parametrize('change', ['empty', 'label', 'order', 'missing', 'infinity'])
def test_invalid_inference_schema_rejected(change):
    x = fixture(); inp = InteractionInputs().fit(x); future = x.iloc[:4].copy()
    if change == 'empty': future = future.iloc[:0]
    elif change == 'label': future['fraud_bool'] = 0
    elif change == 'order': future = future[future.columns[::-1]]
    elif change == 'missing': future = future.drop(columns='income')
    else: future.loc[future.index[0], 'income'] = np.inf
    with pytest.raises(ValueError):
        inp.transform(future)


@pytest.mark.parametrize('change', ['mean', 'scale', 'inactive', 'names', 'spline'])
def test_independent_auditor_rejects_corrupted_statistics_or_marginal(change):
    x = fixture(); inp = InteractionInputs().fit(x)
    if change == 'mean': inp.pair_blocks[0]['left_mean'][0] += .01
    elif change == 'scale': inp.pair_blocks[0]['scale'][0] *= 2
    elif change == 'inactive': inp.pair_blocks[0]['inactive'][0] = not inp.pair_blocks[0]['inactive'][0]
    elif change == 'names': inp.names[-1] += '_corrupt'
    else: inp.additive.splines['income'].bsplines_[0].c[0, 0] += .01
    with pytest.raises(ValueError):
        audit_statistics(inp, x)


def test_finite_declaration_and_all_contrast_factors():
    c = read_config(study.CONFIG); study.guard(c)
    assert len(study.SOURCES) == len(set(study.SOURCES)) == 43
    assert c['candidates'] == study.recipes() and len(study.recipes()) == 8
    rows = study.contrast_factors()
    assert Counter(r[0] for r in rows) == {'original operating reference': 144,
        'policy minus own pooled': 24, 'interaction minus additive': 12,
        'balanced minus unweighted': 12, 'larger C minus smaller C': 12}
    for label, n, k, ref, rk in rows:
        if label == 'original operating reference':
            assert rk == study.ORIGINAL_POLICIES[ref]
    assert c['original_reference_policies'][study.RECENT_FUSION] == 'latest'
    assert c['expected_counts']['all_scores'] == 39


@pytest.mark.parametrize('C', [.1, 1.])
def test_additive_cache_dispatch_does_not_depend_on_old_c01_member_list(tmp_path, C):
    c = dict(output_dir=str(tmp_path))
    spec = next(s for s in study.recipes() if s['representation']=='spline' and s['C']==C)
    ps = study.cache_paths(c, spec, 4, 'validation')
    assert len(ps)==3 and ps[-1]==study.newton.model_path(c,'validation',4,spec['id'])
    assert '-reference.joblib' not in str(ps[-1])


@pytest.mark.parametrize('change', ['pair', 'quota', 'count', 'source_paths', 'candidate'])
def test_configuration_contract_rejects_undeclared_changes(change):
    c=read_config(study.CONFIG)
    if change=='pair':c['interaction_pairs'][0]=['customer_age','income']
    elif change=='quota':c['review_quota']=.01
    elif change=='count':c['expected_counts']['new_development_fits']=11
    elif change=='source_paths':c['source_paths']=c['source_paths'][:-1]
    else:c['candidates'][0]['C']=.2
    with pytest.raises(ValueError):study.guard(c)


@pytest.mark.parametrize('name', [s['id'] for s in study.strategies()])
@pytest.mark.parametrize('policy', study.POLICIES)
def test_every_winner_type_has_exact_bounded_matched_roles(name, policy):
    roles = study.choices_for(name, policy)
    spec = next((s for s in study.recipes() if s['id'] == name), None)
    assert len(roles) == (9 if spec else 6)
    assert roles['recent_fusion'] == [study.RECENT_FUSION, 'latest']
    assert roles['same_model'] == [name, 'pooled']
    if spec:
        for role, changed in [('matched_representation', 'representation'),
                              ('matched_weight', 'class_weight'), ('matched_C', 'C')]:
            other = next(s for s in study.recipes() if s['id'] == roles[role][0])
            assert roles[role][1] == policy
            assert {k for k in ['representation', 'class_weight', 'C'] if spec[k] != other[k]} == {changed}


def analytical_fit(matrix, labels, spec, threads, seed):
    """Identical predictor pairs with opposite labels have a zero optimum."""
    model = LogisticRegression(**study.newton.solver_parameters(spec, seed))
    model.classes_ = np.array([0, 1]); model.n_features_in_ = matrix.shape[1]
    model.coef_ = np.zeros((1, matrix.shape[1])); model.intercept_ = np.zeros(1)
    model.n_iter_ = np.array([1])
    record = dict(**study.newton.objective_gradient(model, matrix, labels, spec),
                  fallback_count=0, fallback_function_sha256=study.newton.fallback_identity(),
                  iterations=1, warnings=[], estimated_hessian_bytes=8 * (matrix.shape[1] + 1) ** 2)
    study.newton.check_numerical(model, [], record, record)
    return model, [], record


def cache_fixture(tmp_path):
    x = fixture().iloc[:80]
    x = x.iloc[np.repeat(np.arange(80), 2)].reset_index(drop=True)
    frame = x.assign(month=np.where(np.arange(160) < 120, 3, 4), fraud_bool=np.tile([0, 1], 80))
    c = dict(output_dir=str(tmp_path), threads=1, seed=2026)
    return c, {'analytical_fixture': True}, frame, x, study.recipes()[4]


def test_checked_interaction_artifact_roundtrip_and_independent_objective(tmp_path):
    c, fp, frame, x, spec = cache_fixture(tmp_path)
    with patch.object(study.newton, 'fit_checked', side_effect=analytical_fit) as fit:
        r = study.fit_interaction(c, fp, frame, x, spec, 4)
        assert fit.call_count == 1
    with patch.object(study.newton, 'fit_checked', side_effect=AssertionError('no new fit')):
        assert study.fit_interaction(c, fp, frame, x, spec, 4) == r
    assert r['numerical']['gradient_max_abs'] < 1e-12
    assert r['resources']['dimension'] == r['transformed_dimension']
    assert r['numerical']['fallback_count'] == 0


@pytest.mark.parametrize('change', ['partial', 'prediction', 'record', 'coefficients', 'gradient', 'fallback'])
def test_saved_numerical_and_cache_corruption_rejected_without_refit(tmp_path, change):
    c, fp, frame, x, spec = cache_fixture(tmp_path)
    with patch.object(study.newton, 'fit_checked', side_effect=analytical_fit):
        study.fit_interaction(c, fp, frame, x, spec, 4)
    rp, pp = study.paths(c, 'validation', 4, spec['id'])
    mp = study.model_path(c, 'validation', 4, spec['id'])
    r = read_config(rp)
    if change == 'partial': mp.unlink()
    elif change == 'prediction': pp.write_bytes(pp.read_bytes() + b'corrupt')
    elif change == 'record': r['train_rows'] += 1
    elif change == 'coefficients':
        a = joblib.load(mp); a['model'].coef_[0, 0] = .1; joblib.dump(a, mp)
        r['model_sha256'] = digest(mp)
    elif change == 'gradient': r['numerical']['gradient'][0] += .1
    else: r['numerical']['fallback_count'] = 1
    write_json(rp, r)
    with patch.object(study.newton, 'fit_checked', side_effect=AssertionError('no refit')):
        with pytest.raises(ValueError):
            study.fit_interaction(c, fp, frame, x, spec, 4)


def test_failed_gate_keeps_evidence_and_never_retries_or_changes_recipe(tmp_path):
    c, fp, frame, x, spec = cache_fixture(tmp_path)
    error = study.newton.NumericalGate('declared gradient failure', {'gradient_max_abs': .5})
    with patch.object(study.newton, 'fit_checked', side_effect=error) as fit:
        with pytest.raises(ValueError, match='gradient'):
            study.fit_interaction(c, fp, frame, x, spec, 4)
        assert fit.call_count == 1
    rp, _ = study.paths(c, 'validation', 4, spec['id'])
    failure = read_config(tmp_path / 'failures' / (rp.stem + '.json'))
    assert failure['numerical']['gradient_max_abs'] == .5
    assert failure['spec'] == spec and failure['resources']['dimension'] > 108
    with patch.object(study.newton, 'fit_checked', side_effect=AssertionError('no retry')):
        with pytest.raises(ValueError, match='Prior failed gate'):
            study.fit_interaction(c, fp, frame, x, spec, 4)


def test_independent_weighted_objective_matches_matrix_formula():
    x = fixture(); inp = InteractionInputs().fit(x); z = inp.transform(x)
    labels = (np.arange(len(x)) % 5 == 0).astype(int)
    for spec in study.recipes()[4:]:
        model = LogisticRegression(**study.newton.solver_parameters(spec))
        model.coef_ = np.linspace(-.1, .1, z.shape[1])[None, :]
        model.intercept_ = np.array([-.2])
        actual = independent_objective(model, inp, x, labels, spec)
        expected = study.newton.objective_gradient(model, z, labels, spec)
        for key in ['objective', 'gradient', 'gradient_max_abs', 'weight_sum']:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=1e-12)


def individual_cache_fixture(tmp_path):
    x,y=miniature_frame();frame=x.assign(month=np.where(np.arange(len(x))<120,3,4),fraud_bool=y)
    c=dict(output_dir=str(tmp_path/'own'),threads=1,seed=2026,
           solver=study.newton.solver_parameters(study.recipes()[0]),
           reference_source_config='configs/calibrated-f2-policy.json')
    for key,folder in [('newton_source_config','newton'),('recent_source_config','recent'),('source_config','tree')]:
        cfg=tmp_path/(folder+'.json');write_json(cfg,{**c,'output_dir':str(tmp_path/folder)})
        c[key]=str(cfg)
    c.update(newton_fit_fingerprint={'original_newton':True},newton_analysis_fingerprint={'analysis':True},
             recent_source_fingerprint={'recent':True},source_fingerprint={'tree':True})
    return c,{'own_fixture':True},frame,x


@pytest.mark.parametrize('kind',['additive','tree'])
def test_wholly_absent_individual_recipe_reproduces_once_in_new_destination(tmp_path,kind):
    # One actual estimator fit per invocation of this parametrized case.
    c,fp,frame,x=individual_cache_fixture(tmp_path)
    spec=(next(s for s in study.recipes() if s['representation']=='spline' and s['C']==1. and s['class_weight'] is None)
          if kind=='additive' else read_config(study.CONFIG)['references'][0])
    r=study.prepare_base(c,fp,frame,x,spec,4)
    assert not r['reused'] and r['fingerprint']==fp
    assert all(p.is_file() for p in study.cache_paths(c,spec,4,'validation'))
    for key in ['newton_source_config','recent_source_config','source_config']:
        root=read_config(c[key])['output_dir']
        assert not Path(root).exists()
    with patch.object(study.newton,'fit_new',side_effect=AssertionError('no repeat fit')), \
         patch.object(study.prior,'reproduce_reference',side_effect=AssertionError('no repeat fit')):
        again=study.prepare_base(c,fp,frame,x,spec,4)
        assert again['prediction_sha256']==r['prediction_sha256']


@pytest.mark.parametrize('where',['own','origin'])
def test_partial_individual_cache_rejected_without_fitting_or_fallback(tmp_path,where):
    c,fp,frame,x=individual_cache_fixture(tmp_path);spec=study.recipes()[2]
    target=c if where=='own' else read_config(c['newton_source_config'])
    rp=study.cache_paths(target,spec,4,'validation')[0];write_json(rp,{'partial':True})
    with patch.object(study.newton,'fit_new',side_effect=AssertionError('no substitution fit')):
        with pytest.raises(ValueError,match='Partial'):
            study.prepare_base(c,fp,frame,x,spec,4)


def test_independent_contrast_factors_equal_complete_declaration():
    c=read_config(study.CONFIG)
    assert expected_factors(c)==study.contrast_factors()


def test_recent_fusion_absent_source_derives_exact_margin_without_logit_of_tree(tmp_path):
    c,fp,frame,x=individual_cache_fixture(tmp_path);c['epsilon']=1e-6
    ids=np.array([4,8,10]);labels=np.array([0,1,0]);p=np.array([0.,.3,1.]);g=np.array([-12.,.6,15.])
    for n,scores in [('spline_c0p1_unweighted',p),(study.FUSION,g)]:
        rp,pp=study.paths(c,'validation',4,n);pp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(pp,row_ids=ids,labels=labels,scores=scores);write_json(rp,{'fixture':n})
    r=study.recent_score(c,fp,4);actual=study.prediction(c,'validation',4,study.RECENT_FUSION)
    q=np.clip(p,1e-6,1-1e-6)
    np.testing.assert_allclose(actual['scores'],.25*(np.log(q)-np.log(1-q))+.75*g,atol=1e-12,rtol=1e-12)
    assert not r['reused'] and len(r['dependencies'])==2
    assert study.recent_score(c,fp,4)==r
    rp,pp=study.paths(c,'validation',4,study.RECENT_FUSION);pp.unlink()
    with pytest.raises(ValueError,match='Partial'):study.recent_score(c,fp,4)


def test_surface_rows_are_independently_reconstructed_for_best_development_interaction(tmp_path):
    from baf.interaction_report import surfaces
    x=fixture();inp=InteractionInputs().fit(x);spec=study.recipes()[4]
    model=LogisticRegression(**study.newton.solver_parameters(spec));model.coef_=np.linspace(-.1,.1,len(inp.names))[None,:]
    model.intercept_=np.array([0.]);model.classes_=np.array([0,1]);model.n_features_in_=len(inp.names)
    c=dict(output_dir=str(tmp_path/'out'));mp=study.model_path(c,'validation',6,spec['id'])
    mp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(dict(inputs=inp,model=model,spec=spec),mp)
    summary=pd.DataFrame([dict(candidate=study.RECENT_FUSION,f2=.4),dict(candidate=spec['id'],f2=.3)])
    result=surfaces(c,{'selected_candidate':study.RECENT_FUSION},summary,tmp_path)
    t=pd.read_csv(tmp_path/'pair_surfaces.csv')
    assert result['rows']==1323 and len(t)==1323 and result['candidate']==spec['id']
    assert len(t[['left','right','left_index','right_index']].drop_duplicates())==1323
    assert (tmp_path/'pair_surfaces.png').is_file()


@pytest.mark.parametrize('winner',[study.FUSION,'interaction_c0p1_unweighted'])
def test_paired_uncertainty_accepts_bounded_roles_and_preserves_aliases(winner):
    from baf.direct_ranking_report import development,intervals
    from baf.score_fusion_report import audit_intervals
    from baf.recall_weighted_metrics import learn,measure,decisions
    data=dict(row_ids=np.arange(6),labels=np.array([0,1,0,1,0,1]),scores=np.array([.1,.9,.2,.8,.3,.7]))
    roles=study.choices_for(winner,'pooled')
    h={n:{m:{**copy.deepcopy(data),'row_ids':data['row_ids']+6*m} for m in [4,5,6]} for n,k in roles.values()}
    rows=[]
    for m in [5,6]:
        for n,k in set(map(tuple,roles.values())):
            rule=learn({t:d for t,d in h[n].items() if t<m},k,m)
            rows.append(dict(candidate=n,policy=k,eval_month=m,parameters=__import__('json').dumps(rule),**measure(data,rule)))
    c=dict(bootstrap_seed=1027,bootstrap_replicates=5)
    dint,ddraws=development(c,{'choices':roles},pd.DataFrame(rows),h)
    final={a:data for a in roles};masks={a:decisions(data,learn(h[n],k,7)) for a,(n,k) in roles.items()}
    fint,fdraws=intervals(final,masks,np.random.default_rng(1027),5,'7_consumed')
    audit_intervals(dint,ddraws,fint,fdraws,roles,5)
    assert len(dint)==3*(len(roles)-1) and len(fint)==len(roles)-1 and len(ddraws)==10
    np.testing.assert_array_equal(fdraws['same_model'],0.)


@pytest.mark.parametrize('name,interactions,additives',[
    ('interaction_c0p1_unweighted',3,0),('spline_c1p0_balanced',1,2),(study.RECENT_FUSION,0,0)])
def test_tighter_transfer_fit_bounds_depend_on_winner_type_and_cache_availability(name,interactions,additives):
    c=read_config(study.CONFIG);roles=study.choices_for(name,'pooled')
    def available(paths):
        text=str(paths[-1]);return 'spline_c1p0_' not in text
    with patch.object(study.tree,'cache_state',side_effect=available):
        bounds=study.final_fit_bounds(c,roles)
    assert bounds['max_new_interaction_fits']==interactions
    assert bounds['max_reproduced_additive_fits']==additives
    assert bounds['max_reproduced_tree_fits']==0
    with patch.object(study.tree,'cache_state',return_value=False):
        missing=study.final_fit_bounds(c,roles)
    assert missing['max_new_interaction_fits']==interactions
    assert missing['max_reproduced_tree_fits']==3
    assert missing['max_reproduced_additive_fits']<=4


def test_complete_export_and_independent_table_audit_with_analytical_models(tmp_path):
    """All keys/CSV types/contrasts/diagnostics, without any estimator fitting."""
    from baf import interaction_report as report
    from baf.interaction_inputs import storage
    x=fixture().iloc[:56];x=x.iloc[np.repeat(np.arange(56),2)].reset_index(drop=True)
    frame=x.assign(month=np.repeat(np.arange(7),16),fraud_bool=np.tile([0,1],56))
    c=read_config(study.CONFIG);c['output_dir']=str(tmp_path/'own')
    source=tmp_path/'empty-recent.json';write_json(source,{'output_dir':str(tmp_path/'empty')})
    c['recent_source_config']=str(source);fp={'analytical_full_export_fixture':True}
    for m in [4,5,6]:
        train=frame.index[frame.month<m];ev=frame.index[frame.month==m]
        inputs={'spline':AdditiveInputs('spline').fit(x.loc[train]),'interaction':InteractionInputs().fit(x.loc[train])}
        for spec in study.recipes():
            inp=inputs[spec['representation']];matrix=inp.transform(x.loc[train])
            model,messages,numerical=analytical_fit(matrix,frame.loc[train,'fraud_bool'].to_numpy(),spec,1,2026)
            rp,pp=study.paths(c,'validation',m,spec['id']);mp=study.model_path(c,'validation',m,spec['id'])
            mp.parent.mkdir(parents=True,exist_ok=True);pp.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(dict(inputs=inp,model=model,spec=spec),mp)
            np.savez_compressed(pp,row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=np.full(len(ev),.5))
            write_json(rp,dict(**study.expected_record(frame,spec,m,'validation',fp),model_sha256=digest(mp),
                prediction_sha256=digest(pp),feature_metadata=inp.metadata(),warnings=[],iterations=1,
                numerical=numerical,solver_params=study.newton.solver_parameters(spec),transformed_dimension=len(inp.names),
                matrix_nnz=matrix.nnz,matrix_density=matrix.nnz/(matrix.shape[0]*matrix.shape[1]),
                fit_seconds=0.,peak_process_rss_bytes=1,reused=spec['representation']=='spline',resources=storage(matrix)))
        for spec in c['references']+c['presets']:
            rp,pp=study.paths(c,'validation',m,spec['id']);pp.parent.mkdir(parents=True,exist_ok=True)
            y=frame.loc[ev,'fraud_bool'].to_numpy();scores=.2+.6*y
            np.savez_compressed(pp,row_ids=ev.to_numpy(),labels=y,scores=scores)
            write_json(rp,dict(fixture=True,candidate=spec['id'],prediction_sha256=digest(pp)))
        study.tree.ensure_score(c,fp,c['fusion_spec'],m)
        study.recent_score(c,fp,m)
    cf=study.crossfit(c);tables,main,pairs=study.tables(c,cf,frame,x)
    out=Path(c['output_dir'])
    for n,t in tables.items():t.to_csv(out/(n+'.csv'),index=False)
    write_json(out/'feature_blocks.json',main);write_json(out/'interaction_blocks.json',pairs)
    # Only the known full-BAF reference constants are inapplicable to this tiny fixture.
    with patch.object(report,'audit_reference_values',return_value=None):
        result=report.audit_tables(c,frame,x)
        assert result['operating_rows']==72 and result['contrasts']==204 and result['feature_rows']==1392
        assert result['pair_rows']==72 and len(main)==24 and len(pairs)==12
        corrupted=pd.read_csv(out/'contrasts.csv');corrupted.loc[0,'reference_policy']='rank_fraction'
        corrupted.to_csv(out/'contrasts.csv',index=False)
        with pytest.raises(ValueError,match='contrast factors'):report.audit_tables(c,frame,x)
