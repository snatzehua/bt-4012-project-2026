"""Finite declarations and checked fitting for the isolated interaction study.

These matrix-level interfaces do not dispatch interaction artifacts through the
older additive-only auditor. Existing study sources remain immutable.
"""
from __future__ import annotations

import itertools
import argparse
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .common import digest, read_config, write_json, load_data, features
from .focal_study import expected_record, now
from .improvements import paths, prediction
from .recall_extension import expected_population, order_summary
from .recall_weighted_metrics import POLICIES, aligned, measure, learn
from .improvement_metrics import confusion_metrics
from .interaction_inputs import InteractionInputs, PAIRS, MAX_CHUNK_ROWS, storage
from .spline_inputs import checked_scores
from .calibrated_policy import verify_table
from . import newton_spline_study as newton
from . import spline_fusion_study as prior
from . import score_fusion_study as tree

CONFIG = 'configs/interaction-spline-logistic.json'
FUSION = newton.FUSION
RECENT_FUSION = 'margin_blend_spline_c0p1_unweighted_a25'
ORIGINAL_POLICIES = {RECENT_FUSION: 'latest', FUSION: 'pooled',
                     'baseline_balanced': 'pooled', 'baseline_lgbm': 'pooled'}
SOURCES = prior.SOURCES + ['baf/interaction_inputs.py', 'baf/interaction_study.py',
                          'baf/interaction_report.py']
TABLES = ['crossfit.csv','summary.csv','pooled.csv','contrasts.csv','score_diagnostics.csv',
          'solver.csv','numerical.csv','feature_diagnostics.csv','pair_diagnostics.csv','coefficients.csv',
          'feature_blocks.json','interaction_blocks.json','resources.csv','declaration.json',
          'preservation.json','miniature.json','resource_preflight.json']


def recipes():
    return [dict(id=f'{rep}_c{str(C).replace(".", "p")}_{tag}', representation=rep,
                 C=C, class_weight=weight)
            for rep in ['spline', 'interaction'] for C in [.1, 1.]
            for tag, weight in [('unweighted', None), ('balanced', 'balanced')]]


def strategies():
    return ([dict(id=s['id'], kind='model', members=[s['id']]) for s in recipes()]
            + [dict(id=n, kind='reference', members=[n]) for n in ORIGINAL_POLICIES])


def contrast_factors():
    lookup = {(s['representation'], s['C'], s['class_weight']): s['id'] for s in recipes()}
    rows = []
    for s in strategies():
        for k in POLICIES:
            rows += [('original operating reference', s['id'], k, n, policy)
                     for n, policy in ORIGINAL_POLICIES.items()]
            if k != 'pooled':
                rows.append(('policy minus own pooled', s['id'], k, s['id'], 'pooled'))
    for C, w, k in itertools.product([.1, 1.], [None, 'balanced'], POLICIES):
        rows.append(('interaction minus additive', lookup['interaction', C, w], k,
                     lookup['spline', C, w], k))
    for rep, C, k in itertools.product(['spline', 'interaction'], [.1, 1.], POLICIES):
        rows.append(('balanced minus unweighted', lookup[rep, C, 'balanced'], k,
                     lookup[rep, C, None], k))
    for rep, w, k in itertools.product(['spline', 'interaction'], [None, 'balanced'], POLICIES):
        rows.append(('larger C minus smaller C', lookup[rep, 1., w], k,
                     lookup[rep, .1, w], k))
    if len(rows) != 204 or len(set(rows)) != 204:
        raise ValueError('Incomplete contrast declaration')
    return rows


def choices_for(name, policy):
    if name not in {s['id'] for s in strategies()} or policy not in POLICIES:
        raise ValueError('Unknown winner/policy')
    roles = dict(selected=[name, policy], recent_fusion=[RECENT_FUSION, 'latest'],
                 fusion=[FUSION, 'pooled'], balanced=['baseline_balanced', 'pooled'],
                 unweighted=['baseline_lgbm', 'pooled'], same_model=[name, 'pooled'])
    spec = next((s for s in recipes() if s['id'] == name), None)
    if spec is not None:
        changes = dict(matched_representation=('representation',
                       'spline' if spec['representation'] == 'interaction' else 'interaction'),
                       matched_weight=('class_weight', None if spec['class_weight'] == 'balanced' else 'balanced'),
                       matched_C=('C', .1 if spec['C'] == 1. else 1.))
        for role, (key, value) in changes.items():
            target = {**spec, key: value}
            match = next(s for s in recipes() if all(s[k] == target[k]
                         for k in ['representation', 'class_weight', 'C']))
            roles[role] = [match['id'], policy]
    return roles


def fingerprint(path=CONFIG):
    fp = prior.fingerprint(path)
    fp['sources'] = {p: digest(p) for p in SOURCES}
    return fp


def verify_hashes(hashes):
    prior.verify_hashes(hashes)


def guard(c):
    source = read_config(c['recent_source_config'])
    prior.guard(source)
    if prior.fingerprint(c['recent_source_config']) != c['recent_source_fingerprint']:
        raise ValueError('Recent fusion identity changed')
    verify_hashes(c['preserve_sha256'])
    if any(c['preserve_sha256'].get(p)!=h for p,h in source['preserve_sha256'].items()):
        raise ValueError('Mandatory ancestral preservation declaration changed')
    selections=[p for p in c['preserve_sha256'] if p.startswith('outputs/') and p.endswith('/selection.json')]
    if len(selections)!=12:raise ValueError('Twelve original selection identities are required')
    for p in selections:
        export=p.replace('outputs/','reports/',1)
        if export not in c['preserve_sha256'] or digest(export)!=digest(p):
            raise ValueError('Original/exported selection pair differs: '+p)
    for k in ['data_path', 'data_sha256', 'tree_params', 'reference_source_config',
              'epsilon', 'references', 'presets', 'fusion_spec', 'solver', 'gradient_limit',
              'feature_contract', 'validation_months', 'test_month', 'beta', 'seed', 'threads',
              'policies', 'bootstrap_replicates', 'bootstrap_seed', 'source_config',
              'source_fingerprint', 'newton_source_config', 'newton_fit_fingerprint',
              'newton_analysis_fingerprint']:
        if c[k] != source[k]:
            raise ValueError('Frozen setting changed: ' + k)
    expected = dict(candidates=recipes(), strategies=strategies(),
                    original_reference_policies=ORIGINAL_POLICIES,
                    interaction_pairs=[list(p) for p in PAIRS],
                    pair_contract=dict(center_marginals=True, center_tensor=True,
                        scale_tensor=True, population_weighting='unweighted',
                        inactive_scale=1., inactive_inference='force_zero',
                        constant_rule='exact_minimum_equals_maximum', chunk_rows=MAX_CHUNK_ROWS,
                        max_added_dimension=108, order='left-major/right-minor'),
                    output_dir='outputs/baf-interaction-spline-logistic-v1',
                    report_dir='reports/baf-interaction-spline-logistic-v1',
                    protocol_path='docs/interaction-spline-logistic-protocol.md')
    for k, value in expected.items():
        if c[k] != value:
            raise ValueError('Interaction declaration changed: ' + k)
    if c['source_paths'] != SOURCES or len(SOURCES) != 43 or len(set(SOURCES)) != 43:
        raise ValueError('Source closure changed')
    if c['expected_counts'] != dict(miniature_fits=8, new_development_fits=12,
            local_base_reuses=21, recent_derived_reuses=3, base_results=33, derived_results=6,
            all_scores=39, selectable_scores=36, strategies=12, forward_rows=72,
            macro_rows=36, pooled_rows=36, score_rows=36, contrasts=204, logistic_blocks=24,
            interaction_blocks=12, feature_diagnostic_rows=1392, pair_diagnostic_rows=72,
            development_draws=1000, consumed_draws=500, surface_rows=1323):
        raise ValueError('Finite output declaration changed')
    if any(k in c for k in ['primary_fraction', 'review_quota', 'precision_floor']):
        raise ValueError('Undeclared operating constraint')
    roots = sorted(set(source['historical_output_roots'] + [source['output_dir']]))
    if c['historical_output_roots'] != roots:
        raise ValueError('Historical roots changed')
    destinations = [Path(c[k]).resolve() for k in ['output_dir', 'report_dir']]
    for a, b in itertools.combinations(destinations + [Path(p).resolve() for p in roots], 2):
        if a in destinations or b in destinations:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError('Study destinations overlap')
    snapshot = Path(c['output_dir']) / 'preservation.json'
    if snapshot.exists():
        r = read_config(snapshot)
        if r['mandatory'] != c['preserve_sha256'] or r['roots'] != roots:
            raise ValueError('Preservation declaration changed')
        verify_hashes(r['optional_present'])
    else:
        initial = Path(c['output_dir']) / 'preimplementation_preservation.json'
        if initial.exists():verify_hashes(read_config(initial))
    gate=Path(c['output_dir'])/'miniature.json'
    if gate.exists():
        mini=read_config(gate)
        if mini['fit_count']!=8 or len(mini['results'])!=8:raise ValueError('Miniature fit count changed')
        verify_hashes({r['artifact']:r['sha256'] for r in mini['results']})


def initialize(config_path=CONFIG):
    """Freeze prospective execution identity only once implementation is ready."""
    c = read_config(config_path); guard(c)
    out = Path(c['output_dir']); fp = fingerprint(config_path)
    snapshot = out / 'preservation.json'
    if not snapshot.exists():
        present = {str(p): digest(p) for root in c['historical_output_roots']
                   for p in Path(root).rglob('*') if p.is_file()}
        write_json(snapshot, dict(mandatory=c['preserve_sha256'], optional_present=present,
                   roots=c['historical_output_roots'], created_at_utc=now()))
    declaration = dict(fingerprint=fp, config_path=str(config_path),
                       protocol_sha256=digest(c['protocol_path']),
                       preservation_sha256=digest(snapshot), candidates=c['candidates'],
                       strategies=c['strategies'], pairs=c['interaction_pairs'],
                       expected_counts=c['expected_counts'], month7_previously_consumed=True)
    path = out / 'declaration.json'
    if path.exists():
        r = read_config(path)
        if any(r.get(k) != v for k, v in declaration.items()):
            raise ValueError('Prospective identity changed')
    else:
        write_json(path, dict(**declaration, created_at_utc=now()))
    return c, fp


def model_path(c, phase, month, name):
    return newton.model_path(c, phase, month, name)


def audit_interaction(c, fp, frame, base, spec, month, phase='validation'):
    from .interaction_report import audit_statistics, audit_inference, independent_objective
    rp, pp = paths(c, phase, month, spec['id']); mp = model_path(c, phase, month, spec['id'])
    if not tree.cache_state([rp, pp, mp]):
        raise ValueError('Interaction model missing at audit')
    r = read_config(rp)
    expected = expected_record(frame, spec, month, phase, fp)
    if any(r.get(k) != v for k, v in expected.items()):
        raise ValueError('Interaction population/identity differs')
    if digest(pp) != r['prediction_sha256'] or digest(mp) != r['model_sha256']:
        raise ValueError('Interaction artifact changed')
    a = joblib.load(mp); inputs, model = a['inputs'], a['model']
    if (a['spec'] != spec or not isinstance(inputs, InteractionInputs)
            or inputs.metadata() != r['feature_metadata']):
        raise ValueError('Interaction metadata/recipe differs')
    if newton.signature(model) != newton.signature(LogisticRegression(**newton.solver_parameters(spec, c['seed']))):
        raise ValueError('Interaction solver parameters differ')
    if r['solver_params'] != newton.solver_parameters(spec, c['seed']):
        raise ValueError('Recorded interaction solver differs')
    train = frame.index[frame.month < month]
    d = prediction(c, phase, month, spec['id'])
    aligned(d, expected_population(frame, month))
    audit_statistics(inputs, base.loc[train])
    with threadpool_limits(limits=c['threads']):
        audit_inference(inputs, model, base.loc[d['row_ids']], d['scores'])
        computed = independent_objective(model, inputs, base.loc[train],
                                        frame.loc[train, 'fraud_bool'].to_numpy(), spec)
    newton.check_numerical(model, r['warnings'], r['numerical'], computed)
    if r['iterations'] != int(model.n_iter_.max()) or r['transformed_dimension'] != len(inputs.names):
        raise ValueError('Recorded numerical dimensions/iterations differ')
    tree.verify_origin(r)
    return r


def fit_interaction(c, fp, frame, base, spec, month, phase='validation',
                    allow_missing=True, inputs=None, matrix=None):
    if spec not in [s for s in recipes() if s['representation'] == 'interaction']:
        raise ValueError('Undeclared interaction recipe')
    rp, pp = paths(c, phase, month, spec['id']); mp = model_path(c, phase, month, spec['id'])
    if tree.cache_state([rp, pp, mp]):
        return audit_interaction(c, fp, frame, base, spec, month, phase)
    if not allow_missing:
        raise ValueError('Interaction model missing at audit')
    failure = Path(c['output_dir']) / 'failures' / (rp.stem + '.json')
    if failure.exists():
        raise ValueError('Prior failed gate retained; refusing automatic refit')
    train = frame.index[frame.month < month]; ev = frame.index[frame.month == month]
    start = time.perf_counter(); numerical = {}; resources = {}
    try:
        inp = InteractionInputs().fit(base.loc[train]) if inputs is None else inputs
        z = inp.transform(base.loc[train]) if matrix is None else matrix
        if z.shape != (len(train), len(inp.names)) or inp.chunk_rows != MAX_CHUNK_ROWS:
            raise ValueError('Interaction matrix reuse dimensions differ')
        if matrix is not None:
            for start_row in range(0, len(train), MAX_CHUNK_ROWS):
                part = base.loc[train[start_row:start_row + MAX_CHUNK_ROWS]]
                difference = z[start_row:start_row + len(part)] - inp.transform(part)
                if difference.nnz and not np.allclose(difference.data, 0., atol=1e-12, rtol=0):
                    raise ValueError('Reused matrix does not match saved predictor map')
        resources = storage(z)
        print(f'Fitting {rp.stem}: rows={len(train):,} dim={z.shape[1]}', flush=True)
        model, messages, numerical = newton.fit_checked(z, frame.loc[train, 'fraud_bool'].to_numpy(),
                                                       spec, c['threads'], c['seed'])
        with threadpool_limits(limits=c['threads']):
            scores = checked_scores(model, inp.transform(base.loc[ev]))
        mp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(model=model, inputs=inp, spec=spec), mp)
        pp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pp, row_ids=ev.to_numpy(), labels=frame.loc[ev, 'fraud_bool'].to_numpy(), scores=scores)
        d = prediction(c, phase, month, spec['id'])
        r = dict(**expected_record(frame, spec, month, phase, fp),
                 **measure(d, dict(kind='pooled', threshold=.5)), model_sha256=digest(mp),
                 prediction_sha256=digest(pp), warnings=messages, iterations=int(model.n_iter_.max()),
                 feature_metadata=inp.metadata(), transformed_dimension=z.shape[1], matrix_nnz=int(z.nnz),
                 matrix_density=z.nnz / (z.shape[0] * z.shape[1]), resources=resources,
                 numerical=numerical, fit_seconds=time.perf_counter() - start,
                 peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 solver_params=newton.solver_parameters(spec, c['seed']), reused=False)
        write_json(rp, r)
        return audit_interaction(c, fp, frame, base, spec, month, phase)
    except Exception as exc:
        write_json(failure, dict(**expected_record(frame, spec, month, phase, fp),
                   error=repr(exc), numerical=getattr(exc, 'diagnostics', numerical), resources=resources,
                   elapsed_seconds=time.perf_counter() - start,
                   peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   created_at_utc=now()))
        raise


def cache_paths(c, spec, month, phase):
    rp, pp = paths(c, phase, month, spec['id'])
    if spec.get('representation') in ['spline', 'interaction']:
        return [rp, pp, model_path(c, phase, month, spec['id'])]
    if spec['id'] == 'xgb_d4_w1':
        return [rp, pp, *tree.model_paths(c, phase, month, spec['id'])]
    return [rp, pp, tree.reference_model_path(c, phase, month, spec['id'])]


def audit_base(c, fp, frame, base, spec, month, phase):
    ps = cache_paths(c, spec, month, phase)
    if not tree.cache_state(ps):
        raise ValueError('Required base missing')
    r = read_config(ps[0]); prior.check_expected_record(r, frame, spec, month, phase, fp)
    if digest(ps[1]) != r['prediction_sha256']:
        raise ValueError('Prediction changed')
    aligned(prediction(c, phase, month, spec['id']), expected_population(frame, month))
    tree.verify_origin(r)
    if spec.get('representation') == 'interaction':
        return audit_interaction(c, fp, frame, base, spec, month, phase)
    if spec.get('representation') == 'spline':
        return newton.audit_new(c, fp, frame, base, spec, month, phase)
    if spec['id'] == 'xgb_d4_w1':
        return tree.audit_model(c, fp, frame, base, spec, month, phase)
    return tree.reference_inference(c, frame, base, spec, month, phase, r)


def prepare_base(c, fp, frame, base, spec, month, phase='validation', allow_missing=True):
    own = cache_paths(c, spec, month, phase)
    if tree.cache_state(own):
        return audit_base(c, fp, frame, base, spec, month, phase)
    if not allow_missing:
        raise ValueError('Required individual model missing')
    if spec.get('representation') == 'interaction':
        return fit_interaction(c, fp, frame, base, spec, month, phase)
    if spec.get('representation') == 'spline':
        origins = [(read_config(c['newton_source_config']), c['newton_fit_fingerprint']),
                   (read_config(c['recent_source_config']), c['recent_source_fingerprint'])]
    else:
        origins = [(read_config(c['source_config']), c['source_fingerprint'])]
    for source, source_fp in origins:
        origin = cache_paths(source, spec, month, phase)
        if not tree.cache_state(origin):
            continue
        r = audit_base(source, source_fp, frame, base, spec, month, phase)
        for src, dst in zip(origin[1:], own[1:]):
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
        r = dict(r, fingerprint=fp, reused=True,
                 origin_artifacts={str(p): digest(p) for p in origin})
        if spec.get('representation') == 'spline':
            r.update(original_fit_fingerprint=c['newton_fit_fingerprint'],
                     original_analysis_fingerprint=c['newton_analysis_fingerprint'])
        elif spec['id'] == 'xgb_d4_w1':
            r['model_path'] = str(own[2])
        else:
            r.update(reference_model_path=str(own[2]), reference_model_new_fit=False)
        write_json(own[0], r)
        return audit_base(c, fp, frame, base, spec, month, phase)
    if spec.get('representation') == 'spline':
        newton.fit_new(c, fp, frame, base, spec, month, phase)
    elif spec['id'] == 'xgb_d4_w1':
        tree.old.fit_model(c, fp, frame, base, spec, month, phase)
    else:
        prior.reproduce_reference(c, fp, frame, base, spec, month, phase)
    return audit_base(c, fp, frame, base, spec, month, phase)


def recent_score(c, fp, month, phase='validation', allow_missing=True):
    rp, pp = paths(c, phase, month, RECENT_FUSION)
    member = 'spline_c0p1_unweighted'
    a, b = [prediction(c, phase, month, n) for n in [member, FUSION]]
    aligned(a, b)
    p = a['scores']
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Recent fusion member is not probability-like')
    p = np.clip(p, c['epsilon'], 1-c['epsilon'])
    score = .25*(np.log(p)-np.log1p(-p)) + .75*b['scores']
    expected = dict(candidate=RECENT_FUSION, phase=phase, eval_month=month, fingerprint=fp,
                    alpha=.25, epsilon=c['epsilon'],
                    dependencies={n:{str(p):digest(p) for p in paths(c, phase, month, n)}
                                  for n in [member, FUSION]})
    if tree.cache_state([rp, pp]):
        r = read_config(rp); actual = prediction(c, phase, month, RECENT_FUSION); aligned(a, actual)
        if (any(r.get(k) != v for k,v in expected.items()) or digest(pp) != r['prediction_sha256']
                or not np.allclose(actual['scores'], score, atol=1e-12, rtol=1e-12)):
            raise ValueError('Recent fusion arithmetic/identity differs')
        verify_hashes(r['origin_artifacts'])
        return r
    if not allow_missing:
        raise ValueError('Recent fusion missing')
    source = read_config(c['recent_source_config']); sr, sp = paths(source, phase, month, RECENT_FUSION)
    origin = {}
    if tree.cache_state([sr, sp]):
        r = read_config(sr)
        spec = next(s for s in source['strategies'] if s['id'] == RECENT_FUSION)
        if r['fingerprint'] != c['recent_source_fingerprint'] or r['spec'] != spec or digest(sp) != r['prediction_sha256']:
            raise ValueError('Recent source identity differs')
        original = prediction(source, phase, month, RECENT_FUSION); aligned(a, original)
        if not np.allclose(original['scores'], score, atol=1e-12, rtol=1e-12):
            raise ValueError('Recent source composition differs')
        origin = {str(sr):digest(sr), str(sp):digest(sp)}
        for n, dep in r['dependencies'].items():
            parent = paths(source, phase, month, n)
            if digest(parent[0]) != dep['record'] or digest(parent[1]) != dep['prediction']:
                raise ValueError('Recent source parent changed')
            origin.update({str(p):digest(p) for p in parent})
    pp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pp, row_ids=a['row_ids'], labels=a['labels'], scores=score)
    write_json(rp, dict(**expected, prediction_sha256=digest(pp), reused=bool(origin),
                       origin_artifacts=origin))
    return recent_score(c, fp, month, phase, False)


def prepare(c, fp, frame, base, phase='validation', names=None, allow_missing=True):
    names = {s['id'] for s in strategies()} if names is None else set(names)
    needed = names & {s['id'] for s in recipes()}
    if RECENT_FUSION in names:
        needed.add('spline_c0p1_unweighted')
    records = []
    for m in c['validation_months'] if phase == 'validation' else [7]:
        for spec in c['references'] + c['presets'] + [s for s in recipes() if s['id'] in needed]:
            records.append(prepare_base(c, fp, frame, base, spec, m, phase, allow_missing))
        tree.ensure_score(c, fp, c['fusion_spec'], m, phase, allow_missing)
        recent_score(c, fp, m, phase, allow_missing)
    return records


def history(c):
    return {s['id']:{m:prediction(c, 'validation', m, s['id']) for m in c['validation_months']}
            for s in strategies()}


def crossfit(c):
    h = history(c); rows = []
    for m, n, k in itertools.product([5, 6], h, POLICIES):
        aligned(h[n][m], h[FUSION][m]); rule = learn({t:d for t,d in h[n].items() if t<m}, k, m)
        rows.append(dict(candidate=n, policy=k, eval_month=m,
                         source_months=','.join(map(str, rule['source_months'])),
                         parameters=json.dumps(rule, sort_keys=True), **measure(h[n][m], rule)))
    return pd.DataFrame(rows)


def tables(c, cf, frame, base):
    summary = order_summary(cf).reset_index(drop=True); lookup = summary.set_index(['candidate','policy'])
    contrast = []
    for label,n,k,ref,rk in contrast_factors():
        a,b = lookup.loc[(n,k)],lookup.loc[(ref,rk)]
        contrast.append(dict(contrast=label,candidate=n,policy=k,reference=ref,reference_policy=rk,
            **{v+'_difference':float(a[v]-b[v]) for v in ['f2','recall','precision','flagged_fraction','average_precision']}))
    pooled = []
    for (n,k),g in cf.groupby(['candidate','policy']):
        t=g[['tn','fp','fn','tp']].sum()
        pooled.append(dict(candidate=n,policy=k,**confusion_metrics(*[int(t[v]) for v in ['tn','fp','fn','tp']]),
                           flagged_fraction=float((t.tp+t.fp)/g.n.sum())))
    scores=[]
    for n,items in history(c).items():
        for m,d in items.items():
            p=d['scores'];scores.append(dict(candidate=n,eval_month=m,n=len(p),score_min=float(p.min()),
                score_max=float(p.max()),score_mean=float(p.mean()),unique_scores=len(np.unique(p)),
                tied_fraction=1-len(np.unique(p))/len(p)))
    solver=[];numerical=[];fd=[];paird=[];coeff=[];main={};pairs={};resources=[]
    for m,spec in itertools.product(c['validation_months'],recipes()):
        n=spec['id'];r=read_config(paths(c,'validation',m,n)[0]);a=joblib.load(model_path(c,'validation',m,n))
        inp=a['inputs'];model=a['model'];key=f'{m}:{n}';reused=bool(r.get('reused'))
        solver.append(dict(candidate=n,eval_month=m,reused=reused,iterations=r['iterations'],
            transformed_dimension=r['transformed_dimension'],matrix_nnz=r['matrix_nnz'],
            matrix_density=r['matrix_density'],fit_seconds=r['fit_seconds'],
            peak_process_rss_bytes=r['peak_process_rss_bytes'],warnings=json.dumps(r['warnings']),
            solver_params=json.dumps(r['solver_params'],sort_keys=True)))
        numerical.append(dict(candidate=n,eval_month=m,reused=reused,**r['numerical']))
        resource_row=dict(candidate=n,eval_month=m,reused=reused,fit_seconds=r['fit_seconds'],
                          peak_process_rss_bytes=r['peak_process_rss_bytes'])
        if 'resources' in r:resource_row.update(r['resources'])
        else:
            z=inp.transform(base.loc[frame.index[frame.month<m]])
            resource_row.update(storage(z));del z
        resources.append(resource_row)
        main[key]=dict(reused=reused,metadata=(inp.additive.metadata() if spec['representation']=='interaction' else inp.metadata()))
        if spec['representation']=='interaction':pairs[key]=inp.metadata()
        for population,mask in [('training',frame.month<m),('evaluation',frame.month==m)]:
            x=base.loc[frame.index[mask]]
            fd += [dict(candidate=n,eval_month=m,population=population,**row) for row in inp.diagnostics(x)]
            if spec['representation']=='interaction':
                paird += [dict(candidate=n,eval_month=m,population=population,**row) for row in inp.pair_diagnostics(x)]
        coeff += [dict(candidate=n,eval_month=m,feature=name,coefficient=float(value))
                  for name,value in zip(inp.names,model.coef_[0])]
        coeff.append(dict(candidate=n,eval_month=m,feature='__intercept__',coefficient=float(model.intercept_[0])))
    return dict(crossfit=cf,summary=summary,pooled=pd.DataFrame(pooled),contrasts=pd.DataFrame(contrast),
                score_diagnostics=pd.DataFrame(scores),solver=pd.DataFrame(solver),numerical=pd.DataFrame(numerical),
                feature_diagnostics=pd.DataFrame(fd),pair_diagnostics=pd.DataFrame(paird),coefficients=pd.DataFrame(coeff),
                resources=pd.DataFrame(resources)), main, pairs


def miniature(config_path=CONFIG):
    """Run only once after all three implementation sources are ready to freeze."""
    c,fp=initialize(config_path);out=Path(c['output_dir']);gate=out/'miniature.json'
    failure=out/'failures'/'miniature.json'
    if failure.exists():raise ValueError('Prior miniature gate failure retained; refusing automatic refit')
    if gate.exists():
        r=read_config(gate)
        if r['fingerprint']!=fp or r['fit_count']!=8:raise ValueError('Stale miniature gate')
        return r
    rows=[]
    for degenerate in [False,True]:
        x,y=newton.miniature_frame()
        if not degenerate:
            rng=np.random.default_rng(128)
            from .spline_inputs import NUMERIC
            for col in NUMERIC:x[col]=rng.normal(size=len(x))
        else:
            x['credit_risk_score']=123.;x['current_address_months_count']=np.nan
        train=x.iloc[:120];future=x.iloc[120:].copy()
        future.loc[future.index[0],'income']=np.nan;future.loc[future.index[1],'device_os']='UNSEEN'
        future.loc[future.index[2],'customer_age']=1e12
        inp=InteractionInputs().fit(train);z=inp.transform(train)
        from .interaction_report import audit_statistics,audit_inference,independent_objective
        audit_statistics(inp,train)
        for spec in [s for s in recipes() if s['representation']=='interaction']:
            progress=dict(fingerprint=fp,completed_fits=len(rows),attempted_fits=len(rows)+1,spec=spec,degenerate=degenerate)
            write_json(out/'miniature_progress.json',progress)
            try:
                model,messages,numerical=newton.fit_checked(z,y[:120],spec,c['threads'],c['seed'])
                computed=independent_objective(model,inp,train,y[:120],spec)
                newton.check_numerical(model,messages,numerical,computed)
                p=audit_inference(inp,model,future)
                path=out/'miniature_models'/f'{int(degenerate)}-{spec["id"]}.joblib'
                path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(dict(inputs=inp,model=model,spec=spec),path)
                restored=joblib.load(path);audit_inference(restored['inputs'],restored['model'],future,p)
                rows.append(dict(spec=spec,degenerate=degenerate,numerical=numerical,resources=storage(z),
                                 artifact=str(path),sha256=digest(path)))
                write_json(out/'miniature_progress.json',dict(progress,completed_fits=len(rows)))
            except Exception as exc:
                write_json(failure,dict(**progress,error=repr(exc),numerical=getattr(exc,'diagnostics',{}),
                           previous_results=rows,created_at_utc=now()))
                raise
    r=dict(fingerprint=fp,fit_count=8,results=rows,created_at_utc=now())
    write_json(gate,r);return r


def resource_preflight(c, fp, frame, base):
    train=base.loc[frame.index[frame.month<6]];start=time.perf_counter()
    inp=InteractionInputs().fit(train)
    sample=inp.transform(train.iloc[:8192]);n=len(train);d=len(inp.names)
    # Include construction copies, model workspace and original dataframe allowance.
    matrix_upper=12*n*d+8*(n+1);workspace_upper=4*matrix_upper+8*(d+1)**2+4*1024**3
    if sys.platform=='darwin':
        ram=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
    else:
        import os
        ram=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    if workspace_upper>ram*.7:raise ValueError('Declared full interaction allocation exceeds host preflight allowance')
    r=dict(fingerprint=fp,host_ram_bytes=ram,training_rows=n,dimension=d,
           max_chunk_rows=8192,sample=storage(sample),matrix_upper_bytes=matrix_upper,
           workspace_upper_bytes=workspace_upper,allowance_fraction=.7,
           elapsed_seconds=time.perf_counter()-start,
           peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(Path(c['output_dir'])/'resource_preflight.json',r)
    return r


def preflight(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir'])
    mini=read_config(out/'miniature.json')
    if mini['fingerprint']!=fp or mini['fit_count']!=8:raise ValueError('Miniature gate missing/stale')
    for row in mini['results']:
        verify_hashes({row['artifact']:row['sha256']})
    path=out/'resource_preflight.json'
    if path.exists():
        r=read_config(path)
        if r['fingerprint']!=fp:raise ValueError('Resource identity changed')
        return r
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
    try:return resource_preflight(c,fp,frame,base)
    except Exception as exc:
        write_json(out/'failures'/'resource-preflight.json',dict(fingerprint=fp,error=repr(exc),created_at_utc=now()))
        raise


def prepare_references(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir'])
    if read_config(out/'resource_preflight.json')['fingerprint']!=fp:raise ValueError('Resource gate missing/stale')
    if (out/'selection.json').exists():raise ValueError('Development is frozen')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);records=[]
    for m in c['validation_months']:
        for spec in c['references']+c['presets']+[s for s in recipes() if s['representation']=='spline']:
            records.append(prepare_base(c,fp,frame,base,spec,m))
        tree.ensure_score(c,fp,c['fusion_spec'],m,'validation',True)
        recent_score(c,fp,m)
    recent=[read_config(paths(c,'validation',m,RECENT_FUSION)[0]) for m in c['validation_months']]
    result=dict(fingerprint=fp,base_results=len(records),base_reuses=sum(bool(r.get('reused')) for r in records),
        reproduced_base_fits=sum(not r.get('reused',False) for r in records),recent_derived_reuses=sum(r['reused'] for r in recent),
        recent_derived_results=3,tree_derived_results=3,created_at_utc=now())
    if len(records)!=21:raise ValueError('Required individual base set differs')
    write_json(out/'reference_preparation.json',result);return result


def validate(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Development is frozen')
    for gate in ['miniature.json','resource_preflight.json','reference_preparation.json']:
        if read_config(out/gate)['fingerprint']!=fp:raise ValueError('Missing/stale execution gate: '+gate)
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
    for m in c['validation_months']:
        specs=[s for s in recipes() if s['representation']=='interaction']
        missing=any(not tree.cache_state(cache_paths(c,s,m,'validation')) for s in specs)
        inp=matrix=None
        if missing:
            try:
                train=base.loc[frame.index[frame.month<m]];inp=InteractionInputs().fit(train);matrix=inp.transform(train)
            except Exception as exc:
                write_json(out/'failures'/f'validation-m{m}-representation.json',
                           dict(fingerprint=fp,error=repr(exc),created_at_utc=now()))
                raise
        for spec in specs:fit_interaction(c,fp,frame,base,spec,m,inputs=inp,matrix=matrix)
        del inp,matrix
    records=prepare(c,fp,frame,base,allow_missing=False)
    cf=crossfit(c);generated,main,pairs=tables(c,cf,frame,base)
    for name,table in generated.items():table.to_csv(out/(name+'.csv'),index=False)
    write_json(out/'feature_blocks.json',main);write_json(out/'interaction_blocks.json',pairs)
    counts=dict(base_results=len(records),base_reuses=sum(bool(r.get('reused')) for r in records),
        new_interaction_fits=sum(r['spec'].get('representation')=='interaction' for r in records),
        reproduced_base_fits=sum(not r.get('reused',False) and r['spec'].get('representation')!='interaction' for r in records),
        recent_derived_reuses=sum(read_config(paths(c,'validation',m,RECENT_FUSION)[0])['reused'] for m in c['validation_months']),
        total_scores=len(list((out/'predictions').glob('validation-*.npz'))),selectable_scores=36,
        derived_results=6,forward_rows=len(cf),miniature_fits=8)
    if counts['base_results']!=33 or counts['new_interaction_fits']!=12 or counts['total_scores']!=39 or len(cf)!=72:
        raise ValueError('Finite development counts differ')
    write_json(out/'validation_complete.json',dict(fingerprint=fp,created_at_utc=now(),
        files={p:digest(out/p) for p in TABLES},**counts))
    from .interaction_report import audit_development
    audit_development(config_path)
    print(generated['summary'].head(12).to_string(index=False),flush=True)


def required_final_specs(c, choices):
    names={n for n,k in choices.values()};needed=names & {s['id'] for s in recipes()}
    needed.add('spline_c0p1_unweighted')
    return c['references']+c['presets']+[s for s in recipes() if s['id'] in needed]


def final_fit_bounds(c, choices):
    specs=required_final_specs(c,choices);available={};counts=dict(interaction=0,additive=0,tree=0)
    for spec in specs:
        rep=spec.get('representation')
        if rep=='interaction':
            counts['interaction']+=1;available[spec['id']]=False;continue
        keys=['newton_source_config','recent_source_config'] if rep=='spline' else ['source_config']
        present=False
        for key in keys:
            source=read_config(c[key])
            if tree.cache_state(cache_paths(source,spec,7,'exploratory')):
                present=True;break
        available[spec['id']]=present
        if not present:counts['additive' if rep=='spline' else 'tree']+=1
    return dict(max_new_interaction_fits=counts['interaction'],max_reproduced_additive_fits=counts['additive'],
                max_reproduced_tree_fits=counts['tree'],available_original_cache=available,
                unique_dependency_count=len(specs))


def freeze(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    done=read_config(out/'validation_complete.json');audit=read_config(out/'independent_development_audit.json')
    if done['fingerprint']!=fp or audit['fingerprint']!=fp or audit['validation_sha256']!=digest(out/'validation_complete.json'):
        raise ValueError('Independent development audit missing/stale')
    verify_hashes({str(out/p):h for p,h in done['files'].items()})
    winner=pd.read_csv(out/'summary.csv').iloc[0];choices=choices_for(str(winner.candidate),str(winner.policy));h=history(c)
    final_specs=required_final_specs(c,choices)
    write_json(out/'selection.json',dict(created_at_utc=now(),fingerprint=fp,config=c,
        selected_candidate=str(winner.candidate),selected_policy=str(winner.policy),selected_development_f2=float(winner.f2),
        choices=choices,rules={a:learn(h[n],k,7) for a,(n,k) in choices.items()},fit_specs=final_specs,
        final_fit_bounds=final_fit_bounds(c,choices),
        validation_files=done['files'],independent_audit_sha256=digest(out/'independent_development_audit.json'),
        artifacts={str(p):digest(p) for folder in ['runs','predictions','models'] for p in (out/folder).glob('*') if p.is_file()},
        month7_previously_consumed=True,exploratory=True))
    print('Frozen',winner.candidate,winner.policy,float(winner.f2),'roles',len(choices),flush=True)


def load_selection(config_path=CONFIG):
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['config']!=c or s['fingerprint']!=fingerprint(config_path):raise ValueError('Frozen source/config/software changed')
    verify_hashes(s['artifacts']);verify_hashes({str(out/p):h for p,h in s['validation_files'].items()})
    if s['independent_audit_sha256']!=digest(out/'independent_development_audit.json'):raise ValueError('Frozen audit changed')
    winner=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0]
    if (s['selected_candidate'],s['selected_policy'])!=(str(winner.candidate),str(winner.policy)):
        raise ValueError('Frozen actual macro winner differs')
    if not np.isclose(s['selected_development_f2'],winner.f2,atol=1e-12,rtol=0):raise ValueError('Frozen F2 differs')
    if s['choices']!=choices_for(str(winner.candidate),str(winner.policy)) or s['fit_specs']!=required_final_specs(c,s['choices']):
        raise ValueError('Frozen bounded roles/dependencies differ')
    if s['final_fit_bounds']!=final_fit_bounds(c,s['choices']):raise ValueError('Frozen dependency fit bounds differ')
    h=history(c)
    if any(s['rules'][a]!=learn(h[n],k,7) for a,(n,k) in s['choices'].items()):raise ValueError('Frozen earlier-only rule differs')
    return c,s


def final_data(c,s):return {a:prediction(c,'exploratory',7,n) for a,(n,k) in s['choices'].items()}


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);sha=digest(out/'selection.json');ap=out/'exploratory_access.json'
    if not ap.exists():write_json(ap,dict(access_at_utc=now(),selection_sha256=sha,month7_previously_consumed=True))
    access=read_config(ap)
    if access['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Freeze/access chronology differs')
    frame=load_data(c);base=features(frame);names={n for n,k in s['choices'].values()}
    records=prepare(c,s['fingerprint'],frame,base,'exploratory',names)
    if {r['candidate'] for r in records}!={s['id'] for s in s['fit_specs']}:raise ValueError('Transfer dependency set differs')
    data=final_data(c,s);rows=[]
    for a,d in data.items():
        aligned(d,data['selected']);rule=s['rules'][a]
        rows.append(dict(role=a,candidate=s['choices'][a][0],policy=s['choices'][a][1],eval_month=7,
                         parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    fresh=[r for r in records if not r.get('reused',False)]
    counts=dict(base_results=len(records),base_reuses=len(records)-len(fresh),new_fits=len(fresh),
        new_interaction_fits=sum(r['spec'].get('representation')=='interaction' for r in fresh),
        reproduced_additive_fits=sum(r['spec'].get('representation')=='spline' for r in fresh),
        reproduced_tree_fits=sum(r['spec'].get('representation') not in ['spline','interaction'] for r in fresh),
        role_count=len(rows),total_scores=len(list((out/'predictions').glob('exploratory-*.npz'))))
    if counts['new_interaction_fits']>3 or counts['reproduced_additive_fits']>4 or counts['reproduced_tree_fits']>3:
        raise ValueError('Transfer work exceeds declared fresh-clone dependency bounds')
    for actual,bound in [('new_interaction_fits','max_new_interaction_fits'),
                         ('reproduced_additive_fits','max_reproduced_additive_fits'),
                         ('reproduced_tree_fits','max_reproduced_tree_fits')]:
        if counts[actual]>s['final_fit_bounds'][bound]:raise ValueError('Transfer exceeds frozen type/cache bound: '+actual)
    write_json(out/'transfer_complete.json',dict(selection_sha256=sha,table_sha256=digest(out/'exploratory.csv'),**counts))
    print(table[['role','candidate','policy','f2','precision','recall','tp','fp','fn']].to_string(index=False),flush=True)


def record_checks(config_path=CONFIG):
    """Receipt for a full passing pytest invocation, separate from search fits."""
    c=read_config(config_path);guard(c);dest=Path(c['report_dir']);xml=dest/'tests.xml'
    root=ET.parse(xml).getroot();cases=list(root.iter('testcase'))
    if not cases or list(root.iter('failure')) or list(root.iter('error')):
        raise ValueError('Full passing pytest XML evidence required')
    names=[t.attrib.get('name','') for t in cases]
    reproductions=[n for n in names if n.startswith('test_wholly_absent_individual_recipe_reproduces_once_in_new_destination[')]
    if len(reproductions)!=2:raise ValueError('Both actual missing-cache fixture fits must be verified')
    path=dest/'test_fit_counts.json'
    history=read_config(path).get('invocations',[]) if path.exists() else []
    sha=digest(xml)
    receipt=dict(junit_sha256=sha,tests_passed=len(cases),new_cache_reproduction_fits=2,
                 fingerprint=fingerprint(config_path),test_source_sha256=digest('tests/test_spline_interactions.py'),
                 created_at_utc=now())
    if not any(r.get('junit_sha256')==sha for r in history):history.append(receipt)
    result=dict(invocations=history,latest=receipt,
                new_cache_reproduction_fits=sum(r['new_cache_reproduction_fits'] for r in history),
                scope='new study missing-cache test cases only; historical test-module fits are not search fits')
    write_json(path,result);return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['initialize','miniature','preflight',
        'prepare_references','validate','freeze','exploratory','record_checks'])
    parser.add_argument('--config',default=CONFIG);args=parser.parse_args()
    result=globals()[args.command](args.config)
    print('Completed',args.command,flush=True)
