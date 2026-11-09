"""Fixed conditional fraud models with global preprocessing and class costs."""
from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import resource
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from . import bagged_anchor_scores as old_scores
from . import bagged_boosting as bagged
from .common import make_model, read_config, write_json, digest
from .focal_study import now
from .improvements import paths, prediction
from .recall_weighted import ids_digest
from .recall_weighted_metrics import POLICIES, aligned, check_data

WEIGHTINGS = ['balanced', 'unweighted']
PARTITIONS = {'payment_type': ['AA', 'AB', 'AC', 'AD', 'AE'],
              'housing_status': ['BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG']}
ALPHAS = [25, 50, 75, 100]
REFERENCE_ROLES = {'bagged_anchor': ['bagged_anchor_balanced_logit_a25', 'pooled'],
                   **old_scores.REFERENCE_ROLES}
MINIMUM = {'rows': 10000, 'fraud': 100, 'legitimate': 100}


def global200(weighting):
    if weighting not in WEIGHTINGS:
        raise ValueError('Undeclared weighting')
    return 'baseline_balanced' if weighting == 'balanced' else 'baseline_lgbm'


def capacity(weighting):
    global200(weighting)
    return 'capacity400_' + weighting


def expert(partition, weighting):
    global200(weighting)
    if partition not in PARTITIONS:
        raise ValueError('Undeclared partition')
    return 'expert_' + partition + '_' + weighting


def candidate(partition, weighting, alpha):
    expert(partition, weighting)
    if isinstance(alpha, bool) or alpha not in ALPHAS:
        raise ValueError('Undeclared conditional weight')
    return expert(partition, weighting) if alpha == 100 else (
        f'expert_blend_{partition}_{weighting}_a{alpha}')


def strategies():
    rows = [dict(id=n, kind='reference') for n, k in REFERENCE_ROLES.values()]
    rows += [dict(id=capacity(w), kind='capacity', weighting=w) for w in WEIGHTINGS]
    rows += [dict(id=candidate(p, w, a), kind='expert' if a == 100 else 'blend',
                  partition=p, weighting=w, alpha_numerator=a, alpha_denominator=100)
             for p, w, a in product(PARTITIONS, WEIGHTINGS, ALPHAS)]
    return rows


def lookup(name):
    spec = next((s for s in strategies() if s['id'] == name), None)
    if spec is None:
        raise ValueError('Undeclared strategy')
    return spec


def choices_for(name, policy):
    s = lookup(name)
    if policy not in POLICIES:
        raise ValueError('Undeclared policy')
    result = dict(selected=[name, policy], **REFERENCE_ROLES,
                  same_strategy=[name, 'pooled'])
    if s['kind'] == 'reference':
        return result
    w = s['weighting']; other = WEIGHTINGS[1 - WEIGHTINGS.index(w)]
    result['global200'] = [global200(w), policy]
    if s['kind'] == 'capacity':
        result['matched_weight'] = [capacity(other), policy]
        return result
    p, a = s['partition'], s['alpha_numerator']
    q = next(v for v in PARTITIONS if v != p)
    result.update(global400=[capacity(w), policy], expert_pure=[expert(p, w), policy],
                  matched_weight=[candidate(p, other, a), policy],
                  matched_partition=[candidate(q, w, a), policy])
    result.update({'alpha_a' + str(v): [candidate(p, w, v), policy] for v in ALPHAS})
    return result


def contrast_factors():
    result = []
    def add(kind, n, k, b, bk, **extra):
        result.append([kind, n, k, b, bk, [[a, v] for a, v in sorted(extra.items())]])
    for s, k in product(strategies(), POLICIES):
        for role, (b, bk) in REFERENCE_ROLES.items():
            add('fixed_reference', s['id'], k, b, bk, reference=role)
        if k != 'pooled':
            add('own_policy', s['id'], k, s['id'], 'pooled')
    for w, k in product(WEIGHTINGS, POLICIES):
        add('capacity400_vs200', capacity(w), k, global200(w), k)
    for p, w, k in product(PARTITIONS, WEIGHTINGS, POLICIES):
        for b, t in [(global200(w), 'expert_vs200'), (capacity(w), 'expert_vs400')]:
            add(t, expert(p, w), k, b, k)
        for a in ALPHAS[:-1]:
            n = candidate(p, w, a)
            for b, t in [(global200(w), 'blend_vs200'), (capacity(w), 'blend_vs400'),
                         (expert(p, w), 'blend_vspure')]:
                add(t, n, k, b, k)
        for lo, hi in combinations(ALPHAS, 2):
            add('higher_alpha', candidate(p, w, hi), k, candidate(p, w, lo), k,
                lower=lo, higher=hi)
    for p, a, k in product(PARTITIONS, ALPHAS, POLICIES):
        add('matched_weighting', candidate(p, 'balanced', a), k,
            candidate(p, 'unweighted', a), k)
    for w, a, k in product(WEIGHTINGS, ALPHAS, POLICIES):
        add('matched_partition', candidate('housing_status', w, a), k,
            candidate('payment_type', w, a), k)
    return result


def probability(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid conditional probability')
    return values


def blend(global_data, expert_data, alpha):
    if isinstance(alpha, bool) or alpha not in [0, .25, .5, .75, 1]:
        raise ValueError('Undeclared blend weight')
    aligned(global_data, expert_data)
    g, e = probability(global_data['scores']), probability(expert_data['scores'])
    return dict(row_ids=global_data['row_ids'], labels=global_data['labels'],
                scores=(1 - alpha) * g + alpha * e)


def eligibility(frame, month, partition):
    if partition not in PARTITIONS:
        raise ValueError('Undeclared partition')
    train = frame.loc[frame.month < month]
    result = []
    for level in PARTITIONS[partition]:
        y = train.loc[train[partition] == level, 'fraud_bool']
        n, fraud = len(y), int(y.sum())
        eligible = n >= MINIMUM['rows'] and fraud >= MINIMUM['fraud'] and n - fraud >= MINIMUM['legitimate']
        result.append(dict(eval_month=month, partition=partition, level=level, train_rows=n,
                           train_fraud=fraud, train_legitimate=n - fraud, eligible=eligible))
    return result


def model_specs(frame, month, pure_methods=None, capacities=None):
    pure_methods = list(product(PARTITIONS, WEIGHTINGS)) if pure_methods is None else pure_methods
    capacities = WEIGHTINGS if capacities is None else capacities
    rows = [dict(id=capacity(w), kind='capacity', weighting=w, rounds=400) for w in capacities]
    for p, w in pure_methods:
        for g in eligibility(frame, month, p):
            if g['eligible']:
                rows.append(dict(id=f'expert_model_{p}_{w}_{g["level"]}', kind='local',
                                 partition=p, level=g['level'], weighting=w, rounds=200))
    return rows


def dependencies(names, frame, month=7):
    old = old_scores.dependencies([n for n, k in REFERENCE_ROLES.values()])
    full, pure, caps = set(old['scores']), set(), set()
    for name in names:
        s = lookup(name)
        if s['kind'] == 'capacity':
            caps.add(s['weighting']); full.add(name)
        elif s['kind'] in ['expert', 'blend']:
            pure.add((s['partition'], s['weighting']))
            full.update([expert(s['partition'], s['weighting']), name])
    return dict(old_bases=old['bases'], old_scores=old['scores'],
                new_models=model_specs(frame, month, sorted(pure), sorted(caps)),
                pure_methods=[list(x) for x in sorted(pure)], capacity_weightings=sorted(caps),
                full_scores=sorted(full))


def class_costs(labels, weighting):
    global200(weighting)
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid earlier training labels')
    n, positive = len(y), int(y.sum())
    if not 0 < positive < n:
        raise ValueError('Both training classes required')
    return {'0': n / (2 * (n - positive)), '1': n / (2 * positive)} if weighting == 'balanced' else {'0': 1., '1': 1.}


def weights_for(labels, costs):
    return np.where(np.asarray(labels) == 1, costs['1'], costs['0']).astype(np.float64)


def weights_digest(weights):
    return hashlib.sha256(np.asarray(weights, dtype='<f8').tobytes()).hexdigest()


def estimator(c, rounds):
    if rounds not in [200, 400]:
        raise ValueError('Undeclared round count')
    original = read_config(c.get('original_base_config', 'configs/study.json'))
    original = dict(original, threads=c['threads'])
    model = make_model('lightgbm', False, original, []).named_steps['model']
    model.set_params(n_estimators=rounds)
    return model


def signature(model):
    return json.loads(json.dumps(model.get_params(), sort_keys=True, default=str))


def checked_booster(c, model, rounds):
    if signature(model) != signature(estimator(c, rounds)) or list(model.classes_) != [0, 1]:
        raise ValueError('Conditional classifier recipe changed')
    actual = model.booster_.params
    required = dict(num_iterations=rounds, num_threads=c['threads'], num_leaves=15,
                    learning_rate=.05, min_child_samples=100, random_state=2026,
                    deterministic=True, force_col_wise=True, objective='binary',
                    subsample=1., subsample_freq=0, colsample_bytree=1.)
    if any(actual.get(k) != v for k, v in required.items()):
        raise ValueError('Effective conditional recipe changed')
    for primary, alias in [('bagging_fraction', 'subsample'), ('bagging_freq', 'subsample_freq'),
                           ('feature_fraction', 'colsample_bytree')]:
        if primary in actual and actual[primary] != required[alias]:
            raise ValueError('Conflicting effective alias')
    if actual.get('is_unbalance', False) or actual.get('scale_pos_weight', 1.) != 1.:
        raise ValueError('Duplicate class costs')
    if not 1 <= model.booster_.current_iteration() <= rounds:
        raise ValueError('Invalid fitted round count')
    return actual


def model_paths(c, phase, month, name):
    return [*paths(c, phase, month, name),
            Path(c['output_dir']) / 'models' / f'{phase}-m{month}-{name}.joblib']


def membership(frame, spec, month):
    train = frame.index[frame.month < month]
    future = frame.index[frame.month == month]
    if spec['kind'] == 'local':
        partition, level = spec['partition'], spec['level']
        groups = eligibility(frame, month, partition)
        if not any(g['level'] == level and g['eligible'] for g in groups):
            raise ValueError('Local fit is not eligible')
        train = train[frame.loc[train, partition].eq(level)]
        future = future[frame.loc[future, partition].eq(level)]
    elif spec['kind'] != 'capacity':
        raise ValueError('Undeclared model kind')
    if spec not in model_specs(frame, month):
        raise ValueError('Undeclared local/capacity recipe')
    return train, future


def expected_model(c, fp, frame, base, spec, month, phase, preprocessor):
    train, ev = membership(frame, spec, month)
    whole = frame.index[frame.month < month]
    labels = frame.loc[whole, 'fraud_bool'].to_numpy()
    costs = class_costs(labels, spec['weighting'])
    global_weights = weights_for(labels, costs)
    local_weights = weights_for(frame.loc[train, 'fraud_bool'], costs)
    pipeline = Pipeline([('preprocess', preprocessor), ('model', estimator(c, spec['rounds']))])
    meta = bagged.audit_preprocessing(pipeline, base.loc[whole])
    return dict(candidate=spec['id'], spec=spec, phase=phase, eval_month=month, fingerprint=fp,
                global_train_rows=len(whole), global_train_fraud=int(labels.sum()),
                global_train_ids_sha256=ids_digest(whole.to_numpy()),
                global_weight_sha256=weights_digest(global_weights), class_costs=costs,
                local_train_rows=len(train), local_train_fraud=int(frame.loc[train, 'fraud_bool'].sum()),
                local_train_ids_sha256=ids_digest(train.to_numpy()),
                local_weight_sha256=weights_digest(local_weights),
                evaluation_ids_sha256=ids_digest(ev.to_numpy()), subset=spec['kind'] == 'local',
                preprocessing=meta, requested_model_params=signature(pipeline.named_steps['model']))


def audit_model(c, fp, frame, base, spec, month, preprocessor, phase='validation'):
    ps = model_paths(c, phase, month, spec['id'])
    if not bagged.cache_state(ps):
        raise ValueError('Required conditional cache absent')
    record = read_config(ps[0])
    expected = expected_model(c, fp, frame, base, spec, month, phase, preprocessor)
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Conditional population/weights/representation identity changed')
    if digest(ps[1]) != record['prediction_sha256'] or digest(ps[2]) != record['model_sha256'] or record['warnings']:
        raise ValueError('Conditional artifact or warnings changed')
    artifact = joblib.load(ps[2]); pipeline = artifact['model']
    if artifact['spec'] != spec or record['preprocessing'] != bagged.audit_preprocessing(
            pipeline, base.loc[frame.month < month]):
        raise ValueError('Saved shared preprocessing changed')
    model = pipeline.named_steps['model']
    if checked_booster(c, model, spec['rounds']) != record['booster_params']:
        raise ValueError('Saved effective parameters changed')
    if record['fitted_tree_count'] != model.booster_.current_iteration():
        raise ValueError('Saved fitted rounds changed')
    _, ev = membership(frame, spec, month)
    d = prediction(c, phase, month, spec['id'])
    if not np.array_equal(d['row_ids'], ev.to_numpy()) or not np.array_equal(d['labels'], frame.loc[ev, 'fraud_bool'].to_numpy()):
        raise ValueError('Conditional member subset changed')
    probability(d['scores'])
    with threadpool_limits(limits=c['threads']):
        actual = pipeline.predict_proba(base.loc[ev])[:, 1] if len(ev) else np.empty(0)
    if not np.allclose(actual, d['scores'], atol=1e-12, rtol=0):
        raise ValueError('Saved conditional inference changed')
    return record


def fit_model(c, fp, frame, base, spec, month, preprocessor, phase='validation', allow_missing=True,
              training_matrix=None):
    ps = model_paths(c, phase, month, spec['id'])
    if bagged.cache_state(ps):
        return audit_model(c, fp, frame, base, spec, month, preprocessor, phase)
    if not allow_missing:
        raise ValueError('Required conditional cache absent')
    failure = Path(c['output_dir']) / 'failures' / (ps[0].stem + '.json')
    if failure.exists():
        raise ValueError('Prior conditional failure retained; no automatic retry')
    expected = expected_model(c, fp, frame, base, spec, month, phase, preprocessor)
    train, ev = membership(frame, spec, month); start = time.perf_counter()
    try:
        pipeline = Pipeline([('preprocess', preprocessor), ('model', estimator(c, spec['rounds']))])
        model = pipeline.named_steps['model']
        whole = frame.index[frame.month < month]
        matrix = preprocessor.transform(base.loc[whole]) if training_matrix is None else training_matrix
        if matrix.shape != (len(whole), expected['preprocessing']['transformed_dimension']):
            raise ValueError('Global training matrix schema differs')
        matrix = matrix[whole.get_indexer(train)]
        y = frame.loc[train, 'fraud_bool'].to_numpy()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=c['threads']):
            warnings.simplefilter('always')
            model.fit(matrix, y, sample_weight=weights_for(y, expected['class_costs']))
            values = pipeline.predict_proba(base.loc[ev])[:, 1] if len(ev) else np.empty(0)
        messages = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if messages:
            raise ValueError('Conditional warning: ' + repr(messages))
        probability(values); effective = checked_booster(c, model, spec['rounds'])
        for path in ps:
            path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(spec=spec, model=pipeline), ps[2])
        np.savez_compressed(ps[1], row_ids=ev.to_numpy(), labels=frame.loc[ev, 'fraud_bool'].to_numpy(), scores=values)
        write_json(ps[0], dict(**expected, model_sha256=digest(ps[2]), prediction_sha256=digest(ps[1]),
            warnings=messages, booster_params=effective, fitted_tree_count=model.booster_.current_iteration(),
            train_row_rounds=len(train) * model.booster_.current_iteration(), reused=False,
            fit_seconds=time.perf_counter() - start,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, created_at_utc=now()))
        return audit_model(c, fp, frame, base, spec, month, preprocessor, phase)
    except Exception as exc:
        if not failure.exists():
            write_json(failure, dict(**expected, error=repr(exc), elapsed_seconds=time.perf_counter() - start,
                                     created_at_utc=now()))
        raise


def stitch(frame, month, partition, global_data, subsets, eligible_levels):
    if partition not in PARTITIONS or not set(eligible_levels) <= set(PARTITIONS[partition]):
        raise ValueError('Invalid routing declaration')
    expected_levels = {g['level'] for g in eligibility(frame, month, partition) if g['eligible']}
    if set(eligible_levels) != expected_levels:
        raise ValueError('Routing must retain exact earlier training eligibility')
    check_data(global_data)
    ev = frame.index[frame.month == month].to_numpy()
    if not np.array_equal(ev, global_data['row_ids']):
        raise ValueError('Global routing population differs')
    if not np.array_equal(frame.loc[ev, 'fraud_bool'].to_numpy(), global_data['labels']):
        raise ValueError('Global routing labels differ')
    if set(subsets) != set(eligible_levels):
        raise ValueError('Exactly the eligible member subsets required')
    scores = probability(global_data['scores']).copy()
    values = frame.loc[ev, partition]
    for level in eligible_levels:
        take = values.eq(level).fillna(False).to_numpy(dtype=bool)
        item = subsets[level]
        if not np.array_equal(item['row_ids'], ev[take]) or not np.array_equal(item['labels'], global_data['labels'][take]):
            raise ValueError('Conditional subset overlap, omission or membership mismatch')
        local = probability(item['scores'])
        if len(local) != int(take.sum()):
            raise ValueError('Conditional subset size differs')
        scores[take] = local
    buckets = values.astype(object).copy()
    buckets.loc[values.isna()] = '__missing__'
    buckets.loc[~values.isna() & ~values.isin(PARTITIONS[partition])] = '__unseen__'
    routing = []
    for level in PARTITIONS[partition] + ['__missing__', '__unseen__']:
        count = int(buckets.eq(level).sum())
        route = 'expert' if level in eligible_levels else 'missing' if level == '__missing__' else 'unseen' if level == '__unseen__' else 'ineligible'
        routing.append(dict(eval_month=month, partition=partition, level=level, rows=count, route=route))
    assert sum(r['rows'] for r in routing) == len(ev)
    return dict(row_ids=ev, labels=global_data['labels'], scores=scores), routing
