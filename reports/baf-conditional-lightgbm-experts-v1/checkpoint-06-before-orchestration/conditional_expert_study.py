"""Isolated contract and immutable source adapters for conditional fraud models."""
from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from . import bagged_anchor_study as parent
from . import conditional_experts as scores
from .common import read_config, write_json, digest, load_data, features
from .focal_study import now
from .improvements import paths, prediction
from .recall_weighted_metrics import aligned, POLICIES, learn, measure, decisions
from .recall_extension import order_summary, expected_population
from .improvement_metrics import confusion_metrics

CONFIG = 'configs/conditional-lightgbm-experts.json'
PLANNING = 'openspec/changes/test-conditional-lightgbm-experts-f2/planning-evidence.json'
SOURCES = parent.SOURCES + ['baf/conditional_experts.py', 'baf/conditional_expert_study.py',
                           'baf/conditional_expert_report.py']
verify_hashes = parent.verify_hashes
cache_state = parent.cache_state


def planning_evidence_path(root=Path('openspec/changes')):
    name = 'test-conditional-lightgbm-experts-f2'
    active = Path(root) / name / 'planning-evidence.json'
    if active.is_file():
        return active
    archived = list((Path(root) / 'archive').glob('*-' + name + '/planning-evidence.json'))
    if len(archived) != 1:
        raise ValueError('Exactly one original planning-evidence file required')
    return archived[0]


def fingerprint(path=CONFIG):
    result = parent.fingerprint(path)
    result['sources'] = {p: digest(p) for p in SOURCES}
    return result


def mandatory_preservation(source):
    result = dict(source['preserve_sha256'])
    for p in parent.SOURCES[-3:] + [parent.CONFIG, source['protocol_path'],
            str(Path(source['output_dir']) / 'selection.json'),
            str(Path(source['report_dir']) / 'report_hashes.json')]:
        result[p] = digest(p)
    result.update(read_config(Path(source['report_dir']) / 'report_hashes.json'))
    return result


def declaration(source):
    planning_path = planning_evidence_path()
    planning = read_config(planning_path)
    return dict(parent_config=parent.CONFIG, parent_fingerprint=parent.fingerprint(),
        preserve_sha256=mandatory_preservation(source), source_paths=SOURCES,
        strategies=scores.strategies(), reference_roles=scores.REFERENCE_ROLES,
        partitions=scores.PARTITIONS, conditional_alphas=scores.ALPHAS,
        conditional_minimum=scores.MINIMUM, conditional_rounds=200, capacity_rounds=400,
        expected_counts=planning['expected_counts'],
        planning_evidence_path=PLANNING, planning_evidence_sha256=digest(planning_path),
        output_dir='outputs/baf-conditional-lightgbm-experts-v1',
        report_dir='reports/baf-conditional-lightgbm-experts-v1',
        protocol_path='docs/conditional-lightgbm-experts-protocol.md',
        historical_output_roots=sorted(set(source['historical_output_roots'] + [source['output_dir']])))


def guard(c):
    source = read_config(parent.CONFIG); parent.guard(source)
    expected = declaration(source)
    if set(c) != set(source) | set(expected) or any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('Prospective conditional declaration changed')
    if any(c.get(k) != v for k, v in source.items() if k not in expected):
        raise ValueError('Frozen reference recipe changed')
    if len(set(SOURCES)) != 58 or len(c['preserve_sha256']) != 1053:
        raise ValueError('Source or preservation closure changed')
    verify_hashes(c['preserve_sha256'])
    selections = [p for p in c['preserve_sha256'] if p.startswith('outputs/') and p.endswith('/selection.json')]
    if len(selections) != 17 or any(c['preserve_sha256'].get(p.replace('outputs/', 'reports/', 1)) != c['preserve_sha256'][p] for p in selections):
        raise ValueError('Seventeen original/exported selection pairs required')
    dest = [Path(c[k]).resolve() for k in ['output_dir', 'report_dir']]
    for a, b in combinations(dest + [Path(p).resolve() for p in c['historical_output_roots']], 2):
        if (a in dest or b in dest) and (a == b or a in b.parents or b in a.parents):
            raise ValueError('Destination overlap')
    initial = Path(c['output_dir']) / 'preimplementation_preservation.json'
    if initial.exists():
        verify_hashes(read_config(initial))
    for key in ['primary_fraction', 'review_quota', 'precision_floor']:
        if key in c:
            raise ValueError('Undeclared operating constraint')


def verify_declaration(config_path=CONFIG):
    c = read_config(config_path); guard(c); p = read_config(planning_evidence_path())
    if scores.strategies() != p['strategies'] or scores.contrast_factors() != p['contrasts']:
        raise ValueError('Independent prospective identities differ')
    if c['preserve_sha256'] != p['mandatory_preservation_sha256']:
        raise ValueError('Planning preservation differs')
    verify_hashes(p['historical_ignored_file_sha256'])
    actual = {str(f) for root in c['historical_output_roots'] for f in Path(root).rglob('*') if f.is_file()}
    if actual != set(p['historical_ignored_file_sha256']):
        raise ValueError('Historical file set changed')
    frame = load_data(c); frame = frame[frame.month < 7]
    eligibility = [g for m in [4, 5, 6, 7] for part in scores.PARTITIONS for g in scores.eligibility(frame, m, part)]
    if eligibility != p['eligibility']:
        raise ValueError('Training-only eligibility differs')
    for case in p['winner_policy_cases']:
        choices = scores.choices_for(case['winner'], case['policy'])
        dag = scores.dependencies([n for n, k in choices.values()], frame)
        old = case['dependencies']
        if choices != case['roles'] or dag['full_scores'] != old['full_scores'] or dag['old_bases'] != old['old_bases']:
            raise ValueError('Winner-dependent roles or source DAG differs')
        capacity = [s['id'] for s in dag['new_models'] if s['kind'] == 'capacity']
        local = [s['id'] for s in dag['new_models'] if s['kind'] == 'local']
        if capacity != old['capacity_bases'] or sorted(local) != old['expert_models']:
            raise ValueError('Winner-dependent exact fit bounds differ')
    out = Path(c['output_dir']); initial = out / 'preimplementation_preservation.json'
    if not initial.exists():
        write_json(initial, p['historical_ignored_file_sha256'])
    receipt = dict(config_sha256=digest(config_path), planned_source_paths=SOURCES,
        mandatory_paths=1053, historical_files=4369, selection_pairs=17, strategies=24,
        winner_policy_cases=72, contrast_identities=738, data_sha256=c['data_sha256'],
        new_model_fits=0, new_score_evaluations=0, created_at_utc=now())
    write_json(out / 'contract_verification.json', receipt)
    return receipt


def required_sources(c):
    names = parent.scores.dependencies([n for n, k in scores.REFERENCE_ROLES.values()])['bases']
    available = {s['id']: s for s in parent.reference_specs(c) + parent.scores.recipes()}
    return [available[n] for n in names]


def prepare_source(c, fp, frame, base, spec, month, phase='validation', allow_missing=True):
    if spec not in required_sources(c):
        raise ValueError('Unrelated source recipe is not required')
    # Parent adapters receive only an individual recipe and never invoke their grids.
    return parent.prepare_source(c, fp, frame, base, spec, month, phase, allow_missing)


def derived_reference(c, fp, name, month, phase='validation', allow_missing=True):
    dag = parent.scores.dependencies([n for n, k in scores.REFERENCE_ROLES.values()])
    if name not in dag['derived']:
        raise ValueError('Unrelated derived score')
    spec = parent.derived_spec(name)
    items = {n: prediction(c, phase, month, n) for n in spec['members']}
    actual = (parent.parent.old_compose(spec, items) if spec['kind'] == 'anchor'
              else parent.scores.compose(spec, items))
    deps = {n: {str(p): digest(p) for p in paths(c, phase, month, n)} for n in spec['members']}
    expected = dict(candidate=name, phase=phase, eval_month=month, fingerprint=fp,
                    spec=spec, dependencies=deps, arithmetic=parent.arithmetic(spec), epsilon=parent.scores.EPSILON)
    rp, pp = paths(c, phase, month, name)
    if cache_state([rp, pp]):
        record = read_config(rp); saved = prediction(c, phase, month, name); aligned(actual, saved)
        if any(record.get(k) != v for k, v in expected.items()) or digest(pp) != record['prediction_sha256'] or not np.array_equal(actual['scores'], saved['scores']):
            raise ValueError('Derived reference identity or arithmetic changed')
        verify_hashes(record['origin_artifacts'])
        return record
    if not allow_missing:
        raise ValueError('Required derived reference absent')
    source = read_config(c['parent_config']); sr, sp = paths(source, phase, month, name)
    origin = {}
    if cache_state([sr, sp]):
        parent.derived(source, c['parent_fingerprint'], name, month, phase, allow_missing=False)
        saved = prediction(source, phase, month, name); aligned(actual, saved)
        if not np.array_equal(actual['scores'], saved['scores']):
            raise ValueError('Original reference scores differ')
        origin = {str(p): digest(p) for p in [sr, sp]}
        for member in spec['members']:
            origin.update({str(p): digest(p) for p in paths(source, phase, month, member)})
    pp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pp, **actual)
    record = dict(**expected, prediction_sha256=digest(pp), origin_artifacts=origin, reused=bool(origin))
    write_json(rp, record)
    return record


def global_preprocessor(c, fp, frame, base, month, phase='validation'):
    required = {s['id']: s for s in required_sources(c)}
    pipes = []
    for name in ['baseline_lgbm', 'baseline_balanced']:
        parent.audit_source(c, fp, frame, base, required[name], month, phase)
        ps = parent.source_paths(c, required[name], month, phase)
        if not cache_state(ps):
            raise ValueError('Audited global source required before conditional fitting')
        artifact = joblib.load(ps[2])
        pipe = artifact['model']
        pipes.append(pipe)
    whole = frame.index[frame.month < month]
    metadata = [scores.bagged.audit_preprocessing(pipe, base.loc[whole]) for pipe in pipes]
    if metadata[0] != metadata[1]:
        raise ValueError('Global weighting changed preprocessing')
    return pipes[0].named_steps['preprocess']


def physical_names():
    old = parent.scores.dependencies([n for n, k in scores.REFERENCE_ROLES.values()])['scores']
    return sorted(set(old) | {s['id'] for s in scores.strategies()})


def history(c, *, physical=False):
    names = physical_names() if physical else [s['id'] for s in scores.strategies()]
    return {n: {m: prediction(c, 'validation', m, n) for m in c['validation_months']} for n in names}


def new_derived(c, fp, frame, name, month, phase='validation', allow_missing=True):
    spec = scores.lookup(name)
    if spec['kind'] not in ['expert', 'blend']:
        raise ValueError('Not a new derived conditional score')
    p, w = spec['partition'], spec['weighting']; member = scores.global200(w)
    routing = []
    if spec['kind'] == 'expert':
        eligible = [g['level'] for g in scores.eligibility(frame, month, p) if g['eligible']]
        subsets = {v: prediction(c, phase, month, f'expert_model_{p}_{w}_{v}') for v in eligible}
        d, routing = scores.stitch(frame, month, p, prediction(c, phase, month, member), subsets, eligible)
        names = [member] + [f'expert_model_{p}_{w}_{v}' for v in eligible]
        arithmetic = 'one eligible member per original partition value, else global200'
    else:
        names = [member, scores.expert(p, w)]
        d = scores.blend(prediction(c, phase, month, names[0]), prediction(c, phase, month, names[1]),
                         spec['alpha_numerator'] / spec['alpha_denominator'])
        arithmetic = '(1-alpha)*global200+alpha*stitched_expert in float64'
    deps = {n: {str(v): digest(v) for v in paths(c, phase, month, n)} for n in names}
    expected = dict(candidate=name, phase=phase, eval_month=month, spec=spec, fingerprint=fp,
                    dependencies=deps, arithmetic=arithmetic, routing=routing)
    rp, pp = paths(c, phase, month, name)
    if cache_state([rp, pp]):
        r = read_config(rp); saved = prediction(c, phase, month, name); aligned(d, saved)
        if any(r.get(k) != v for k, v in expected.items()) or digest(pp) != r['prediction_sha256'] or not np.array_equal(d['scores'], saved['scores']):
            raise ValueError('Conditional derived identity or arithmetic changed')
        return r
    if not allow_missing:
        raise ValueError('Required conditional derived score absent')
    pp.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(pp, **d)
    r = dict(**expected, prediction_sha256=digest(pp), reused=False)
    write_json(rp, r)
    return r


def prepare(c, fp, frame, base, phase='validation', names=None, allow_missing=True):
    names = [s['id'] for s in scores.strategies()] if names is None else sorted(set(names))
    old_names = [n for n, k in scores.REFERENCE_ROLES.values()]
    old_dag = parent.scores.dependencies(old_names)
    old_specs = {s['id']: s for s in required_sources(c)}
    base_records, derived_records, work = [], [], []
    for m in c['validation_months'] if phase == 'validation' else [7]:
        for n in old_dag['bases']:
            print('Preparing source', phase, m, n, flush=True)
            existed = cache_state(parent.source_paths(c, old_specs[n], m, phase))
            r = prepare_source(c, fp, frame, base, old_specs[n], m, phase, allow_missing)
            base_records.append(r)
            work.append(dict(candidate=n, eval_month=m, kind='reference',
                             work='audit existing' if existed else 'reuse origin' if r.get('reused') else 'fit exact recipe'))
        for n in parent.derived_order(old_names):
            existed = cache_state(paths(c, phase, m, n))
            r = derived_reference(c, fp, n, m, phase, allow_missing); derived_records.append(r)
            work.append(dict(candidate=n, eval_month=m, kind='old_derived',
                             work='audit existing' if existed else 'reuse origin' if r['reused'] else 'calculate'))
        dag = scores.dependencies(names, frame, m)
        new_specs = dag['new_models']
        if new_specs:
            pre = global_preprocessor(c, fp, frame, base, m, phase)
            matrix = pre.transform(base.loc[frame.month < m]) if allow_missing else None
            for spec in new_specs:
                n = spec['id']; print('Preparing conditional', phase, m, n, flush=True)
                existed = cache_state(scores.model_paths(c, phase, m, n))
                r = scores.fit_model(c, fp, frame, base, spec, m, pre, phase, allow_missing, training_matrix=matrix)
                base_records.append(r)
                work.append(dict(candidate=n, eval_month=m, kind=spec['kind'],
                                 work='audit existing' if existed else 'fit conditional recipe'))
            del matrix
        pure = [scores.expert(p, w) for p, w in dag['pure_methods']]
        additions = [n for n in dag['full_scores'] if n not in old_dag['scores'] and scores.lookup(n)['kind'] == 'blend']
        for n in pure + additions:
            existed = cache_state(paths(c, phase, m, n))
            r = new_derived(c, fp, frame, n, m, phase, allow_missing); derived_records.append(r)
            work.append(dict(candidate=n, eval_month=m, kind='new_derived', work='audit existing' if existed else 'calculate'))
        if allow_missing:
            write_json(Path(c['output_dir']) / (phase + '_progress.json'), dict(fingerprint=fp, work=work))
    return base_records, derived_records, work


def crossfit(c, h=None):
    h = history(c) if h is None else h
    rows = []
    reference = scores.REFERENCE_ROLES['balanced'][0]
    for m, name, policy in product([5, 6], [s['id'] for s in scores.strategies()], POLICIES):
        aligned(h[name][m], h[reference][m])
        rule = learn({t: d for t, d in h[name].items() if t < m}, policy, m)
        rows.append(dict(candidate=name, policy=policy, eval_month=m,
                         source_months=','.join(map(str, rule['source_months'])),
                         parameters=json.dumps(rule, sort_keys=True), **measure(h[name][m], rule)))
    return pd.DataFrame(rows)


def group_buckets(frame, ids, partition):
    v = frame.loc[ids, partition]; result = v.astype(object).copy()
    result.loc[v.isna()] = '__missing__'
    result.loc[~v.isna() & ~v.isin(scores.PARTITIONS[partition])] = '__unseen__'
    return result.to_numpy()


def group_metrics(tn, fp, fn, tp):
    if tn + fp + fn + tp:
        return confusion_metrics(tn, fp, fn, tp)
    return {k: 0 for k in ['precision','recall','f1','f2','mcc','balanced_accuracy','fpr',
                           'accuracy','flagged','tp','fp','fn','tn']}


def tables(c, frame, h=None):
    h = history(c, physical=True) if h is None else h
    if set(h) != set(physical_names()) or any(set(v) != {4, 5, 6} for v in h.values()):
        raise ValueError('Exact full physical score grid required')
    for n, m in product(h, [4, 5, 6]):
        aligned(h[n][m], expected_population(frame, m))
    cf = crossfit(c, h); summary = order_summary(cf).reset_index(drop=True)
    index = summary.set_index(['candidate', 'policy']); contrasts = []; pooled = []
    for kind, n, k, b, bk, extra in scores.contrast_factors():
        a, ref = index.loc[(n, k)], index.loc[(b, bk)]
        contrasts.append(dict(contrast=kind, candidate=n, policy=k, reference=b, reference_policy=bk,
            factor_details=json.dumps(extra, separators=(',', ':')),
            **{v + '_difference': float(a[v] - ref[v]) for v in ['f2','recall','precision','flagged_fraction','average_precision']}))
    for (n, k), group in cf.groupby(['candidate', 'policy']):
        t = group[['tn','fp','fn','tp']].sum()
        pooled.append(dict(candidate=n, policy=k, **confusion_metrics(*[int(t[v]) for v in ['tn','fp','fn','tp']]),
                           flagged_fraction=float((t.tp + t.fp) / group.n.sum())))
    eligibility = []; routing = []; errors = []; monthly = []
    for m, p, w in product([4, 5, 6], scores.PARTITIONS, scores.WEIGHTINGS):
        gs = scores.eligibility(frame, m, p)
        eligibility.extend([dict(g, weighting=w, fallback=not g['eligible']) for g in gs])
        ids = h[scores.global200(w)][m]['row_ids']; buckets = group_buckets(frame, ids, p)
        eligible = {g['level'] for g in gs if g['eligible']}
        for level in scores.PARTITIONS[p] + ['__missing__','__unseen__']:
            route = 'expert' if level in eligible else 'missing' if level == '__missing__' else 'unseen' if level == '__unseen__' else 'ineligible'
            routing.append(dict(eval_month=m, partition=p, weighting=w, level=level,
                                rows=int(np.sum(buckets == level)), route=route))
    for n, items in h.items():
        for m, d in items.items():
            v = d['scores']
            monthly.append(dict(candidate=n, eval_month=m, n=len(v), score_min=float(v.min()),
                score_max=float(v.max()), score_mean=float(v.mean()), unique_scores=len(np.unique(v)),
                tied_fraction=1-len(np.unique(v))/len(v)))
    for r in cf.to_dict('records'):
        n, k, m = r['candidate'], r['policy'], r['eval_month']; d = h[n][m]
        pred = decisions(d, json.loads(r['parameters'])); y = d['labels'].astype(bool)
        for p in scores.PARTITIONS:
            buckets = group_buckets(frame, d['row_ids'], p)
            for level in scores.PARTITIONS[p] + ['__missing__','__unseen__']:
                take = buckets == level
                tn, fp, fn, tp = [int(np.sum(take & mask)) for mask in [~y & ~pred, ~y & pred, y & ~pred, y & pred]]
                errors.append(dict(candidate=n, policy=k, eval_month=m, partition=p, level=level,
                    n=int(take.sum()), fraud=int(np.sum(take & y)), **group_metrics(tn, fp, fn, tp),
                    flagged_fraction=(tp + fp) / int(take.sum()) if take.any() else 0.))
    monthly = pd.DataFrame(monthly)
    return dict(crossfit=cf, summary=summary, pooled=pd.DataFrame(pooled), contrasts=pd.DataFrame(contrasts),
        eligibility=pd.DataFrame(eligibility), routing=pd.DataFrame(routing), stratum_errors=pd.DataFrame(errors),
        physical_score_diagnostics=monthly,
        score_diagnostics=monthly[monthly.candidate.isin([s['id'] for s in scores.strategies()])].reset_index(drop=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify-declaration'])
    parser.add_argument('--config', default=CONFIG)
    args = parser.parse_args()
    if args.action == 'verify-declaration':
        print(verify_declaration(args.config))


if __name__ == '__main__':
    main()
