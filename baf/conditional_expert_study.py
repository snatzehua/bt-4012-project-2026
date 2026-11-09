"""Isolated contract and immutable source adapters for conditional fraud models."""
from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import shutil
import copy
import os
import resource
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from . import bagged_anchor_study as parent
from . import conditional_experts as scores
from .common import read_config, write_json, digest, load_data, features, make_model, CATEGORICAL
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
TABLES = [n + '.csv' for n in ['crossfit', 'summary', 'pooled', 'contrasts', 'eligibility',
    'routing', 'stratum_errors', 'physical_score_diagnostics', 'score_diagnostics']]
WORK_TABLES = ['conditional_model_work.csv','conditional_inference_work.csv']
TEST_SOURCE = 'tests/test_conditional_experts.py'


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


def present_history(c,original):
    actual={str(f) for root in c['historical_output_roots'] for f in Path(root).rglob('*') if f.is_file()}
    if not actual<=set(original):raise ValueError('Undeclared historical files appeared')
    present={name:original[name] for name in sorted(actual)}
    verify_hashes(present)
    return present


def verify_declaration(config_path=CONFIG):
    c = read_config(config_path); guard(c); p = read_config(planning_evidence_path())
    if scores.strategies() != p['strategies'] or scores.contrast_factors() != p['contrasts']:
        raise ValueError('Independent prospective identities differ')
    if c['preserve_sha256'] != p['mandatory_preservation_sha256']:
        raise ValueError('Planning preservation differs')
    original = p['historical_ignored_file_sha256']
    present=present_history(c,original)
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
        write_json(initial,present)
    receipt = dict(config_sha256=digest(config_path), planned_source_paths=SOURCES,
        mandatory_paths=1053, historical_files=4369, selection_pairs=17, strategies=24,
        winner_policy_cases=72, contrast_identities=738, data_sha256=c['data_sha256'],
        new_model_fits=0, new_score_evaluations=0, created_at_utc=now())
    path=out/'contract_verification.json'
    if path.exists():
        previous=read_config(path)
        if any(previous.get(k)!=v for k,v in receipt.items() if k!='created_at_utc'):
            raise ValueError('Original contract verification changed')
        return previous
    write_json(path, receipt);return receipt


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


def initialize(config_path=CONFIG):
    """Freeze the execution identity; never silently update it after code edits."""
    c = read_config(config_path); guard(c); fp = fingerprint(config_path)
    out = Path(c['output_dir']); preservation = out / 'preservation.json'
    expected = dict(mandatory=c['preserve_sha256'],
        optional_present=read_config(out / 'preimplementation_preservation.json'),
        roots=c['historical_output_roots'])
    verify_hashes(expected['optional_present'])
    if preservation.exists():
        if read_config(preservation) != expected:
            raise ValueError('Frozen preservation declaration changed')
    else:
        write_json(preservation, expected)
    record = dict(fingerprint=fp, config_path=str(config_path),
        protocol_sha256=digest(c['protocol_path']), preservation_sha256=digest(preservation),
        strategies=c['strategies'], expected_counts=c['expected_counts'], month7_previously_consumed=True)
    path = out / 'declaration.json'
    if path.exists():
        if any(read_config(path).get(k) != v for k, v in record.items()):
            raise ValueError('Frozen execution declaration changed')
    else:
        write_json(path, dict(**record, created_at_utc=now()))
    return c, fp


def check_receipt(c, fp):
    out = Path(c['output_dir']); dest = Path(c['report_dir'])
    r = read_config(out / 'checks.json')
    if r['fingerprint'] != fp or r['test_source_sha256'] != digest(TEST_SOURCE):
        raise ValueError('Current complete methodological checks required')
    if r['junit_sha256'] != digest(dest / 'tests.xml') or r['actual_fit_ledger_sha256'] != digest(dest / 'test_model_fits.json'):
        raise ValueError('Execution test evidence changed')
    verify_hashes(read_config(dest / 'test_model_fits.json')['execution_source_sha256'])
    return r


def record_checks(config_path=CONFIG):
    """Attest exact full-suite identities and observed fits before freezing code."""
    c = read_config(config_path); guard(c); out = Path(c['output_dir']); dest = Path(c['report_dir'])
    if (out / 'selection.json').exists():
        raise ValueError('Frozen check receipt is immutable')
    xml = dest / 'tests.xml'; root = ET.parse(xml).getroot(); cases = list(root.iter('testcase'))
    if not cases or any(list(root.iter(n)) for n in ['failure', 'error', 'skipped']):
        raise ValueError('Complete passing methodological suite required')
    own = [t.get('name', '') for t in cases if t.get('classname', '').endswith('test_conditional_experts')]
    expected_formats = {'bagged_balanced', 'bagged_unweighted', 'spline_balanced', 'spline_unweighted', 'xgboost', 'lightgbm'}
    formats = [n for n in own if n.startswith('test_six_reference_formats_through_new_destination[')]
    if len(formats) != 6 or {n.split('[')[-1].removesuffix(']') for n in formats} != expected_formats:
        raise ValueError('All six genuine reference-format fits required')
    if sum(n.startswith('test_four_new_formats_fit_audit_reuse_and_reject_corruption[') for n in own) != 4:
        raise ValueError('All four genuine conditional-format fits required')
    if sum(n.startswith('test_all_winner_policy_cases_preserve_roles_and_bounds[') for n in own) != 72:
        raise ValueError('All 72 winner/policy cases required')
    actual = [(t.get('classname', ''), t.get('name', '')) for t in cases]
    for path in sorted(Path('tests').glob('test_*.py')):
        prefix = 'tests.' + path.stem
        if not any(cls == prefix or cls.startswith(prefix + '.') for cls, name in actual):
            raise ValueError('Every methodological module required')
    lp = dest / 'test_model_fits.json'; ledger = read_config(lp)
    if ledger['exit_code'] != 0 or ledger['junit_sha256'] != digest(xml) or ledger['deselected_nodes']:
        raise ValueError('Complete actual-fit ledger required')
    expected = []
    for node in ledger['collected_nodes']:
        parts = node.split('::')
        expected.append(('.'.join([parts[0].removesuffix('.py').replace('/', '.'), *parts[1:-1]]), parts[-1]))
    if len(actual) != len(set(actual)) or len(expected) != len(set(expected)) or set(actual) != set(expected):
        raise ValueError('Exact collected/executed identities differ')
    sources=set(SOURCES+[CONFIG,c['protocol_path'],'scripts/check_conditional_experts.py']+[
        str(p) for p in sorted(Path('tests').glob('test_*.py'))])
    if set(ledger['execution_source_sha256'])!=sources:
        raise ValueError('Complete execution source closure required')
    verify_hashes(ledger['execution_source_sha256'])
    current = [r for r in ledger['fits'] if r['nodeid'].startswith(TEST_SOURCE + '::')]
    if len(current) != 10 or any(r['status'] != 'completed' for r in current):
        raise ValueError('Exactly ten actual current-study tiny fits required')
    allowed = [TEST_SOURCE + '::test_' + s for s in [
        'four_new_formats_fit_audit_reuse_and_reject_corruption[',
        'six_reference_formats_through_new_destination[']]
    if any(not any(r['nodeid'].startswith(p) for p in allowed) for r in current):
        raise ValueError('Unexpected current-study tiny fit')
    if any(r['nodeid'] not in ledger['collected_nodes'] or r['status'] not in ['completed', 'failed'] for r in ledger['fits']):
        raise ValueError('Unattributed or unterminated classifier fit')
    historical = [r for r in ledger['fits'] if r not in current]
    receipt = dict(fingerprint=fingerprint(config_path), junit_sha256=digest(xml),
        tests_passed=len(cases), own_tests=len(own), actual_fit_ledger_sha256=digest(lp),
        actual_current_test_fits=10, actual_historical_test_fit_attempts=len(historical),
        actual_historical_test_fits=sum(r['status'] == 'completed' for r in historical),
        actual_historical_test_fit_failures=sum(r['status'] == 'failed' for r in historical),
        test_source_sha256=digest(TEST_SOURCE), created_at_utc=now())
    path = dest / 'test_fit_counts.json'
    invocations = read_config(path)['invocations'] if path.exists() else []
    if not any(r['junit_sha256'] == receipt['junit_sha256'] for r in invocations):
        invocations.append(receipt)
    write_json(path, dict(invocations=invocations,
        total_current_tiny_fits=sum(r['actual_current_test_fits'] for r in invocations)))
    write_json(out / 'checks.json', receipt)
    return receipt


def execution_gate(c, fp):
    check_receipt(c, fp); out = Path(c['output_dir'])
    for name in ['resource_preflight.json', 'source_miniature_gate.json', 'older_miniature_audit.json', 'miniature_gate.json']:
        r = read_config(out / name)
        if r['fingerprint'] != fp:
            raise ValueError('Missing/stale execution gate ' + name)
        verify_hashes(r.get('artifacts', {}))
    mini=read_config(out / 'miniature_gate.json')
    if (mini['fit_count'] != 16 or len(mini['attempts'])!=16
            or any(r['status']!='completed' for r in mini['attempts'])
            or not all(mini[k] is True for k in ['original_explicit_equivalent','same_seed_repeatable','shared_global_preprocessing'])):
        raise ValueError('Sixteen-fit conditional miniature gate required')


def _source_miniature_gate(c, fp):
    # The preserved sampling gate belongs to the seed study, not its fusion child.
    ancestor = read_config(c['parent_config'])
    sampling=read_config(ancestor['parent_config'])
    exported=Path(sampling['report_dir'])/'miniature_gate.json';sample=read_config(exported)
    if (sample['fingerprint']!=ancestor['parent_fingerprint'] or sample['fit_count']!=9
            or not all(sample.get(k) is True for k in ['same_seed_repeatable','inactive_seed_controls_equal','active_seed_diversity'])):
        raise ValueError('Preserved nine-fit sampling gate differs')
    expected=[(s['id'],s,True) for s in parent.scores.recipes()]+[
        ('repeat',parent.scores.recipes()[0],True),('inactive17',parent.scores.recipes()[0],False),
        ('inactive43',parent.scores.recipes()[2],False)]
    if len(sample['results'])!=9 or set(sample['files'])!={p for r in sample['results'] for p in r['files']}:
        raise ValueError('Preserved sampling artifact union differs')
    for r,(tag,spec,active) in zip(sample['results'],expected):
        if (r['tag'],r['spec'],r['active'])!=(tag,spec,active):
            raise ValueError('Preserved sampling recipe differs')
    sampling_present={p:h for p,h in sample['files'].items() if Path(p).exists()}
    if sampling_present and len(sampling_present)!=len(sample['files']):
        raise ValueError('Partial optional sampling miniature set')
    verify_hashes(sampling_present)
    if sampling_present:
        frame,base=parent.parent.synthetic_data();train=frame.index[frame.month<2];ev=frame.index[frame.month==2];values={}
        for r in sample['results']:
            model_path,prediction_path=list(r['files']);model=joblib.load(model_path)
            with np.load(prediction_path) as saved:values[r['tag']]=saved['scores']
            parent.parent.audit_preprocessing(model,base.loc[train])
            parent.parent.effective_parameters(model.named_steps['model'],r['spec'],c['threads'],active=r['active'])
            with threadpool_limits(limits=c['threads']):actual=model.predict_proba(base.loc[ev])[:,1]
            if not np.allclose(actual,values[r['tag']],atol=1e-12,rtol=0):
                raise ValueError('Preserved sampling saved inference differs')
        if not np.array_equal(values['repeat'],values['bagged_balanced_s17']) or not np.array_equal(values['inactive17'],values['inactive43']):
            raise ValueError('Preserved repeat/inactive sampling evidence differs')
        for w in scores.WEIGHTINGS:
            if all(np.array_equal(values[f'bagged_{w}_s17'],values[f'bagged_{w}_s{s}']) for s in [29,43]):
                raise ValueError('Preserved active sampling has no diversity')
    # Audit the older optional set as a whole and reconstruct its saved
    # preprocessing, objective and inference, rather than merely trusting hashes.
    fusion=read_config(c['fusion_source_config'])
    inherited_path=Path(fusion['report_dir'])/'source_miniature_gate.json';inherited=read_config(inherited_path)
    if inherited['fingerprint']!=c['fusion_source_fingerprint'] or inherited['preserved_fits']!=8:
        raise ValueError('Preserved older source gate differs')
    original=Path(inherited['source']);old_path=Path(str(original).replace('outputs/','reports/',1))
    older=read_config(old_path)
    if digest(old_path)!=inherited['sha256'] or older['fingerprint']!=c['interaction_source_fingerprint'] or older['fit_count']!=8:
        raise ValueError('Exported eight-fit ancestry identity differs')
    original_receipt={str(original):digest(original)} if original.exists() else {}
    if original_receipt and digest(original)!=inherited['sha256']:
        raise ValueError('Present optional older receipt differs')
    files={r['artifact']:r['sha256'] for r in older['results']}
    if len(files)!=8:raise ValueError('Eight distinct ancestry miniature artifacts required')
    present={p:h for p,h in files.items() if Path(p).exists()}
    if present and len(present)!=len(files):raise ValueError('Partial optional older miniature set')
    verify_hashes(present)
    if present:
        from . import interaction_study as interaction
        from .interaction_report import audit_statistics, audit_inference, independent_objective
        for r in older['results']:
            x,y=interaction.newton.miniature_frame()
            if not r['degenerate']:
                rng=np.random.default_rng(128)
                from .spline_inputs import NUMERIC
                for col in NUMERIC:x[col]=rng.normal(size=len(x))
            else:
                x['credit_risk_score']=123.;x['current_address_months_count']=np.nan
            train=x.iloc[:120];future=x.iloc[120:].copy()
            future.loc[future.index[0],'income']=np.nan
            future.loc[future.index[1],'device_os']='UNSEEN'
            future.loc[future.index[2],'customer_age']=1e12
            saved=joblib.load(r['artifact']);inp,model=saved['inputs'],saved['model']
            if saved['spec']!=r['spec']:raise ValueError('Older miniature recipe differs')
            audit_statistics(inp,train)
            computed=independent_objective(model,inp,train,y[:120],r['spec'])
            interaction.newton.check_numerical(model,[],r['numerical'],computed)
            audit_inference(inp,model,future)
    receipt=Path(c['output_dir'])/'older_miniature_audit.json'
    checked=dict(fingerprint=fp,source=str(old_path),source_sha256=digest(old_path),
        optional_artifacts_present=len(present),saved_models_audited=len(present),
        artifacts={str(old_path):digest(old_path),str(inherited_path):digest(inherited_path),**original_receipt,**present},new_model_fits=0)
    if receipt.exists() and read_config(receipt)!=checked:
        raise ValueError('Older miniature audit changed')
    write_json(receipt,checked)
    result=dict(fingerprint=fp,preserved_sampling_fits=9,preserved_older_fits=8,
        optional_sampling_artifacts_present=len(sampling_present),optional_older_artifacts_present=len(present),
        new_production_miniature_fits=0,older_ancestry=checked,
        artifacts={str(exported):digest(exported),str(receipt):digest(receipt),**sampling_present,**checked['artifacts']})
    path=Path(c['output_dir'])/'source_miniature_gate.json'
    if path.exists():
        old=read_config(path)
        if any(old.get(k)!=v for k,v in result.items()):raise ValueError('Preserved source gate changed')
        return old
    write_json(path,dict(**result,created_at_utc=now()))
    result=read_config(path)
    return result


def source_miniature_gate(c,fp):
    failure=Path(c['output_dir'])/'failures'/'source-miniature.json'
    if failure.exists():raise ValueError('Prior source-miniature failure retained')
    try:
        return _source_miniature_gate(c,fp)
    except Exception as exc:
        if not failure.exists():
            write_json(failure,dict(fingerprint=fp,error=repr(exc),new_production_miniature_fits=0,created_at_utc=now()))
        raise


def preflight(config_path=CONFIG):
    c, fp = initialize(config_path); check_receipt(c, fp)
    out = Path(c['output_dir']); path = out / 'resource_preflight.json'
    source_miniature_gate(c, fp)
    if path.exists():
        if read_config(path)['fingerprint'] != fp:
            raise ValueError('Resource gate identity changed')
        return read_config(path)
    failure = out / 'failures' / 'resource-preflight.json'
    if failure.exists():
        raise ValueError('Prior resource failure retained')
    start = time.perf_counter()
    try:
        from .spline_inputs import AdditiveInputs
        from .interaction_inputs import storage
        frame = load_data(c); frame = frame[frame.month < 7]; base = features(frame)
        additive = AdditiveInputs('spline').fit(base); sample = additive.transform(base.iloc[:8192])
        pre = make_model('lightgbm', False, dict(read_config('configs/study.json'), threads=c['threads']), base.columns).named_steps['preprocess'].fit(base)
        tree_sample = pre.transform(base.iloc[:8192]); n = len(base)
        d = max(len(additive.names), tree_sample.shape[1])
        matrix = 12 * n * d + 8 * (n + 1)
        workspace = 4 * matrix + 8 * (d + 1)**2 + 4 * 1024**3
        ram = (int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)) if sys.platform == 'darwin'
               else os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
        result = dict(fingerprint=fp, host_ram_bytes=ram, training_rows=n,
            training_months=list(range(7)), additive_dimension=len(additive.names),
            original_tree_dimension=tree_sample.shape[1], max_dimension=d, max_chunk_rows=8192,
            additive_sample=storage(sample), original_tree_sample=storage(tree_sample),
            matrix_upper_bytes=matrix, workspace_upper_bytes=workspace, allowance_fraction=.7,
            predictor_only=True, elapsed_seconds=time.perf_counter()-start,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if workspace > ram * .7:
            raise ValueError('Largest required allocation exceeds70% hostRAM')
        write_json(path, result); return result
    except Exception as exc:
        write_json(failure, dict(fingerprint=fp, error=repr(exc), created_at_utc=now())); raise


def miniature_data(columns):
    """Fixed synthetic full predictor schema; original eligibility minima apply."""
    rng = np.random.default_rng(882); n = 36000; ids = np.arange(n)
    raw = pd.DataFrame({name: rng.uniform(0, 10, n) for name in columns if name not in CATEGORICAL})
    for col, values in {'payment_type':['AA','AB'], 'housing_status':['BA','BB'],
            'employment_status':['CA','CB','CC'], 'source':['INTERNET','TELEAPP'],
            'device_os':['windows','linux','macintosh']}.items():
        divisor = 2 if col == 'housing_status' else 1
        raw[col] = np.asarray(values, dtype=object)[(ids // divisor) % len(values)]
    raw = raw[list(columns)]
    numeric = [v for v in columns if v not in CATEGORICAL]
    for i, name in enumerate(numeric):
        raw.loc[ids % (23 + i) == 0, name] = np.nan
    signal = raw[numeric[0]].fillna(0).to_numpy()
    label = (signal + rng.normal(0, 2, n) > np.where(ids % 2 == 0, 7., 10.)).astype(int)
    frame = raw.assign(month=ids // 12000, fraud_bool=label)
    for offset, pair in enumerate([['AE','BF'], [None,None], ['unseen','unseen']]):
        frame.loc[24000 + offset, ['payment_type','housing_status']] = pair
    raw[['payment_type','housing_status']] = frame[['payment_type','housing_status']]
    return frame, raw


def miniature(config_path=CONFIG):
    c, fp = initialize(config_path); check_receipt(c, fp); out = Path(c['output_dir'])
    for name in ['resource_preflight.json', 'source_miniature_gate.json']:
        if read_config(out / name)['fingerprint'] != fp:
            raise ValueError('Resource/source gates required before miniature fits')
    path = out / 'miniature_gate.json'; failure = out / 'failures' / 'miniature.json'
    if path.exists():
        r = read_config(path)
        if r['fingerprint'] != fp or r['fit_count'] != 16:
            raise ValueError('Conditional miniature identity changed')
        verify_hashes(r['artifacts']); return r
    if failure.exists() or (out / 'miniature_progress.json').exists():
        raise ValueError('Prior miniature attempt retained; no automatic retry')
    frame, base = miniature_data(pd.read_csv(c['data_path'], nrows=0).drop(columns=['fraud_bool','month','device_fraud_count']).columns)
    train = frame.index[frame.month < 2]; ev = frame.index[frame.month == 2]
    original = dict(read_config('configs/study.json'), threads=c['threads'])
    pre = make_model('lightgbm', False, original, base.columns).named_steps['preprocess'].fit(base.loc[train])
    matrix = pre.transform(base.loc[train]); results = []; attempts = []; artifacts = {}; values = {}
    start = time.perf_counter(); folder = out / 'miniature'
    write_json(out / 'miniature_progress.json', dict(fingerprint=fp, attempts=attempts))
    def fit(tag, weighting, rounds, spec=None, original_cost=False):
        earlier, future = (train, ev) if spec is None else scores.membership(frame, spec, 2)
        model = (make_model('lightgbm', weighting == 'balanced', original, base.columns).named_steps['model']
                 if original_cost else scores.estimator(c, rounds))
        pipe = Pipeline([('preprocess', pre), ('model', model)])
        y = frame.loc[earlier, 'fraud_bool'].to_numpy()
        costs = scores.class_costs(frame.loc[train, 'fraud_bool'], weighting)
        row = dict(tag=tag, status='started', weighting=weighting, rounds=rounds,
                   train_rows=len(earlier), started_at_utc=now()); attempts.append(row)
        write_json(out / 'miniature_progress.json', dict(fingerprint=fp, attempts=attempts))
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=c['threads']):
            warnings.simplefilter('always')
            kwargs = {} if original_cost else {'sample_weight':scores.weights_for(y, costs)}
            model.fit(matrix[train.get_indexer(earlier)], y, **kwargs)
            row['status'] = 'completed'
            v = pipe.predict_proba(base.loc[future])[:,1]
        messages = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if messages:
            raise ValueError('Miniature warning: ' + repr(messages))
        checked = copy.copy(model)
        if original_cost:
            expected = make_model('lightgbm', weighting == 'balanced', original, base.columns).named_steps['model']
            if scores.signature(model) != scores.signature(expected):
                raise ValueError('Original miniature recipe changed')
            checked.set_params(class_weight=None)
        effective = scores.checked_booster(c, checked, rounds)
        scores.probability(v); scores.bagged.audit_preprocessing(pipe, base.loc[train])
        folder.mkdir(parents=True, exist_ok=True); model_path = folder / (tag + '.joblib'); pred_path = folder / (tag + '.npz')
        joblib.dump(pipe, model_path)
        np.savez_compressed(pred_path, row_ids=future.to_numpy(), labels=frame.loc[future,'fraud_bool'].to_numpy(), scores=v)
        loaded = joblib.load(model_path)
        with threadpool_limits(limits=c['threads']):
            actual = loaded.predict_proba(base.loc[future])[:,1]
        if not np.allclose(v, actual, atol=1e-12, rtol=0):
            raise ValueError('Miniature persisted inference differs')
        files = {str(p):digest(p) for p in [model_path, pred_path]}; artifacts.update(files)
        results.append(dict(tag=tag, weighting=weighting, spec=spec, original_class_weight=original_cost,
            class_costs=costs, local_rows=len(earlier), local_fraud=int(y.sum()),
            global_train_ids_sha256=scores.ids_digest(train.to_numpy()),
            local_train_ids_sha256=scores.ids_digest(earlier.to_numpy()),
            local_weight_sha256=scores.weights_digest(scores.weights_for(y,costs)),
            preprocessing=scores.bagged.audit_preprocessing(pipe,base.loc[train]),
            requested_model_params=scores.signature(model), booster_params=effective,
            fitted_tree_count=model.booster_.current_iteration(), warnings=messages, files=files))
        values[tag] = dict(row_ids=future.to_numpy(), labels=frame.loc[future,'fraud_bool'].to_numpy(), scores=v)
        write_json(out / 'miniature_progress.json', dict(fingerprint=fp, attempts=attempts, results=results))
    try:
        for w in scores.WEIGHTINGS:
            fit('original200_' + w, w, 200, original_cost=True)
            fit('explicit200_' + w, w, 200)
            fit(scores.capacity(w), w, 400)
            if not np.allclose(values['original200_'+w]['scores'], values['explicit200_'+w]['scores'], atol=1e-12,rtol=0):
                raise ValueError('Original/explicit global weighting differs')
        specs = [s for s in scores.model_specs(frame,2) if s['kind']=='local']
        if len(specs) != 8 or any(g['train_rows'] != 12000 for p in scores.PARTITIONS for g in scores.eligibility(frame,2,p) if g['eligible']):
            raise ValueError('Original miniature eligibility minima not met')
        for spec in specs:
            fit(spec['id'], spec['weighting'], 200, spec)
        repeated = next(s for s in specs if s['weighting']=='balanced')
        fit('repeat_expert', 'balanced', 200, repeated); fit('repeat_capacity', 'balanced', 400)
        if not np.array_equal(values['repeat_expert']['scores'], values[repeated['id']]['scores']) or not np.array_equal(values['repeat_capacity']['scores'], values[scores.capacity('balanced')]['scores']):
            raise ValueError('Miniature deterministic repeat differs')
        routing = []
        for p,w in product(scores.PARTITIONS, scores.WEIGHTINGS):
            groups = [s for s in specs if s['partition']==p and s['weighting']==w]
            d, routes = scores.stitch(frame,2,p,values['original200_'+w],{s['level']:values[s['id']] for s in groups},[s['level'] for s in groups])
            routing.extend([dict(r,weighting=w) for r in routes])
            if set(r['route'] for r in routes) != {'expert','ineligible','missing','unseen'}:
                raise ValueError('Miniature fallback coverage differs')
            for a in [.25,.5,.75]: scores.blend(values['original200_'+w],d,a)
        if len(attempts)!=16 or any(r['status']!='completed' for r in attempts):
            raise ValueError('Sixteen completed miniature fits required')
        result = dict(fingerprint=fp, fit_count=len(attempts), results=results, routing=routing,
            attempts=attempts, artifacts=artifacts, original_explicit_equivalent=True,
            same_seed_repeatable=True, shared_global_preprocessing=True,
            training_rows=24000, evaluation_rows=12000, predictor_columns=list(base),
            eligibility_minimum=scores.MINIMUM, elapsed_seconds=time.perf_counter()-start,
            created_at_utc=now())
        write_json(path,result); return result
    except Exception as exc:
        write_json(failure,dict(fingerprint=fp,error=repr(exc),attempts=attempts,
            completed_fits=sum(r['status']=='completed' for r in attempts),artifacts=artifacts,created_at_utc=now())); raise


def work_accounting(c, records, derived_records, work):
    old_specs = {s['id']:s for s in required_sources(c)}
    refs = [r for r in records if r['candidate'] in old_specs]
    new = [r for r in records if r['candidate'] not in old_specs]
    def kind(r):
        s=old_specs[r['candidate']]
        return 'bagged' if s['id'].startswith('bagged_') else 'spline' if s.get('representation')=='spline' else 'tree'
    old_derived = [r for r in derived_records if r['candidate'] in parent.scores.dependencies([n for n,k in scores.REFERENCE_ROLES.values()])['derived']]
    return dict(base_results=len(records), reference_base_results=len(refs),
        reference_reuses=sum(bool(r.get('reused')) for r in refs),
        **{'reproduced_'+k+'_fits':sum(kind(r)==k and not r.get('reused',False) for r in refs) for k in ['bagged','spline','tree']},
        local_fits=sum(r['spec']['kind']=='local' for r in new),
        capacity_fits=sum(r['spec']['kind']=='capacity' for r in new),
        derived_results=len(derived_records), reference_derived_results=len(old_derived),
        reference_derived_reuses=sum(bool(r['reused']) for r in old_derived),
        new_compositions=len(derived_records)-len(old_derived),
        conditional_train_row_rounds=sum(r['train_row_rounds'] for r in new),
        conditional_stored_trees=sum(r['fitted_tree_count'] for r in new),
        conditional_original_fit_seconds=sum(r['fit_seconds'] for r in new),
        invocation_new_conditional_fits=sum(r['work']=='fit conditional recipe' for r in work),
        invocation_exact_reference_fits=sum(r['work']=='fit exact recipe' for r in work), invocation_work=work)


def inference_round_work(frame,month,partition,weighting,global_rounds,local_rounds):
    """Actual shared computation: global scores for all rows, locals for members."""
    if not isinstance(global_rounds,int) or global_rounds<1 or any(not isinstance(v,int) or v<1 for v in local_rounds.values()):
        raise ValueError('Positive actual fitted round counts required')
    ev=frame.index[frame.month==month];b=group_buckets(frame,ev,partition);rows=[]
    eligible={g['level'] for g in scores.eligibility(frame,month,partition) if g['eligible']}
    if set(local_rounds)!=eligible:raise ValueError('Exact eligible local-round evidence required')
    for level in scores.PARTITIONS[partition]+['__missing__','__unseen__']:
        n=int(np.sum(b==level));extra=local_rounds.get(level,0)
        rows.append(dict(eval_month=month,partition=partition,weighting=weighting,level=level,
            rows=n,global_fitted_rounds=global_rounds,local_fitted_rounds=extra,
            computed_rounds_per_row=global_rounds+extra,global_row_rounds=n*global_rounds,
            local_row_rounds=n*extra,total_row_rounds=n*(global_rounds+extra),
            shared_pure_and_blend_scores=True))
    return rows


def measured_work_tables(c,frame,records,phase='validation'):
    new=[r for r in records if r['candidate'].startswith(('expert_model_','capacity400_'))]
    model_rows=[];inference=[]
    for r in new:
        spec=r['spec']
        model_rows.append(dict(candidate=r['candidate'],eval_month=r['eval_month'],kind=spec['kind'],
            weighting=spec['weighting'],partition=spec.get('partition',''),level=spec.get('level',''),
            requested_rounds=spec['rounds'],fitted_tree_count=r['fitted_tree_count'],
            **{k:r[k] for k in ['global_train_rows','global_train_fraud','local_train_rows','local_train_fraud',
                'train_row_rounds','fit_seconds','peak_process_rss_bytes','global_weight_sha256','local_weight_sha256']},
            transformed_dimension=r['preprocessing']['transformed_dimension'],
            warnings=json.dumps(r['warnings']),requested_model_params=json.dumps(r['requested_model_params'],sort_keys=True),
            booster_params=json.dumps(r['booster_params'],sort_keys=True)))
    local=[r for r in new if r['spec']['kind']=='local']
    groups=sorted({(r['eval_month'],r['spec']['partition'],r['spec']['weighting']) for r in local})
    specs={s['id']:s for s in required_sources(c)}
    for m,p,w in groups:
        global_path=parent.source_paths(c,specs[scores.global200(w)],m,phase)[2]
        global_model=joblib.load(global_path)['model'].named_steps['model']
        actual=int(global_model.booster_.current_iteration())
        local_rounds={r['spec']['level']:r['fitted_tree_count'] for r in local if (r['eval_month'],r['spec']['partition'],r['spec']['weighting'])==(m,p,w)}
        inference.extend(inference_round_work(frame,m,p,w,actual,local_rounds))
    return pd.DataFrame(model_rows),pd.DataFrame(inference)


def exact_artifact_grid(c, frame, phase, names=None):
    names = [s['id'] for s in scores.strategies()] if names is None else names
    months = [4,5,6] if phase=='validation' else [7]; wanted_scores=set(); wanted_models=set(); object_count=0
    old_specs={s['id']:s for s in required_sources(c)}
    for m in months:
        dag=scores.dependencies(names,frame,m)
        model_names=set(dag['old_bases']) | {s['id'] for s in dag['new_models']}
        object_count+=len(model_names)
        score_names=set(dag['full_scores']) | {s['id'] for s in dag['new_models'] if s['kind']=='local'}
        wanted_scores.update(f'{phase}-m{m}-{n}' for n in score_names)
        for n in dag['old_bases']:
            wanted_models.update(p.name for p in parent.source_paths(c,old_specs[n],m,phase)[2:])
        for s in dag['new_models']:
            wanted_models.add(scores.model_paths(c,phase,m,s['id'])[2].name)
    out=Path(c['output_dir'])
    for folder,suffix,wanted in [('runs','.json',wanted_scores),('predictions','.npz',wanted_scores)]:
        if {p.stem for p in (out/folder).glob(phase+'-*'+suffix)}!=wanted:
            raise ValueError('Exact bounded artifact grid differs: '+folder)
    if {p.name for p in (out/'models').glob(phase+'-*') if p.is_file()}!=wanted_models:
        raise ValueError('Exact bounded model artifact grid differs')
    return dict(model_artifacts=object_count, model_files=len(wanted_models), prediction_artifacts=len(wanted_scores))


def prepare_development(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Development is frozen')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    records,ds,work=prepare(c,fp,frame,base)
    result=work_accounting(c,records,ds,work)
    if (result['base_results'],result['reference_base_results'],result['local_fits'],result['capacity_fits'],result['derived_results'])!=(84,24,54,6,63):
        raise ValueError('Complete required development DAG differs')
    grid=exact_artifact_grid(c,frame,'validation')
    if grid!={'model_artifacts':84,'model_files':87,'prediction_artifacts':147}:
        raise ValueError('Full/subset development accounting differs')
    result.update(fingerprint=fp,elapsed_seconds=time.perf_counter()-start,**grid)
    measured=measured_work_tables(c,frame,records)
    if (len(measured[0]),len(measured[1]))!=(60,96):raise ValueError('Measured model/inference work grid differs')
    for name,table in zip(WORK_TABLES,measured):
        data=table.to_csv(index=False).encode();path=out/name
        if path.exists() and path.read_bytes()!=data:raise ValueError('Measured work evidence changed')
        if not path.exists():path.write_bytes(data)
    result['work_files']={n:digest(out/n) for n in WORK_TABLES}
    path=out/'preparation.json'
    if path.exists():
        old=read_config(path)
        for k,v in result.items():
            if k not in ['invocation_work','invocation_new_conditional_fits','invocation_exact_reference_fits','elapsed_seconds'] and old[k]!=v:
                raise ValueError('Preparation accounting changed')
        return old
    write_json(path,result);return result


def validate(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists() or (out/'validation_complete.json').exists():
        raise ValueError('Completed/frozen validation cannot be overwritten')
    if read_config(out/'preparation.json')['fingerprint']!=fp:
        raise ValueError('Preparation missing/stale')
    frame=load_data(c);frame=frame[frame.month<7]
    generated=tables(c,frame)
    for name,table in generated.items():table.to_csv(out/(name+'.csv'),index=False)
    write_json(out/'validation_complete.json',dict(fingerprint=fp,created_at_utc=now(),files={n:digest(out/n) for n in TABLES+WORK_TABLES}))
    from .conditional_expert_report import audit_development
    return audit_development(config_path)


def final_fit_bounds(c, choices, frame):
    """Freeze availability only in immutable origins, excluding own new caches."""
    dag=scores.dependencies([n for n,k in choices.values()],frame)
    specs={s['id']:s for s in required_sources(c)};available={};missing=dict(bagged=0,spline=0,tree=0)
    for n in dag['old_bases']:
        spec=specs[n];found=None
        for source,identity in parent.source_origins(c,spec):
            ps=parent.source_paths(source,spec,7,'exploratory')
            if cache_state(ps):found={str(p):digest(p) for p in ps};break
        available[n]=found
        if found is None:
            missing['bagged' if n.startswith('bagged_') else 'spline' if spec.get('representation')=='spline' else 'tree']+=1
    eligibility=[g for p in scores.PARTITIONS for g in scores.eligibility(frame,7,p)]
    return dict(unique_base_results=len(dag['old_bases'])+len(dag['new_models']),
        physical_scores=len(dag['full_scores']),subset_scores=sum(s['kind']=='local' for s in dag['new_models']),
        max_new_local_fits=sum(s['kind']=='local' for s in dag['new_models']),
        max_new_capacity_fits=sum(s['kind']=='capacity' for s in dag['new_models']),
        **{'max_reproduced_'+k+'_fits':v for k,v in missing.items()},
        original_availability=available,eligibility=eligibility)


def freeze(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    done=read_config(out/'validation_complete.json');audit=read_config(out/'independent_development_audit.json')
    if done['fingerprint']!=fp or audit['fingerprint']!=fp or audit['validation_sha256']!=digest(out/'validation_complete.json'):
        raise ValueError('Independent prefreeze audit missing/stale')
    if audit['actual_source_models_audited'] is not True or audit['saved_model_checks']!=84:
        raise ValueError('All 84 actual saved-model audits required')
    verify_hashes({str(out/p):h for p,h in done['files'].items()})
    w=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0]
    if (audit['winner'],audit['policy'])!=(str(w.candidate),str(w.policy)) or not np.isclose(audit['development_f2'],w.f2,atol=1e-12,rtol=0):
        raise ValueError('Independently audited actual winner differs')
    choices=scores.choices_for(str(w.candidate),str(w.policy));h=history(c)
    frame=load_data(c);frame=frame[frame.month<7]  # eligibility is training-only
    record=dict(created_at_utc=now(),fingerprint=fp,config=c,selected_candidate=str(w.candidate),
        selected_policy=str(w.policy),selected_development_f2=float(w.f2),choices=choices,
        rules={a:learn(h[n],k,7) for a,(n,k) in choices.items()},
        score_dependencies=scores.dependencies([n for n,k in choices.values()],frame),
        final_fit_bounds=final_fit_bounds(c,choices,frame),validation_files=done['files'],
        independent_audit_sha256=digest(out/'independent_development_audit.json'),
        artifacts={str(p):digest(p) for folder in ['runs','predictions','models'] for p in (out/folder).glob('validation-*') if p.is_file()},
        month7_previously_consumed=True,exploratory=True)
    write_json(out/'selection.json',record);return record


def load_selection(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['fingerprint']!=fp or s['config']!=c:raise ValueError('Frozen execution identity changed')
    verify_hashes(s['artifacts']);verify_hashes({str(out/p):h for p,h in s['validation_files'].items()})
    if s['independent_audit_sha256']!=digest(out/'independent_development_audit.json'):
        raise ValueError('Frozen independent audit changed')
    w=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0];roles=scores.choices_for(str(w.candidate),str(w.policy))
    frame=load_data(c);frame=frame[frame.month<7];h=history(c)
    if (s['selected_candidate'],s['selected_policy'])!=(str(w.candidate),str(w.policy)) or not np.isclose(s['selected_development_f2'],w.f2,atol=1e-12,rtol=0):
        raise ValueError('Actual macro winner differs')
    if s['choices']!=roles or s['score_dependencies']!=scores.dependencies([n for n,k in roles.values()],frame) or s['final_fit_bounds']!=final_fit_bounds(c,roles,frame):
        raise ValueError('Frozen roles/DAG/bounds differ')
    if any(s['rules'][a]!=learn(h[n],k,7) for a,(n,k) in roles.items()):
        raise ValueError('Final earlier-only rules differ')
    return c,s


def final_data(c,s):
    return {a:prediction(c,'exploratory',7,n) for a,(n,k) in s['choices'].items()}


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);execution_gate(c,s['fingerprint']);out=Path(c['output_dir'])
    sha=digest(out/'selection.json');path=out/'exploratory_access.json'
    if (out/'transfer_complete.json').exists():raise ValueError('Completed transfer cannot be overwritten')
    if not path.exists():write_json(path,dict(access_at_utc=now(),selection_sha256=sha,month7_previously_consumed=True))
    access=read_config(path)
    if access['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:
        raise ValueError('Freeze/access chronology differs')
    frame=load_data(c);base=features(frame);names={n for n,k in s['choices'].values()}
    records,ds,work=prepare(c,s['fingerprint'],frame,base,'exploratory',names)
    data=final_data(c,s);rows=[]
    for a,d in data.items():
        aligned(d,data['selected']);rule=s['rules'][a]
        rows.append(dict(role=a,candidate=s['choices'][a][0],policy=s['choices'][a][1],eval_month=7,
                         parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    counts=work_accounting(c,records,ds,work);bounds=s['final_fit_bounds']
    for k,b in [('local_fits','max_new_local_fits'),('capacity_fits','max_new_capacity_fits')]+[
            ('reproduced_'+t+'_fits','max_reproduced_'+t+'_fits') for t in ['bagged','spline','tree']]:
        if counts[k]>bounds[b]:raise ValueError('Frozen fit bound exceeded')
    grid=exact_artifact_grid(c,frame,'exploratory',names)
    if counts['base_results']!=bounds['unique_base_results'] or grid['prediction_artifacts']!=bounds['physical_scores']+bounds['subset_scores']:
        raise ValueError('Frozen transfer DAG work differs')
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    result=dict(fingerprint=s['fingerprint'],selection_sha256=sha,table_sha256=digest(out/'exploratory.csv'),
        role_count=len(rows),total_scores=bounds['physical_scores'],subset_scores=bounds['subset_scores'],**grid,**counts)
    write_json(out/'transfer_complete.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify-declaration','record-checks','preflight','miniature',
        'prepare-development','validate','freeze','exploratory'])
    parser.add_argument('--config',default=CONFIG);args=parser.parse_args()
    print(json.dumps(globals()[args.action.replace('-','_')](args.config),indent=2))


if __name__ == '__main__':
    main()
