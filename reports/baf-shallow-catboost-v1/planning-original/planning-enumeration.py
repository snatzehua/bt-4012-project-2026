"""Read-only ancestry checks and prospective enumeration; no fitting or scoring."""
from collections import Counter
from itertools import combinations, product
from pathlib import Path
import importlib.metadata
import json
import sys

from baf import conditional_expert_study as parent
from baf.common import digest, read_config, write_json

ROOT = Path(__file__).resolve().parent
DEPTHS, ROUNDS, COSTS, ALPHAS = [3, 4, 6], [200, 400], [1, 100], [10, 25, 50]
POLICIES = ['pooled', 'latest', 'rank_fraction']
COMMON_PARAMS = dict(learning_rate=.05, l2_leaf_reg=5, border_count=128,
    max_ctr_complexity=1, loss_function='Logloss', task_type='CPU',
    boosting_type='Plain', grow_policy='SymmetricTree', random_seed=2026,
    thread_count=6, use_best_model=False, allow_writing_files=False, verbose=False)
CATEGORICAL = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
c = read_config(parent.CONFIG)
parent.guard(c)
assert c['seed'] == 2026 and c['threads'] == 6
REFS = parent.scores.REFERENCE_ROLES
ANCHOR = REFS['bagged_anchor'][0]
OLD = parent.parent.scores.dependencies([n for n, k in REFS.values()])
assert (len(REFS), len(OLD['bases']), len(OLD['scores'])) == (6, 8, 13)


def native(d, r, w):
    return f'cb_d{d}_r{r}_w{w}'


def candidate(d, r, w, a):
    if a == 0:
        return ANCHOR
    if a == 100:
        return native(d, r, w)
    assert a in ALPHAS
    return f'cb_anchor_d{d}_r{r}_w{w}_a{a:02d}'


RECIPES = [dict(id=native(d, r, w), family='catboost', engineered=False,
                depth=d, iterations=r, positive_class_cost=w,
                params=dict(COMMON_PARAMS, depth=d, iterations=r,
                            class_weights=[1, w], cat_features=CATEGORICAL))
           for d, r, w in product(DEPTHS, ROUNDS, COSTS)]
STRATEGIES = [dict(id=n, kind='reference') for n, k in REFS.values()]
STRATEGIES += [dict(id=s['id'], kind='native', depth=s['depth'],
                    iterations=s['iterations'], positive_class_cost=s['positive_class_cost'],
                    alpha_numerator=100, alpha_denominator=100) for s in RECIPES]
STRATEGIES += [dict(id=candidate(d, r, w, a), kind='blend', depth=d, iterations=r,
                    positive_class_cost=w, alpha_numerator=a, alpha_denominator=100,
                    members=[ANCHOR, native(d, r, w)])
               for d, r, w, a in product(DEPTHS, ROUNDS, COSTS, ALPHAS)]
LOOK = {s['id']: s for s in STRATEGIES}
assert len(LOOK) == len(STRATEGIES) == 54


def roles(n, k):
    s = LOOK[n]
    result = dict(selected=[n, k], **REFS, same_strategy=[n, 'pooled'])
    if s['kind'] == 'reference':
        return result
    d, r, w, a = [s[x] for x in ['depth', 'iterations', 'positive_class_cost', 'alpha_numerator']]
    result.update(anchor_at_policy=[ANCHOR, k], pure_catboost=[native(d, r, w), k])
    result.update({f'depth_d{v}': [candidate(v, r, w, a), k] for v in DEPTHS})
    result.update({f'rounds_r{v}': [candidate(d, v, w, a), k] for v in ROUNDS})
    result.update({f'cost_w{v}': [candidate(d, r, v, a), k] for v in COSTS})
    result.update({f'alpha_a{v:03d}': [candidate(d, r, w, v), k] for v in [0, *ALPHAS, 100]})
    return result


def dependencies(role_map):
    new_bases, new_blends = set(), set()
    for n, k in role_map.values():
        s = LOOK[n]
        if s['kind'] == 'reference':
            continue
        new_bases.add(native(s['depth'], s['iterations'], s['positive_class_cost']))
        if s['kind'] == 'blend':
            new_blends.add(n)
    return dict(old_bases=OLD['bases'], old_scores=OLD['scores'],
                old_derived=OLD['derived'], catboost_bases=sorted(new_bases),
                new_blends=sorted(new_blends),
                full_scores=sorted(set(OLD['scores']) | new_bases | new_blends))


# The original bagged-anchor individual adapters retain their own ancestry.
# This is a destination/ancestry view, never a declaration passed to its guard.
adapter_config = c['parent_config']
adapter_fp = c['parent_fingerprint']
adapter_source = read_config(adapter_config)
assert adapter_config == parent.parent.CONFIG
assert adapter_fp == parent.parent.fingerprint(adapter_config)
specs = {s['id']: s for s in parent.required_sources(c)}
source_state = []
for phase, months in [('validation', [4, 5, 6]), ('exploratory', [7])]:
    for m, n in product(months, OLD['bases']):
        spec = specs[n]
        origins = [(adapter_source, adapter_fp)] + parent.parent.source_origins(adapter_source, spec)
        observed, available = [], False
        # Record only origins actually inspected, matching first-complete precedence.
        for origin, fp in origins:
            paths = parent.parent.source_paths(origin, spec, m, phase)
            complete = parent.cache_state(paths)  # rejects a partial required set
            observed.append(dict(output_dir=origin['output_dir'], complete=complete,
                                 paths=[str(p) for p in paths]))
            if complete:
                available = True
                break
        source_state.append(dict(phase=phase, eval_month=m, candidate=n,
                                 complete_origin_present=available, inspected_origins=observed))
assert len(source_state) == 32 and all(s['complete_origin_present'] for s in source_state)
# Availability is not saved-inference verification; the apply phase must audit it.

CASES = []
for s, k in product(STRATEGIES, POLICIES):
    role_map = roles(s['id'], k)
    dag = dependencies(role_map)
    kind = s['kind']
    assert len(role_map) == (8 if kind == 'reference' else 22)
    assert len(dag['catboost_bases']) == (0 if kind == 'reference' else 5)
    assert len(dag['new_blends']) == dict(reference=0, native=3, blend=7)[kind]
    assert len(dag['full_scores']) == dict(reference=13, native=21, blend=25)[kind]
    pairs = set(tuple(v) for v in role_map.values())
    reference_types = dict(bagged=3, spline=2, tree=3)
    CASES.append(dict(winner=s['id'], policy=k, kind=kind, roles=role_map,
        dependencies=dag, visible_roles=len(role_map), unique_operating_pairs=len(pairs),
        paired_interval_rows=4*(len(role_map)-1),
        available_cache_reference_reproduction_bound=dict(bagged=0, spline=0, tree=0),
        wholly_absent_reference_reproduction_bounds=reference_types,
        new_catboost_fit_bound=len(dag['catboost_bases']),
        model_object_bound=8+len(dag['catboost_bases']),
        model_file_bound=9+2*len(dag['catboost_bases']),
        physical_score_bound=len(dag['full_scores'])))
assert len(CASES) == 162

CONTRASTS = []


def add(kind, n, k, b, bk, **factors):
    CONTRASTS.append(dict(kind=kind, candidate=n, policy=k, comparator=b,
                          comparator_policy=bk, factors=factors))


for s, k in product(STRATEGIES, POLICIES):
    for role, (n, nk) in REFS.items():
        add('fixed_reference', s['id'], k, n, nk, reference=role)
    if k != 'pooled':
        add('own_policy', s['id'], k, s['id'], 'pooled')
for r, w, a, k in product(ROUNDS, COSTS, [*ALPHAS, 100], POLICIES):
    for lo, hi in combinations(DEPTHS, 2):
        add('matched_depth', candidate(hi, r, w, a), k, candidate(lo, r, w, a), k,
            lower_depth=lo, higher_depth=hi, iterations=r, positive_class_cost=w, alpha=a)
for d, w, a, k in product(DEPTHS, COSTS, [*ALPHAS, 100], POLICIES):
    add('matched_rounds', candidate(d, 400, w, a), k, candidate(d, 200, w, a), k,
        depth=d, positive_class_cost=w, alpha=a)
for d, r, a, k in product(DEPTHS, ROUNDS, [*ALPHAS, 100], POLICIES):
    add('matched_cost', candidate(d, r, 100, a), k, candidate(d, r, 1, a), k,
        depth=d, iterations=r, alpha=a)
for d, r, w, k in product(DEPTHS, ROUNDS, COSTS, POLICIES):
    add('native_vs_anchor', native(d, r, w), k, ANCHOR, k)
    for a in ALPHAS:
        for kind, b in [('blend_vs_native', native(d, r, w)), ('blend_vs_anchor', ANCHOR)]:
            add(kind, candidate(d, r, w, a), k, b, k)
    for lo, hi in combinations(ALPHAS, 2):
        add('higher_alpha', candidate(d, r, w, hi), k, candidate(d, r, w, lo), k,
            lower_alpha=lo, higher_alpha=hi)
assert len(CONTRASTS) == len({json.dumps(v, sort_keys=True) for v in CONTRASTS}) == 1728
contrast_counts = dict(sorted(Counter(v['kind'] for v in CONTRASTS).items()))
assert contrast_counts == dict(blend_vs_anchor=108, blend_vs_native=108, fixed_reference=972,
    higher_alpha=108, matched_cost=72, matched_depth=144, matched_rounds=72,
    native_vs_anchor=36, own_policy=108)

preserve = dict(c['preserve_sha256'])
for p in parent.SOURCES[-3:] + [parent.CONFIG, c['protocol_path'],
        str(Path(c['output_dir'])/'selection.json'), str(Path(c['report_dir'])/'report_hashes.json')]:
    preserve[p] = digest(p)
preserve.update(read_config(Path(c['report_dir'])/'report_hashes.json'))
parent.verify_hashes(preserve)
selections = [p for p in preserve if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(preserve) == 1860 and len(selections) == 18
assert all(preserve[p] == preserve[p.replace('outputs/', 'reports/', 1)] for p in selections)
roots = sorted(set(c['historical_output_roots'] + [c['output_dir']]))
snapshot = {str(p): digest(p) for root in roots for p in Path(root).rglob('*') if p.is_file()}
assert len(snapshot) == 4848
sources = parent.SOURCES + ['baf/shallow_catboost.py', 'baf/shallow_catboost_study.py',
                           'baf/shallow_catboost_report.py']
assert len(sources) == len(set(sources)) == 61
pins = {}
for name in ['requirements.txt', 'requirements-improvements.txt', 'requirements-xgboost.txt']:
    for line in Path(name).read_text().splitlines():
        if '==' in line:
            package, version = line.split('==')
            assert importlib.metadata.version(package) == version
            pins[package] = version
assert len(pins) == 27 and sys.version.split()[0] == '3.12.9'
assert digest(c['data_path']) == c['data_sha256']

old_catboost = []
for name in ['configs/improvements.json', 'configs/recall-weighted.json',
             'configs/recall-weighted-extension.json', 'configs/chronological-stacking.json']:
    data = read_config(name)
    for s in data.get('candidates', []):
        if s.get('family') == 'catboost':
            old_catboost.append(dict(config=name, config_sha256=digest(name), recipe=s))
assert old_catboost and all(s['recipe']['params']['iterations'] not in ROUNDS for s in old_catboost)

physical_names = sorted(set(OLD['scores']) | {s['id'] for s in STRATEGIES})
assert len(physical_names) == 61
row_identities = dict(
    physical_scores=[[m, n] for m, n in product([4, 5, 6], physical_names)],
    selectable_scores=[[m, s['id']] for m, s in product([4, 5, 6], STRATEGIES)],
    forward=[[m, s['id'], k] for m, s, k in product([5, 6], STRATEGIES, POLICIES)],
    macro=[[s['id'], k] for s, k in product(STRATEGIES, POLICIES)],
    pooled=[[s['id'], k] for s, k in product(STRATEGIES, POLICIES)],
    models=[[m, n] for m, n in product([4, 5, 6], OLD['bases'] + [r['id'] for r in RECIPES])],
    new_parameters=[[m, r['id']] for m, r in product([4, 5, 6], RECIPES)],
    diversity=[[m, r['id'], ANCHOR] for m, r in product([4, 5, 6], RECIPES)],
    overlap=[], curves=[])
for m, s, k in product([5, 6], STRATEGIES, POLICIES):
    if s['kind'] != 'reference':
        comparators = [ANCHOR]
        if s['kind'] == 'blend':
            comparators.append(native(s['depth'], s['iterations'], s['positive_class_cost']))
        row_identities['overlap'] += [[m, s['id'], k, n] for n in comparators]
for d, r, w, k, a in product(DEPTHS, ROUNDS, COSTS, POLICIES, [0, *ALPHAS, 100]):
    row_identities['curves'].append([d, r, w, k, a, candidate(d, r, w, a)])
assert {n: len(v) for n, v in row_identities.items()} == dict(
    physical_scores=183, selectable_scores=162, forward=324, macro=162, pooled=162,
    models=60, new_parameters=36, diversity=36, overlap=504, curves=180)
assert all(len(v) == len({tuple(row) for row in v}) for v in row_identities.values())

counts = dict(selectable_strategies=54, strategy_policy_cases=162,
    new_catboost_recipes=12, new_native_score_months=36, new_blend_score_months=108,
    reference_base_results=24, reference_derived_results=15, physical_full_scores=183,
    selectable_score_months=162, available_cache_model_objects=60, available_cache_model_files=99,
    new_full_development_fits=36, forward_rows=324, macro_rows=162, pooled_rows=162,
    contrasts=1728, physical_score_diagnostic_rows=183, selectable_score_diagnostic_rows=162,
    native_anchor_diversity_rows=36, decision_overlap_rows=504,
    model_work_rows=60, new_preprocessing_records=36, new_model_parameter_rows=36,
    operating_points=162, alpha_curves=36, canonical_alpha_curve_rows=180,
    production_catboost_miniature_fits=14, current_tiny_fits_per_complete_invocation=18,
    expected_historical_tiny_calls_per_complete_invocation=85,
    paired_draws=1500)
write_json(ROOT/'planning-evidence.json', dict(
    parent_config=parent.CONFIG, parent_fingerprint=parent.fingerprint(),
    parent_selection_sha256=digest(Path(c['output_dir'])/'selection.json'),
    parent_report_manifest_sha256=digest(Path(c['report_dir'])/'report_hashes.json'),
    reference_adapter_config=adapter_config, reference_adapter_fingerprint=adapter_fp,
    mandatory_preservation_sha256=preserve, historical_output_roots=roots,
    historical_ignored_file_sha256=snapshot, planned_source_paths=sources,
    reference_roles=REFS, reference_dag=OLD, reference_cache_presence=source_state,
    catboost_common_params=COMMON_PARAMS, categorical_columns=CATEGORICAL,
    recipes=RECIPES, strategies=STRATEGIES, winner_policy_cases=CASES,
    contrasts=CONTRASTS, contrast_kind_counts=contrast_counts, expected_counts=counts,
    row_identity_sets=row_identities,
    old_catboost_candidates=old_catboost, verified_pins=pins, python=sys.version.split()[0],
    data_sha256=c['data_sha256'], new_model_fits=0, new_score_evaluations=0,
    scope='Prospective enumeration, ancestry hashes and cache presence only. No fitting, inference, policy evaluation or month7 outcome read. Cache presence does not prove saved inference.'))
print(json.dumps(dict(counts=counts, contrasts=contrast_counts,
    preservation=dict(mandatory=len(preserve), historical=len(snapshot), selection_pairs=len(selections)),
    case_shapes=sorted({(v['kind'], v['visible_roles'], v['new_catboost_fit_bound'],
        v['physical_score_bound'], v['paired_interval_rows']) for v in CASES})), indent=2))
