"""Declared row-wise LightGBM additions to four preserved finite margins."""
from itertools import combinations, product

import numpy as np

from .recall_weighted_metrics import POLICIES, aligned, check_data
from .score_fusion import EPSILON, log_odds, clipping
from .interaction_fusion_scores import diversity
from .interaction_study import recipes as historical_recipes, FUSION, RECENT_FUSION

LAMBDAS = [.1, .25, .5, .75]
ADDED = {'balanced': 'baseline_balanced', 'unweighted': 'baseline_lgbm'}
ANCHORS = {
    'tree': FUSION,
    'recent': RECENT_FUSION,
    'latest': 'interaction_blend_spline_c1p0_balanced_a10',
    'interaction': 'interaction_blend_interaction_c1p0_balanced_a10',
}
ALIAS = 'logit_mean_xgb_d4_w1_baseline_balanced_a25'
BASES = ['xgb_d4_w1', *ADDED.values(), 'spline_c0p1_unweighted',
         'spline_c1p0_balanced', 'interaction_c1p0_balanced']
REFERENCE_ROLES = {
    'latest_fusion': [ANCHORS['latest'], 'latest'],
    'recent_fusion': [RECENT_FUSION, 'latest'],
    'fusion': [FUSION, 'pooled'],
    'balanced': ['baseline_balanced', 'pooled'],
    'unweighted': ['baseline_lgbm', 'pooled'],
}
ORIGINAL_POLICIES = {n: k for n, k in REFERENCE_ROLES.values()}


def recipes():
    return [s for s in historical_recipes() if s['id'] in BASES]


def strategy_id(anchor, added, weight):
    if anchor not in ANCHORS or added not in ADDED or isinstance(weight, bool) or weight not in LAMBDAS:
        raise ValueError('Undeclared anchor, added member or lambda')
    if (anchor, added, weight) == ('tree', 'balanced', .5):
        return ALIAS
    return f'lgbm_blend_{anchor}_{added}_l{int(100*weight)}'


def strategies():
    rows = [dict(id=FUSION, kind='anchor', anchor='tree',
                 members=['xgb_d4_w1', 'baseline_balanced'], alpha=.5)]
    for tag, member, alpha in [('recent', 'spline_c0p1_unweighted', .25),
                               ('latest', 'spline_c1p0_balanced', .1),
                               ('interaction', 'interaction_c1p0_balanced', .1)]:
        rows.append(dict(id=ANCHORS[tag], kind='anchor', anchor=tag,
                         members=[member, FUSION], alpha=alpha))
    rows += [dict(id=n, kind='native', members=[n], added=tag) for tag, n in ADDED.items()]
    rows += [dict(id='logit_'+n, kind='logit', members=[n], added=tag) for tag, n in ADDED.items()]
    for tag, added, weight in product(ANCHORS, ADDED, LAMBDAS):
        alias = (tag, added, weight) == ('tree', 'balanced', .5)
        rows.append(dict(id=strategy_id(tag, added, weight), kind='addition', anchor=tag,
                         added=added, weight=weight,
                         members=['xgb_d4_w1', 'baseline_balanced'] if alias
                                 else [ADDED[added], ANCHORS[tag]],
                         arithmetic='historical direct 0.25/0.75' if alias else 'weighted margin addition'))
    if len(rows) != 40 or len({s['id'] for s in rows}) != 40:
        raise ValueError('Canonical finite union differs')
    return rows


def lookup(name):
    result = next((s for s in strategies() if s['id'] == name), None)
    if result is None:
        raise ValueError('Undeclared strategy')
    return result


def compose(spec, items):
    if spec not in strategies() or spec['kind'] == 'native':
        raise ValueError('Undeclared derived recipe')
    ds = [items[n] for n in spec['members']]
    for d in ds:
        aligned(d, ds[0])
    u = log_odds(ds[0]['scores'], EPSILON)
    if spec['id'] == ALIAS:
        # Preserve old multiplication/addition association and tied decisions.
        u = .25*u + .75*log_odds(ds[1]['scores'], EPSILON)
    elif spec['kind'] == 'anchor' and spec['anchor'] == 'tree':
        u = .5*u + .5*log_odds(ds[1]['scores'], EPSILON)
    elif spec['kind'] != 'logit':
        g = np.asarray(ds[1]['scores'], dtype=float)
        if g.ndim != 1 or g.shape != u.shape or not np.isfinite(g).all():
            raise ValueError('Invalid finite anchor margin')
        weight = spec['alpha'] if spec['kind'] == 'anchor' else spec['weight']
        u = weight*u + (1-weight)*g
    result = dict(row_ids=ds[0]['row_ids'], labels=ds[0]['labels'], scores=u)
    check_data(result)
    return result


def leaf_coefficients(name):
    """Canonical coefficients on source logits, with repeated leaves combined."""
    s = lookup(name)
    if s['kind'] == 'native':
        return {name: 1.}
    if s['kind'] == 'logit':
        return {s['members'][0]: 1.}
    if name == ALIAS:
        return {'xgb_d4_w1': .25, 'baseline_balanced': .75}
    if name == FUSION:
        return {'xgb_d4_w1': .5, 'baseline_balanced': .5}
    if s['kind'] == 'anchor':
        weight, member, anchor = s['alpha'], s['members'][0], FUSION
    else:
        weight, member, anchor = s['weight'], ADDED[s['added']], ANCHORS[s['anchor']]
    coefficients = {k: (1-weight)*v for k, v in leaf_coefficients(anchor).items()}
    coefficients[member] = coefficients.get(member, 0.) + weight
    return coefficients


def dependencies(names):
    """Transitive DAG, retaining only actual computation dependencies."""
    found, active = set(), set()

    def visit(name):
        if name in found:
            return
        if name in active:
            raise ValueError('Cyclic score declaration')
        active.add(name)
        if name not in BASES:
            for member in lookup(name)['members']:
                visit(member)
        found.add(name)
        active.remove(name)

    for name in names:
        visit(name)
    bases = found & set(BASES)
    return dict(bases=sorted(bases), scores=sorted(found), derived=sorted(found-bases))


def choices_for(name, policy):
    s = lookup(name)
    if policy not in POLICIES:
        raise ValueError('Undeclared policy')
    roles = dict(selected=[name, policy], **{k: list(v) for k, v in REFERENCE_ROLES.items()},
                 same_strategy=[name, 'pooled'])
    if name in ORIGINAL_POLICIES:
        return roles
    if s['kind'] == 'anchor':
        roles['matched_anchor'] = [ANCHORS['latest'], policy]
    elif s['kind'] == 'logit':
        other = 'unweighted' if s['added'] == 'balanced' else 'balanced'
        roles.update(native_added_member=[ADDED[s['added']], policy],
                     matched_added_weight=['logit_'+ADDED[other], policy])
    elif s['kind'] == 'addition':
        other = 'unweighted' if s['added'] == 'balanced' else 'balanced'
        roles.update(native_added_member=[ADDED[s['added']], policy],
                     transformed_added_member=['logit_'+ADDED[s['added']], policy],
                     anchor_member=[ANCHORS[s['anchor']], policy],
                     matched_added_weight=[strategy_id(s['anchor'], other, s['weight']), policy])
        for weight in LAMBDAS:
            if weight != s['weight']:
                roles['other_lambda_'+str(int(100*weight))] = [strategy_id(s['anchor'], s['added'], weight), policy]
        if s['anchor'] in ['latest', 'interaction']:
            other_anchor = 'interaction' if s['anchor'] == 'latest' else 'latest'
            roles['matched_anchor'] = [strategy_id(other_anchor, s['added'], s['weight']), policy]
    else:
        raise ValueError('Unrecognized winner branch')
    return roles


def contrast_factors():
    rows = []
    for s, k in product(strategies(), POLICIES):
        rows += [('fixed operating reference', s['id'], k, ref, rk) for ref, rk in ORIGINAL_POLICIES.items()]
        if k != 'pooled':
            rows.append(('policy minus own pooled', s['id'], k, s['id'], 'pooled'))
    for a, b, w, k in product(ANCHORS, ADDED, LAMBDAS, POLICIES):
        for label, ref in [('addition minus anchor', ANCHORS[a]),
                           ('addition minus native', ADDED[b]),
                           ('addition minus transformed', 'logit_'+ADDED[b])]:
            rows.append((label, strategy_id(a, b, w), k, ref, k))
    for a, w, k in product(ANCHORS, LAMBDAS, POLICIES):
        rows.append(('unweighted minus balanced', strategy_id(a, 'unweighted', w), k,
                     strategy_id(a, 'balanced', w), k))
    for prefix, k in product(['', 'logit_'], POLICIES):
        rows.append(('unweighted minus balanced', prefix+ADDED['unweighted'], k, prefix+ADDED['balanced'], k))
    for b, w, k in product(ADDED, LAMBDAS, POLICIES):
        rows.append(('interaction minus additive anchor', strategy_id('interaction', b, w), k,
                     strategy_id('latest', b, w), k))
    for k in POLICIES:
        rows.append(('interaction minus additive anchor', ANCHORS['interaction'], k, ANCHORS['latest'], k))
    for a, b, (lo, hi), k in product(ANCHORS, ADDED, list(combinations(LAMBDAS, 2)), POLICIES):
        rows.append(('larger lambda minus smaller lambda', strategy_id(a, b, hi), k, strategy_id(a, b, lo), k))
    for n, k in product(ADDED.values(), POLICIES):
        rows.append(('transformed minus native', 'logit_'+n, k, n, k))
    if len(rows) != 1199 or len(set(rows)) != 1199:
        raise ValueError('Complete contrast union differs')
    return rows
