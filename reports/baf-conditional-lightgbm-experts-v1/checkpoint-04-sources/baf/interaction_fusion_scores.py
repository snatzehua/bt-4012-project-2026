"""Finite fixed score compositions; labels never enter score arithmetic."""
from itertools import combinations, product

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .recall_weighted_metrics import POLICIES, aligned, check_data
from .score_fusion import log_odds, clipping
from .interaction_study import recipes, FUSION, RECENT_FUSION, ORIGINAL_POLICIES

EPSILON = 1e-6
ALPHAS = [.1, .25, .5, .75]
FORMS = ['native', 'logit', *ALPHAS]
MEMBERS = [s['id'] for s in recipes()]


def strategy_id(member, form):
    if member not in MEMBERS or form not in FORMS:
        raise ValueError('Undeclared member or score form')
    if form == 'native':
        return member
    if form == 'logit':
        return 'logit_' + member
    if member == 'spline_c0p1_unweighted' and form == .25:
        return RECENT_FUSION
    return f'interaction_blend_{member}_a{int(100*form)}'


def strategies():
    rows = []
    for member, form in product(MEMBERS, FORMS):
        rows.append(dict(id=strategy_id(member, form),
                         kind='native' if form == 'native' else 'logit' if form == 'logit' else 'blend',
                         members=[member] if isinstance(form, str) else [member, FUSION],
                         form=form, **({} if isinstance(form, str) else dict(alpha=form))))
    rows += [dict(id=n, kind='reference', members=[n])
             for n in [FUSION, 'baseline_balanced', 'baseline_lgbm']]
    if len(rows) != 51 or len({s['id'] for s in rows}) != 51:
        raise ValueError('Canonical strategy union differs')
    return rows


def lookup(name):
    found = next((s for s in strategies() if s['id'] == name), None)
    if found is None:
        raise ValueError('Undeclared strategy')
    return found


def compose(spec, items):
    if spec not in strategies() or spec['kind'] not in ['logit', 'blend']:
        raise ValueError('Undeclared derived recipe')
    ds = [items[n] for n in spec['members']]
    for d in ds:
        aligned(d, ds[0])
    u = log_odds(ds[0]['scores'], EPSILON)
    if spec['kind'] == 'blend':
        g = np.asarray(ds[1]['scores'], dtype=float)
        if g.ndim != 1 or g.shape != u.shape or not np.isfinite(g).all():
            raise ValueError('Invalid tree margin')
        u = spec['alpha'] * u + (1-spec['alpha']) * g
    result = dict(row_ids=ds[0]['row_ids'], labels=ds[0]['labels'], scores=u)
    check_data(result)
    return result


def matched(member, factor):
    original = next((s for s in recipes() if s['id'] == member), None)
    if original is None:
        raise ValueError('Unknown source member')
    target = dict(original)
    if factor == 'representation':
        target[factor] = 'interaction' if original[factor] == 'spline' else 'spline'
    elif factor == 'class_weight':
        target[factor] = None if original[factor] == 'balanced' else 'balanced'
    elif factor == 'C':
        target[factor] = .1 if original[factor] == 1. else 1.
    else:
        raise ValueError('Unknown matching factor')
    return next(s['id'] for s in recipes()
                if all(s[k] == target[k] for k in ['representation', 'C', 'class_weight']))


def choices_for(name, policy):
    s = lookup(name)
    if policy not in POLICIES:
        raise ValueError('Undeclared policy')
    roles = dict(selected=[name, policy], recent_fusion=[RECENT_FUSION, 'latest'],
                 fusion=[FUSION, 'pooled'], balanced=['baseline_balanced', 'pooled'],
                 unweighted=['baseline_lgbm', 'pooled'], same_strategy=[name, 'pooled'])
    if name in ORIGINAL_POLICIES:  # Identity precedence includes the canonical blend.
        return roles
    member, form = s['members'][0], s['form']
    roles.update(original_member=[member, policy], transformed_member=['logit_'+member, policy])
    for role, factor in [('matched_representation', 'representation'),
                         ('matched_weight', 'class_weight'), ('matched_C', 'C')]:
        roles[role] = [strategy_id(matched(member, factor), form), policy]
    if s['kind'] == 'blend':
        for a in ALPHAS:
            if a != s['alpha']:
                roles['other_alpha_'+str(int(100*a))] = [strategy_id(member, a), policy]
    return roles


def contrast_factors():
    rows = []
    for s, k in product(strategies(), POLICIES):
        n = s['id']
        rows += [('original operating reference', n, k, r, rk)
                 for r, rk in ORIGINAL_POLICIES.items()]
        if k != 'pooled':
            rows.append(('policy minus own pooled', n, k, n, 'pooled'))
    for n, a, k in product(MEMBERS, ALPHAS, POLICIES):
        for label, ref in [('blend minus native', n), ('blend minus transformed', 'logit_'+n),
                           ('blend minus tree', FUSION)]:
            rows.append((label, strategy_id(n, a), k, ref, k))
    for factor, label, predicate in [
        ('representation', 'interaction minus additive', lambda s: s['representation'] == 'interaction'),
        ('class_weight', 'balanced minus unweighted', lambda s: s['class_weight'] == 'balanced'),
        ('C', 'larger C minus smaller C', lambda s: s['C'] == 1.)]:
        for s, form, k in product([s for s in recipes() if predicate(s)], FORMS, POLICIES):
            rows.append((label, strategy_id(s['id'], form), k,
                         strategy_id(matched(s['id'], factor), form), k))
    for n, (lo, hi), k in product(MEMBERS, list(combinations(ALPHAS, 2)), POLICIES):
        rows.append(('larger alpha minus smaller alpha', strategy_id(n, hi), k, strategy_id(n, lo), k))
    rows += [('transformed minus native', 'logit_'+n, k, n, k) for n, k in product(MEMBERS, POLICIES)]
    if len(rows) != 1386 or len(set(rows)) != 1386:
        raise ValueError('Incomplete contrast union')
    return rows


def diversity(a, b, pa, pb):
    aligned(a, b)
    x, z = a['scores'], b['scores']
    pa, pb = np.asarray(pa), np.asarray(pb)
    if pa.dtype != bool or pb.dtype != bool or pa.shape != x.shape or pb.shape != x.shape:
        raise ValueError('Invalid diversity decisions')
    constant = bool(np.ptp(x) == 0 or np.ptp(z) == 0)
    out = dict(pearson=None if constant else float(pearsonr(x, z).statistic),
               spearman=None if constant else float(spearmanr(x, z).statistic),
               correlation_reason='constant member' if constant else 'defined',
               both_flagged=int((pa & pb).sum()), member_only=int((pa & ~pb).sum()),
               tree_only=int((~pa & pb).sum()), neither_flagged=int((~pa & ~pb).sum()))
    for value, label in [(1, 'fraud'), (0, 'legitimate')]:
        mask = a['labels'] == value
        out.update({label+'_'+key: int((mask & v).sum()) for key, v in
                    [('both', pa & pb), ('member_only', pa & ~pb), ('tree_only', ~pa & pb),
                     ('neither', ~pa & ~pb)]})
    return out
