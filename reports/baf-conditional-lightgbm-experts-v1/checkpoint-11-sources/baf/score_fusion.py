"""Fixed row-wise score fusion; no labels, learned weights or batch ranks."""
from __future__ import annotations

import numpy as np
from .recall_weighted_metrics import aligned, check_data

EPSILON = 1e-6
ANCHORS = ['baseline_balanced', 'baseline_lgbm']


def checked(scores):
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError('Invalid base score vector')
    return x


def log_odds(scores, epsilon=EPSILON):
    if not np.isfinite(epsilon) or not 0 < epsilon < .5:
        raise ValueError('Invalid logit clipping epsilon')
    x = np.clip(checked(scores), epsilon, 1-epsilon)
    return np.log(x) - np.log1p(-x)


def fuse(a, b, alpha, mode, epsilon=EPSILON):
    a, b = checked(a), checked(b)
    if a.shape != b.shape or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('Invalid fusion members or weight')
    if mode == 'logit_mean':
        a, b = log_odds(a, epsilon), log_odds(b, epsilon)
    elif mode != 'raw_mean':
        raise ValueError('Unknown fusion mode')
    result = alpha*a + (1-alpha)*b
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite composed score')
    return result


def strategies(new_ids, reference_ids):
    singles = list(new_ids)+list(reference_ids)
    if len(set(singles)) != 7 or list(reference_ids) != ANCHORS+['native31']:
        raise ValueError('Undeclared base grid')
    return ([dict(id=n, kind='raw', members=[n]) for n in singles]
            + [dict(id='logit_'+n, kind='logit', members=[n]) for n in list(new_ids)+ANCHORS]
            + [dict(id=f'{mode}_{n}_{anchor}_a{weight}', kind=mode,
                    members=[n, anchor], alpha=weight/100)
               for mode in ['raw_mean', 'logit_mean'] for n in new_ids
               for anchor in ANCHORS for weight in [25, 50, 75]])


def compose(spec, members, epsilon=EPSILON):
    kind = spec['kind']; names = spec['members']
    if kind not in ['raw', 'logit', 'raw_mean', 'logit_mean'] or len(names) != (2 if kind.endswith('_mean') else 1):
        raise ValueError('Invalid score strategy')
    if len(set(names)) != len(names) or any(n not in members for n in names):
        raise ValueError('Unknown or duplicate member')
    items = [members[n] for n in names]
    for d in items:
        aligned(d, items[0]); checked(d['scores'])
    if kind == 'raw':
        scores = checked(items[0]['scores'])
    elif kind == 'logit':
        scores = log_odds(items[0]['scores'], epsilon)
    else:
        scores = fuse(items[0]['scores'], items[1]['scores'], spec['alpha'], kind, epsilon)
    result = dict(row_ids=items[0]['row_ids'], labels=items[0]['labels'], scores=scores)
    check_data(result)
    return result


def clipping(scores, epsilon=EPSILON):
    x = checked(scores)
    log_odds(x, epsilon)  # validate the declared clipping setting
    return dict(n=len(x), epsilon=epsilon, input_min=float(x.min()), input_max=float(x.max()),
                below_epsilon=int((x < epsilon).sum()), above_one_minus_epsilon=int((x > 1-epsilon).sum()),
                changed_count=int((x != np.clip(x, epsilon, 1-epsilon)).sum()))


def choices_for(spec, policy):
    name, kind = spec['id'], spec['kind']
    choices = dict(selected=[name, policy], balanced=['baseline_balanced', 'pooled'],
                   unweighted=['baseline_lgbm', 'pooled'], same_model=[name, 'pooled'])
    if kind in ['raw_mean', 'logit_mean']:
        xgb, anchor = spec['members']; prefix = 'logit_' if kind == 'logit_mean' else ''
        choices.update(xgb_member=[prefix+xgb, policy], anchor_member=[prefix+anchor, policy])
        if kind == 'logit_mean':
            choices.update(raw_xgb=[xgb, policy], raw_anchor=[anchor, policy])
    elif kind == 'logit':
        choices['raw_member'] = [spec['members'][0], policy]
    elif kind == 'raw' and name != 'native31':
        choices['logit_member'] = ['logit_'+name, policy]
    elif kind != 'raw':
        raise ValueError('Unknown winner type')
    return choices
