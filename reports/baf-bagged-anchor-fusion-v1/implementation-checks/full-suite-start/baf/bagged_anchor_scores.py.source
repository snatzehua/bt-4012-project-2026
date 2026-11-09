"""Declared fixed additions to the preserved anchor; no fitted composition."""
from itertools import combinations, product

import numpy as np

from . import bagged_boosting_scores as parent
from .recall_weighted_metrics import POLICIES, aligned, check_data
from .score_fusion import log_odds

SEEDS = [17, 29, 43]
WEIGHTINGS = ['balanced', 'unweighted']
FORMS = ['prob', 'logit']
ALPHAS = [.1, .25, .5]
EPSILON = 1e-6
REFERENCE_ROLES = parent.REFERENCE_ROLES
REFERENCE_POLICIES = parent.REFERENCE_POLICIES
REFERENCE_BASES = parent.REFERENCE_BASES
ANCHOR = REFERENCE_ROLES['latest_fusion'][0]
TREE = REFERENCE_ROLES['fusion'][0]
recipes = parent.recipes


def probability_mean(w):
    if w not in WEIGHTINGS:
        raise ValueError('Undeclared weighting')
    return f'bagged_{w}_prob_mean'


def margin(w, form):
    if w not in WEIGHTINGS or form not in FORMS:
        raise ValueError('Undeclared weighting or form')
    return ('logit_' + probability_mean(w) if form == 'prob'
            else f'bagged_{w}_logit_mean')


def blend_id(w, form, alpha):
    if w not in WEIGHTINGS or form not in FORMS or isinstance(alpha, bool) or alpha not in ALPHAS:
        raise ValueError('Undeclared blend')
    return f'bagged_anchor_{w}_{form}_a{round(100*alpha):02d}'


def strategies():
    rows = parent.strategies()
    sources = [s['id'] for s in recipes()] + [probability_mean(w) for w in WEIGHTINGS]
    sources += [parent.unbagged(w) for w in WEIGHTINGS]
    rows += [dict(id='logit_' + n, kind='logit', members=[n]) for n in sources]
    rows += [dict(id=blend_id(w, f, a), kind='blend', weighting=w, form=f,
                  alpha=a, members=[ANCHOR, margin(w, f)])
             for w, f, a in product(WEIGHTINGS, FORMS, ALPHAS)]
    if len(rows) != 37 or len({s['id'] for s in rows}) != 37:
        raise ValueError('Finite strategy union changed')
    return rows


def lookup(name):
    spec = next((s for s in strategies() if s['id'] == name), None)
    if spec is None:
        raise ValueError('Undeclared strategy')
    return spec


def compose(spec, items):
    """Evaluate the declared association, retaining a finite unbounded anchor."""
    if spec not in strategies() or spec['kind'] not in ['logit', 'blend', 'aggregate']:
        raise ValueError('Undeclared derived recipe')
    if set(items) != set(spec['members']) or len(items) != len(spec['members']):
        raise ValueError('Exactly the declared unique members required')
    ds = [items[n] for n in spec['members']]
    for d in ds:
        aligned(ds[0], d)
    if spec['kind'] == 'aggregate':
        return parent.compose(spec, items)
    if spec['kind'] == 'logit':
        values = log_odds(ds[0]['scores'], EPSILON)
    else:
        h, g = [np.asarray(d['scores'], dtype=np.float64) for d in ds]
        values = (1-spec['alpha'])*h + spec['alpha']*g
    result = dict(row_ids=ds[0]['row_ids'], labels=ds[0]['labels'], scores=values)
    check_data(result)
    return result


def matched_weight(name):
    lookup(name)
    if name in REFERENCE_POLICIES:
        raise ValueError('Original reference has precedence')
    if name == 'logit_baseline_balanced':
        return 'logit_baseline_lgbm'
    if name == 'logit_baseline_lgbm':
        return 'logit_baseline_balanced'
    w = weighting(name)
    other = 'unweighted' if w == 'balanced' else 'balanced'
    return name.replace('_'+w+'_', '_'+other+'_', 1)


def weighting(name):
    if name in ['logit_baseline_balanced', 'baseline_balanced']:
        return 'balanced'
    if name in ['logit_baseline_lgbm', 'baseline_lgbm']:
        return 'unweighted'
    if '_balanced_' in name:
        return 'balanced'
    if '_unweighted_' in name:
        return 'unweighted'
    raise ValueError('No declared weighting')


def winner_kind(name):
    s = lookup(name)
    if name in REFERENCE_POLICIES:
        return 'reference'
    if name in ['logit_baseline_balanced', 'logit_baseline_lgbm']:
        return 'baseline_logit'
    if s['kind'] == 'blend':
        return 'blend'
    native = s['members'][0] if s['kind'] == 'logit' else name
    return 'single' if native in {s['id'] for s in recipes()} else 'aggregate'


def choices_for(name, policy):
    s = lookup(name)
    if policy not in POLICIES:
        raise ValueError('Undeclared policy')
    result = dict(selected=[name, policy], **{a:list(v) for a,v in REFERENCE_ROLES.items()},
                  same_strategy=[name, 'pooled'])
    kind = winner_kind(name)
    if kind == 'reference':
        return result
    if kind == 'baseline_logit':
        result.update(native_member=[s['members'][0], policy],
                      matched_weight=[matched_weight(name), policy])
        return result
    w = weighting(name)
    for seed in SEEDS:
        result[f'member_s{seed}'] = [f'bagged_{w}_s{seed}', policy]
        result[f'member_logit_s{seed}'] = [f'logit_bagged_{w}_s{seed}', policy]
    result.update(probability_mean=[probability_mean(w), policy],
                  probability_mean_logit=[margin(w, 'prob'), policy],
                  logit_mean=[margin(w, 'logit'), policy],
                  unbagged_member=[parent.unbagged(w), policy],
                  unbagged_logit=['logit_'+parent.unbagged(w), policy])
    if kind == 'blend':
        f, a = s['form'], s['alpha']
        result.update(anchor_at_policy=[ANCHOR, policy],
                      added_member_native=[probability_mean(w) if f=='prob' else margin(w,f), policy],
                      added_member_margin=[margin(w,f), policy],
                      matched_weight=[matched_weight(name), policy],
                      matched_form=[blend_id(w, 'logit' if f=='prob' else 'prob', a), policy])
        for a in ALPHAS:
            result[f'alpha_a{round(100*a):02d}'] = [blend_id(w,f,a), policy]
    else:
        native = s['members'][0] if s['kind']=='logit' else name
        transformed = native if native==margin(w,'logit') else 'logit_'+native
        result.update(native_member=[native,policy], transformed_member=[transformed,policy],
                      matched_weight=[matched_weight(name),policy])
    return result


def dependencies(names):
    found, bases, active = set(), set(), set()
    inherited = {s['id'] for s in parent.strategies()}
    def visit(n):
        if n in found:
            return
        if n in active:
            raise ValueError('Cyclic dependency')
        active.add(n)
        if n in inherited:
            d = parent.dependencies([n]);found.update(d['scores']);bases.update(d['bases'])
        else:
            for member in lookup(n)['members']:
                visit(member)
            found.add(n)
        active.remove(n)
    for name in names:
        visit(name)
    return dict(bases=sorted(bases), scores=sorted(found), derived=sorted(found-bases),
                bagged_bases=sorted(bases & {s['id'] for s in recipes()}))


def expression_rows():
    """Diagnostic atoms only: do not use this flattened expansion for scoring."""
    rows = []
    for s in strategies():
        if s['kind'] != 'blend':
            continue
        a, w, f = s['alpha'], s['weighting'], s['form']
        atoms = [('logit(spline_c1p0_balanced)', (1-a)*.1),
                 ('logit(xgb_d4_w1)', (1-a)*.9*.5),
                 ('logit(baseline_balanced)', (1-a)*.9*.5)]
        atoms += ([(f'logit(bagged_{w}_s{seed})', a/3) for seed in SEEDS] if f=='logit'
                  else [(f'logit({probability_mean(w)})', a)])
        rows += [dict(candidate=s['id'], atom=atom, coefficient=value,
                      atom_kind='nonlinear probability mean' if 'prob_mean)' in atom else 'source logit',
                      evaluation_arithmetic='(1-alpha)*anchor + alpha*margin')
                 for atom,value in atoms]
    return rows


def diagnostic_pairs():
    return ([('anchor versus '+w+' '+f, ANCHOR, margin(w,f)) for w,f in product(WEIGHTINGS,FORMS)]
            + [('aggregation forms '+w, margin(w,'prob'), margin(w,'logit')) for w in WEIGHTINGS])


def curve_controls(w,form):
    """Canonical endpoint IDs, never duplicate evaluated zero/unit strategies."""
    added=margin(w,form)
    return [(0.,ANCHOR)]+[(a,blend_id(w,form,a)) for a in ALPHAS]+[(1.,added)]


def contrast_factors():
    rows = []
    def add(label,n,k,r,rk=None):
        rows.append((label,n,k,r,k if rk is None else rk))
    names = [s['id'] for s in strategies()]
    blends = [s for s in strategies() if s['kind']=='blend']
    logits = [s for s in strategies() if s['kind']=='logit']
    for n,k in product(names,POLICIES):
        for role,(ref,rk) in REFERENCE_ROLES.items():
            add('fixed reference '+role,n,k,ref,rk)
        if k!='pooled':
            add('policy minus own pooled',n,k,n,'pooled')
    for s,k in product(blends,POLICIES):
        add('blend minus anchor',s['id'],k,ANCHOR)
        add('blend minus added margin',s['id'],k,s['members'][1])
    for w,f,(a,b),k in product(WEIGHTINGS,FORMS,list(combinations(ALPHAS,2)),POLICIES):
        add('higher minus lower blend weight',blend_id(w,f,b),k,blend_id(w,f,a))
    for f,a,k in product(FORMS,ALPHAS,POLICIES):
        add('blend balanced minus unweighted',blend_id('balanced',f,a),k,blend_id('unweighted',f,a))
    for w,a,k in product(WEIGHTINGS,ALPHAS,POLICIES):
        add('blend mean-logit minus logit-mean-probability',blend_id(w,'logit',a),k,blend_id(w,'prob',a))
    for s,k in product(logits,POLICIES):
        add('transformed minus native',s['id'],k,s['members'][0])
    for w,k in product(WEIGHTINGS,POLICIES):
        add('added mean-logit minus logit-mean-probability',margin(w,'logit'),k,margin(w,'prob'))
    for form,k in product(['s17','s29','s43','prob_mean','logit_mean'],POLICIES):
        add('native balanced minus unweighted',f'bagged_balanced_{form}',k,f'bagged_unweighted_{form}')
    for s,k in product([s for s in logits if weighting(s['id'])=='balanced'],POLICIES):
        add('logit balanced minus unweighted',s['id'],k,matched_weight(s['id']))
    for w,k,seed in product(WEIGHTINGS,POLICIES,SEEDS):
        add('probability aggregate minus native member',probability_mean(w),k,f'bagged_{w}_s{seed}')
        add('mean-logit aggregate minus logit member',margin(w,'logit'),k,f'logit_bagged_{w}_s{seed}')
        add('logit-probability aggregate minus logit member',margin(w,'prob'),k,f'logit_bagged_{w}_s{seed}')
    controls = [s for s in strategies() if s['kind'] in ['bagged','aggregate']
                or (s['kind']=='logit' and s['members'][0].startswith('bagged_'))]
    for s,k in product(controls,POLICIES):
        w=weighting(s['id']);is_margin=s['kind']=='logit' or s.get('form')=='logit'
        add('randomized minus matched unbagged',s['id'],k,('logit_' if is_margin else '')+parent.unbagged(w))
    if len(rows)!=947 or len(set(rows))!=947:
        raise ValueError('Complete factor union differs')
    return rows


def declaration_counts():
    return dict(strategies=37, physical_development_scores=120, selectable_scores=111,
                forward_rows=222, macro_rows=111, pooled_rows=111, score_rows=111,
                contrasts=947, source_bases=33, source_derived=21, new_derived=66,
                diversity_rows=18, overlap_rows=36, clipping_rows=39, expression_leaf_rows=60,
                winner_policy_cases=111, development_draws=1000, consumed_draws=500,
                new_production_miniature_fits=0, cache_test_fits_per_full_invocation=6)
