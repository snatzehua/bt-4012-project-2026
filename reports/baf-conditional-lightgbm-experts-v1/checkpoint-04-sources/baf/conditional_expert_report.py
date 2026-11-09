"""Independent prospective role and class-cost checks for conditional studies."""
from itertools import product
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from . import conditional_expert_study as run
from .common import read_config, digest
from .improvements import prediction, paths
from .spline_fusion_report import independent_rule, independent_mask, independent_metrics

POLICIES = ['pooled', 'latest', 'rank_fraction']
PARTITIONS = {'payment_type': ['AA','AB','AC','AD','AE'],
              'housing_status': ['BA','BB','BC','BD','BE','BF','BG']}


def direct_class_costs(labels, weighted):
    y = np.asarray(labels)
    counts = [int(np.count_nonzero(y == value)) for value in [0, 1]]
    if not all(counts) or sum(counts) != len(y):
        raise ValueError('Both binary training classes required')
    return {'0': len(y) / (2 * counts[0]), '1': len(y) / (2 * counts[1])} if weighted else {'0': 1., '1': 1.}


def independent_strategy_ids():
    refs = ['bagged_anchor_balanced_logit_a25', 'interaction_blend_spline_c1p0_balanced_a10',
            'margin_blend_spline_c0p1_unweighted_a25', 'logit_mean_xgb_d4_w1_baseline_balanced_a50',
            'baseline_balanced', 'baseline_lgbm']
    return refs + ['capacity400_' + w for w in ['balanced', 'unweighted']] + [
        'expert_' + p + '_' + w if a == 100 else f'expert_blend_{p}_{w}_a{a}'
        for p, w, a in product(['payment_type', 'housing_status'], ['balanced', 'unweighted'], [25, 50, 75, 100])]


def keys(table, columns, expected):
    actual = list(table[columns].itertuples(index=False, name=None))
    expected = set(expected)
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Independent table key grid differs: ' + ','.join(columns))


def close(row, expected):
    for key, value in expected.items():
        if isinstance(value, (str, bool)):
            valid = row[key] == value
        elif value is None:
            valid = row[key] is None or pd.isna(row[key])
        else:
            valid = np.isclose(row[key], value, atol=1e-12, rtol=0, equal_nan=True)
        if not valid:
            raise ValueError('Independent table value differs: ' + key)


def buckets(frame, ids, partition):
    # Reconstruct each bucket directly; do not use the study routing helper.
    result = []
    for value in frame.loc[ids, partition].to_numpy():
        result.append('__missing__' if pd.isna(value) else value if value in PARTITIONS[partition] else '__unseen__')
    return np.asarray(result, dtype=object)


def direct_group_metrics(labels, mask):
    n = len(labels)
    tn, fp, fn, tp = ([int(x) for x in confusion_matrix(labels, mask, labels=[0,1]).ravel()]
                       if n else [0, 0, 0, 0])
    def divide(a, b): return a / b if b else 0.
    recall = divide(tp, tp+fn); specificity = divide(tn, tn+fp)
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(n=n, fraud=int(np.sum(labels)), tn=tn, fp=fp, fn=fn, tp=tp,
        precision=divide(tp,tp+fp), recall=recall, f1=divide(2*tp,2*tp+fp+fn),
        f2=divide(5*tp,5*tp+fp+4*fn), accuracy=divide(tp+tn,n),
        fpr=divide(fp,fp+tn), balanced_accuracy=(recall+specificity)/2,
        mcc=divide(tp*tn-fp*fn,denominator),
        flagged=tp+fp, flagged_fraction=divide(tp+fp,n))


def audit_tables(c, frame, h=None, generated=None):
    h = run.history(c, physical=True) if h is None else h
    generated = ({n: pd.read_csv(Path(c['output_dir'])/(n+'.csv')) for n in [
        'crossfit','summary','pooled','contrasts','eligibility','routing','stratum_errors',
        'physical_score_diagnostics','score_diagnostics']} if generated is None else generated)
    names = independent_strategy_ids()
    if set(h) != set(run.physical_names()) or len(h) != 31 or any(set(v) != {4,5,6} for v in h.values()):
        raise ValueError('Independent physical score grid differs')
    for n, m in product(h, [4,5,6]):
        d = h[n][m]; ids = frame.index[frame.month == m].to_numpy()
        if not np.array_equal(d['row_ids'],ids) or not np.array_equal(d['labels'],frame.loc[ids,'fraud_bool'].to_numpy()):
            raise ValueError('Independent physical population differs')
        if not np.isfinite(d['scores']).all():
            raise ValueError('Nonfinite physical scores')
    cf, summary, pooled = [generated[n] for n in ['crossfit','summary','pooled']]
    keys(cf,['candidate','policy','eval_month'],product(names,POLICIES,[5,6]))
    keys(summary,['candidate','policy'],product(names,POLICIES))
    keys(pooled,['candidate','policy'],product(names,POLICIES))
    masks = {}; independent = []
    for row in cf.to_dict('records'):
        n,k,m = row['candidate'],row['policy'],row['eval_month']; d=h[n][m]
        rule=independent_rule(h[n],k,m); mask=independent_mask(d,rule); masks[n,k,m]=mask
        if json.loads(row['parameters']) != rule or str(row['source_months']) != ','.join(map(str,rule['source_months'])):
            raise ValueError('Independent policy chronology or rule differs')
        direct=independent_metrics(d,mask); close(row,direct)
        independent.append(dict(candidate=n,policy=k,eval_month=m,**direct))
    direct_cf = pd.DataFrame(independent)
    metrics=['f2','f1','f3','precision','recall','flagged_fraction','average_precision','tp','fp','fn']
    direct_summary=direct_cf.groupby(['candidate','policy'],as_index=False)[metrics].mean().sort_values(
        ['f2','recall','precision','candidate','policy'],ascending=[False,False,False,True,True]).reset_index(drop=True)
    if list(summary[['candidate','policy']].itertuples(index=False,name=None)) != list(direct_summary[['candidate','policy']].itertuples(index=False,name=None)):
        raise ValueError('Actual F2 selection order differs')
    for i, row in summary.iterrows(): close(row,direct_summary.iloc[i][metrics].to_dict())
    for row in pooled.to_dict('records'):
        n,k=row['candidate'],row['policy']; labels=np.concatenate([h[n][m]['labels'] for m in [5,6]])
        mask=np.concatenate([masks[n,k,m] for m in [5,6]])
        close(row,{key:v for key,v in direct_group_metrics(labels,mask).items() if key not in ['n','fraud']})
    factors = read_config(run.PLANNING)['contrasts']
    expected_factors = [(t,n,k,b,bk,json.dumps(extra,separators=(',',':'))) for t,n,k,b,bk,extra in factors]
    contrast=generated['contrasts']
    keys(contrast,['contrast','candidate','policy','reference','reference_policy','factor_details'],expected_factors)
    index=direct_summary.set_index(['candidate','policy'])
    for row in contrast.to_dict('records'):
        a,b=index.loc[(row['candidate'],row['policy'])],index.loc[(row['reference'],row['reference_policy'])]
        close(row,{v+'_difference':float(a[v]-b[v]) for v in ['f2','recall','precision','flagged_fraction','average_precision']})
    groups=[(m,p,w,level) for m,p,w in product([4,5,6],PARTITIONS,['balanced','unweighted']) for level in PARTITIONS[p]]
    keys(generated['eligibility'],['eval_month','partition','weighting','level'],groups)
    eligible={}
    for row in generated['eligibility'].to_dict('records'):
        m,p,w,level=[row[k] for k in ['eval_month','partition','weighting','level']]
        y=frame.loc[(frame.month<m)&frame[p].eq(level),'fraud_bool'];n=len(y);fraud=int(y.sum())
        valid=n>=10000 and fraud>=100 and n-fraud>=100;eligible[m,p,w,level]=valid
        close(row,dict(train_rows=n,train_fraud=fraud,train_legitimate=n-fraud,eligible=valid,fallback=not valid))
    bucket_cache={(m,p):buckets(frame,frame.index[frame.month==m].to_numpy(),p) for m,p in product([4,5,6],PARTITIONS)}
    routekeys=[(m,p,w,level) for m,p,w in product([4,5,6],PARTITIONS,['balanced','unweighted']) for level in PARTITIONS[p]+['__missing__','__unseen__']]
    keys(generated['routing'],['eval_month','partition','weighting','level'],routekeys)
    for row in generated['routing'].to_dict('records'):
        m,p,w,level=[row[k] for k in ['eval_month','partition','weighting','level']]
        count=int(np.sum(bucket_cache[m,p]==level))
        route='missing' if level=='__missing__' else 'unseen' if level=='__unseen__' else 'expert' if eligible[m,p,w,level] else 'ineligible'
        close(row,dict(rows=count,route=route))
    error_keys=[(n,k,m,p,level) for n,k,m,p in product(names,POLICIES,[5,6],PARTITIONS) for level in PARTITIONS[p]+['__missing__','__unseen__']]
    keys(generated['stratum_errors'],['candidate','policy','eval_month','partition','level'],error_keys)
    for row in generated['stratum_errors'].to_dict('records'):
        n,k,m,p,level=[row[key] for key in ['candidate','policy','eval_month','partition','level']]
        d=h[n][m];take=bucket_cache[m,p]==level
        close(row,direct_group_metrics(d['labels'][take],masks[n,k,m][take]))
    for table_name, source_names in [('physical_score_diagnostics',list(h)),('score_diagnostics',names)]:
        table=generated[table_name];keys(table,['candidate','eval_month'],product(source_names,[4,5,6]))
        for row in table.to_dict('records'):
            v=h[row['candidate']][row['eval_month']]['scores']
            close(row,dict(n=len(v),score_min=float(v.min()),score_max=float(v.max()),score_mean=float(v.mean()),
                           unique_scores=len(set(v)),tied_fraction=1-len(set(v))/len(v)))
    counts={'crossfit':144,'summary':72,'pooled':72,'contrasts':738,'eligibility':72,'routing':96,
            'stratum_errors':2304,'physical_score_diagnostics':93,'score_diagnostics':72}
    if any(len(generated[n]) != size for n,size in counts.items()):
        raise ValueError('Independent table cardinality differs')
    return dict(counts=counts,winner=str(summary.iloc[0].candidate),policy=str(summary.iloc[0].policy),
                development_f2=float(summary.iloc[0].f2),actual_source_models_audited=False)


def audit_new_compositions(c, fp, frame, months, phase='validation', names=None):
    names = independent_strategy_ids() if names is None else names
    needed = set()
    for name in names:
        if name not in independent_strategy_ids():
            raise ValueError('Independent composition name undeclared')
        if name.startswith('expert_'):
            needed.add(name)
            match = re.fullmatch(r'expert_blend_(payment_type|housing_status)_(balanced|unweighted)_a(25|50|75)',name)
            if match:
                p,w,a=match.groups(); needed.add(f'expert_{p}_{w}')
    checks = 0; subsets = set()
    for m in months:
        ids=frame.index[frame.month==m].to_numpy(); labels=frame.loc[ids,'fraud_bool'].to_numpy()
        for name in sorted(needed, key=lambda n:(n.startswith('expert_blend_'),n)):
            match=re.fullmatch(r'expert_(payment_type|housing_status)_(balanced|unweighted)',name)
            route=[]
            if match:
                p,w=match.groups();a=100
                spec=dict(id=name,kind='expert',partition=p,weighting=w,alpha_numerator=100,alpha_denominator=100)
            else:
                match=re.fullmatch(r'expert_blend_(payment_type|housing_status)_(balanced|unweighted)_a(25|50|75)',name)
                if match is None:raise ValueError('Independent composition identity invalid')
                p,w,a=match.groups();a=int(a)
                spec=dict(id=name,kind='blend',partition=p,weighting=w,alpha_numerator=a,alpha_denominator=100)
            member='baseline_balanced' if w=='balanced' else 'baseline_lgbm'
            global_data=prediction(c,phase,m,member)
            if not np.array_equal(global_data['row_ids'],ids) or not np.array_equal(global_data['labels'],labels):
                raise ValueError('Independent global population differs')
            values=global_data['scores'].astype(np.float64).copy()
            if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
                raise ValueError('Independent global probability invalid')
            members=[member]
            if a==100:
                training=frame.loc[frame.month<m]; eligible=[]
                for level in PARTITIONS[p]:
                    y=training.loc[training[p].eq(level),'fraud_bool'];n=len(y);fraud=int(y.sum())
                    if n>=10000 and fraud>=100 and n-fraud>=100:eligible.append(level)
                for level in eligible:
                    local_name=f'expert_model_{p}_{w}_{level}';members.append(local_name);subsets.add((m,local_name))
                    item=prediction(c,phase,m,local_name)
                    take=frame.loc[ids,p].eq(level).fillna(False).to_numpy(dtype=bool)
                    if not np.array_equal(item['row_ids'],ids[take]) or not np.array_equal(item['labels'],labels[take]):
                        raise ValueError('Independent expert subset membership differs')
                    local=item['scores']
                    if local.shape!=(int(take.sum()),) or not np.isfinite(local).all() or ((local<0)|(local>1)).any():
                        raise ValueError('Independent expert subset probability invalid')
                    values[take]=local
                b=buckets(frame,ids,p)
                for level in PARTITIONS[p]+['__missing__','__unseen__']:
                    routing='missing' if level=='__missing__' else 'unseen' if level=='__unseen__' else 'expert' if level in eligible else 'ineligible'
                    route.append(dict(eval_month=m,partition=p,level=level,rows=int(np.sum(b==level)),route=routing))
                arithmetic='one eligible member per original partition value, else global200'
            else:
                local_name=f'expert_{p}_{w}';members.append(local_name)
                item=prediction(c,phase,m,local_name)
                if not np.array_equal(item['row_ids'],ids) or not np.array_equal(item['labels'],labels):
                    raise ValueError('Independent blend population differs')
                values=(1-a/100)*values+(a/100)*item['scores']
                arithmetic='(1-alpha)*global200+alpha*stitched_expert in float64'
            rp,pp=paths(c,phase,m,name);record=read_config(rp);saved=prediction(c,phase,m,name)
            deps={n:{str(v):digest(v) for v in paths(c,phase,m,n)} for n in members}
            expected=dict(candidate=name,phase=phase,eval_month=m,spec=spec,fingerprint=fp,
                          dependencies=deps,arithmetic=arithmetic,routing=route)
            if any(record.get(k)!=v for k,v in expected.items()) or record['prediction_sha256']!=digest(pp):
                raise ValueError('Independent composition provenance differs')
            if not np.array_equal(saved['row_ids'],ids) or not np.array_equal(saved['labels'],labels) or not np.array_equal(saved['scores'],values):
                raise ValueError('Independent stitching or blend arithmetic differs')
            checks+=1
    return dict(new_full_compositions=checks,member_subsets=len(subsets),actual_source_models_audited=False)
