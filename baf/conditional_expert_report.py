"""Independent prospective role and class-cost checks for conditional studies."""
from itertools import product
import json
import math
from pathlib import Path
import re
import argparse
import resource
import shutil
import tempfile
import time

import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from . import conditional_expert_study as run
from .common import read_config, digest, write_json, load_data, features
from .improvements import prediction, paths
from .spline_fusion_report import independent_rule, independent_mask, independent_metrics
from .recall_weighted_metrics import aligned

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
    factors = read_config(run.planning_evidence_path())['contrasts']
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


def audit_actual_models(c, fp, frame, base, phase='validation', names=None):
    """Audit every saved model plus independently reconstructed raw memberships/costs."""
    records, ds, work = run.prepare(c,fp,frame,base,phase,names,allow_missing=False)
    current=[r for r in records if r['candidate'].startswith(('capacity400_','expert_model_'))]
    for r in current:
        spec=r['spec'];m=r['eval_month'];whole=frame.index[frame.month<m]
        local=whole if spec['kind']=='capacity' else whole[frame.loc[whole,spec['partition']].eq(spec['level'])]
        ev=frame.index[frame.month==m]
        if spec['kind']=='local':ev=ev[frame.loc[ev,spec['partition']].eq(spec['level'])]
        costs=direct_class_costs(frame.loc[whole,'fraud_bool'],spec['weighting']=='balanced')
        weights=np.asarray([costs[str(int(y))] for y in frame.loc[local,'fraud_bool']],dtype='<f8')
        import hashlib
        expected=dict(global_train_rows=len(whole),global_train_fraud=int(frame.loc[whole,'fraud_bool'].sum()),
            local_train_rows=len(local),local_train_fraud=int(frame.loc[local,'fraud_bool'].sum()),
            local_weight_sha256=hashlib.sha256(weights.tobytes()).hexdigest())
        close(r,expected)
        if r['class_costs']!=costs or r['subset']!=(spec['kind']=='local'):
            raise ValueError('Independent whole-training costs/subset identity differs')
        model_path=run.scores.model_paths(c,phase,m,spec['id'])[2]
        pipe=joblib.load(model_path)['model']
        if pipe.named_steps['model'].class_weight is not None:
            raise ValueError('Independent duplicate class weighting')
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=c['threads']):
            actual=pipe.predict_proba(base.loc[ev])[:,1] if len(ev) else np.empty(0)
        saved=prediction(c,phase,m,spec['id'])
        if not np.array_equal(saved['row_ids'],ev.to_numpy()) or not np.array_equal(saved['labels'],frame.loc[ev,'fraud_bool'].to_numpy()):
            raise ValueError('Independent saved member population differs')
        if not np.allclose(saved['scores'],actual,atol=1e-12,rtol=0):
            raise ValueError('Independent saved inference differs')
        if r['train_row_rounds']!=len(local)*pipe.named_steps['model'].booster_.current_iteration():
            raise ValueError('Independent measured tree work differs')
    from .bagged_anchor_report import audit_compositions
    old_names=[n for n,k in run.scores.REFERENCE_ROLES.values()]
    audit_compositions(c,fp,[4,5,6] if phase=='validation' else [7],phase,old_names)
    compositions=audit_new_compositions(c,fp,frame,[4,5,6] if phase=='validation' else [7],phase,names)
    run.exact_artifact_grid(c,frame,phase,names)
    return records,ds,work,compositions


def static_work(counts):
    return {k:v for k,v in counts.items() if not k.startswith('invocation_')}


def audit_measured_work(c,frame,records):
    out=Path(c['output_dir']);table=pd.read_csv(out/'conditional_model_work.csv').fillna('')
    new=[r for r in records if r['candidate'].startswith(('capacity400_','expert_model_'))]
    keys(table,['candidate','eval_month'],[(r['candidate'],r['eval_month']) for r in new])
    lookup={(r['candidate'],r['eval_month']):r for r in new}
    for row in table.to_dict('records'):
        r=lookup[row['candidate'],row['eval_month']];spec=r['spec']
        expected=dict(kind=spec['kind'],weighting=spec['weighting'],partition=spec.get('partition',''),level=spec.get('level',''),
            requested_rounds=spec['rounds'],fitted_tree_count=r['fitted_tree_count'],
            transformed_dimension=r['preprocessing']['transformed_dimension'])
        expected.update({k:r[k] for k in ['global_train_rows','global_train_fraud','local_train_rows','local_train_fraud',
            'train_row_rounds','fit_seconds','peak_process_rss_bytes','global_weight_sha256','local_weight_sha256']})
        close(row,expected)
        if any(json.loads(row[k])!=r[k] for k in ['warnings','requested_model_params','booster_params']):
            raise ValueError('Measured model recipe evidence differs')
    inference=pd.read_csv(out/'conditional_inference_work.csv')
    keys(inference,['eval_month','partition','weighting','level'],[
        (m,p,w,level) for m,p,w in product([4,5,6],PARTITIONS,['balanced','unweighted']) for level in PARTITIONS[p]+['__missing__','__unseen__']])
    specs={s['id']:s for s in run.required_sources(c)};global_rounds={}
    for m,w in product([4,5,6],['balanced','unweighted']):
        n='baseline_balanced' if w=='balanced' else 'baseline_lgbm'
        artifact=joblib.load(run.parent.source_paths(c,specs[n],m,'validation')[2])
        global_rounds[m,w]=int(artifact['model'].named_steps['model'].booster_.current_iteration())
    for row in inference.to_dict('records'):
        m,p,w,level=[row[k] for k in ['eval_month','partition','weighting','level']]
        future=frame.index[frame.month==m];n=int(np.count_nonzero(buckets(frame,future,p)==level))
        item=lookup.get((f'expert_model_{p}_{w}_{level}',m));local=0 if item is None else item['fitted_tree_count']
        global_count=global_rounds[m,w]
        close(row,dict(rows=n,global_fitted_rounds=global_count,local_fitted_rounds=local,
            computed_rounds_per_row=global_count+local,global_row_rounds=n*global_count,
            local_row_rounds=n*local,total_row_rounds=n*(global_count+local),shared_pure_and_blend_scores=True))
    return dict(model_rows=len(table),inference_rows=len(inference),shared_computation_row_rounds=int(inference.total_row_rounds.sum()))


def audit_development(config_path=run.CONFIG, *, write=True):
    c,fp=run.initialize(config_path);run.execution_gate(c,fp);out=Path(c['output_dir'])
    if write and (out/'selection.json').exists():
        raise ValueError('Frozen independent audit cannot be overwritten')
    done=read_config(out/'validation_complete.json')
    if done['fingerprint']!=fp:raise ValueError('Development fingerprint differs')
    run.verify_hashes({str(out/p):h for p,h in done['files'].items()})
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    records,ds,work,compositions=audit_actual_models(c,fp,frame,base)
    if len(records)!=84 or len(ds)!=63 or compositions['new_full_compositions']!=48 or compositions['member_subsets']!=54:
        raise ValueError('Complete actual development DAG required')
    actual=static_work(run.work_accounting(c,records,ds,work));prep=read_config(out/'preparation.json')
    if any(prep.get(k)!=v for k,v in actual.items()):
        raise ValueError('Actual preparation accounting differs')
    result=audit_tables(c,frame)
    measured_work=audit_measured_work(c,frame,records)
    index=pd.read_csv(out/'summary.csv').set_index(['candidate','policy'])
    # Original fixed values remain fixed; no new policy can replace them.
    parent_index=pd.read_csv(Path(read_config(c['parent_config'])['output_dir'])/'summary.csv').set_index(['candidate','policy'])
    for role,nk in run.scores.REFERENCE_ROLES.items():
        close(index.loc[tuple(nk)],parent_index.loc[tuple(nk)][['f2','precision','recall','flagged_fraction']].to_dict())
    run.guard(c)
    result.update(fingerprint=fp,validation_sha256=digest(out/'validation_complete.json'),
        saved_model_checks=len(records),actual_source_models_audited=True,
        independent_preprocessing_inference_costs_memberships=True,
        independent_policies_metrics_factors=True,elapsed_seconds=time.perf_counter()-start,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        created_at_utc=run.now(),compositions=compositions,measured_work=measured_work)
    if write:write_json(out/'independent_development_audit.json',result)
    return result


def audit_final(c,s,frame,base):
    out=Path(c['output_dir']);names={n for n,k in s['choices'].values()};sha=digest(out/'selection.json')
    records,ds,work,compositions=audit_actual_models(c,s['fingerprint'],frame,base,'exploratory',names)
    access=read_config(out/'exploratory_access.json');done=read_config(out/'transfer_complete.json')
    if access['selection_sha256']!=sha or done['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc'] or done['table_sha256']!=digest(out/'exploratory.csv'):
        raise ValueError('Transfer evidence differs')
    actual=static_work(run.work_accounting(c,records,ds,work))
    if any(done.get(k)!=v for k,v in actual.items()):raise ValueError('Actual transfer accounting differs')
    bounds=s['final_fit_bounds'];dag=s['score_dependencies']
    if len(records)!=bounds['unique_base_results'] or done['total_scores']!=len(dag['full_scores']) or done['subset_scores']!=bounds['subset_scores']:
        raise ValueError('Exact frozen transfer bounds differ')
    data=run.final_data(c,s);h=run.history(c);final=pd.read_csv(out/'exploratory.csv');masks={}
    keys(final,['role'],[(a,) for a in s['choices']])
    for a,d in data.items():
        aligned(d,data['selected']);n,k=s['choices'][a]
        if s['rules'][a]!=independent_rule(h[n],k,7):raise ValueError('Final earlier-only rule differs')
        masks[a]=independent_mask(d,s['rules'][a]);row=final[final.role==a].iloc[0]
        if row.candidate!=n or row.policy!=k or json.loads(row.parameters)!=s['rules'][a]:
            raise ValueError('Final role/rule differs')
        close(row,independent_metrics(d,masks[a]))
    return records,data,masks,final


def uncertainty(c,s,cf,h,data,masks):
    # This preserved routine is role-generic and independently reconstructs
    # every fixed-decision draw. No new score policy or bootstrap selection.
    from .bagged_anchor_report import uncertainty as paired
    result=paired(c,s,cf,h,data,masks)
    dint,ddraws,fint,fdraws=result
    roles=len(s['choices'])
    if roles not in [8,10,17] or (len(dint),len(fint))!=(3*(roles-1),roles-1):
        raise ValueError('Declared role-specific uncertainty grid differs')
    if c['bootstrap_replicates']!=500 or (len(ddraws),len(fdraws))!=(1000,500):
        raise ValueError('All 1500 paired draws required')
    return result


def curve_rows(summary):
    index=summary.set_index(['candidate','policy']);curves=[];capacity=[]
    for p,w,k in product(PARTITIONS,['balanced','unweighted'],POLICIES):
        for a in [0,25,50,75,100]:
            name=('baseline_balanced' if w=='balanced' else 'baseline_lgbm') if a==0 else (
                f'expert_{p}_{w}' if a==100 else f'expert_blend_{p}_{w}_a{a}')
            curves.append(dict(partition=p,weighting=w,policy=k,alpha=a/100,candidate=name,f2=float(index.loc[(name,k),'f2'])))
    for w,k in product(['balanced','unweighted'],POLICIES):
        name='capacity400_'+w
        capacity.append(dict(weighting=w,policy=k,candidate=name,f2=float(index.loc[(name,k),'f2'])))
    return pd.DataFrame(curves),pd.DataFrame(capacity)


def operating_figure(summary,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    keys(summary,['candidate','policy'],product(independent_strategy_ids(),POLICIES))
    curves,capacity=curve_rows(summary);fig,axes=plt.subplots(2,2,figsize=(14,10));legends={}
    for k,marker in zip(POLICIES,['o','^','s']):
        group=summary[summary.policy==k]
        artist=axes[0,0].scatter(group.recall,group.precision,label=k,marker=marker,s=22,alpha=.7)
        if not np.allclose(artist.get_offsets(),group[['recall','precision']].to_numpy(),atol=1e-12,rtol=0):
            raise ValueError('Rendered operating points differ')
    axes[0,0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 72 strategy / policy operating points')
    axes[0,0].legend();legends['operating']=axes[0,0].get_legend_handles_labels()[1]
    for ax,k in zip([axes[0,1],axes[1,0],axes[1,1]],POLICIES):
        for p,w in product(PARTITIONS,['balanced','unweighted']):
            g=curves[(curves.partition==p)&(curves.weighting==w)&(curves.policy==k)]
            line,=ax.plot(g.alpha,g.f2,marker='o',label=p+' / '+w,
                color='tab:blue' if w=='balanced' else 'tab:orange',linestyle='-' if p=='payment_type' else '--')
            if not np.array_equal(line.get_xdata(),g.alpha.to_numpy()) or not np.array_equal(line.get_ydata(),g.f2.to_numpy()):
                raise ValueError('Rendered alpha curve differs')
        for w in ['balanced','unweighted']:
            v=float(capacity[(capacity.weighting==w)&(capacity.policy==k)].iloc[0].f2)
            line=ax.axhline(v,color='tab:blue' if w=='balanced' else 'tab:orange',linestyle=':',label='global400 / '+w)
            if not np.array_equal(line.get_ydata(),[v,v]):raise ValueError('Rendered capacity line differs')
        ax.set(xlabel='Conditional weight (0 = global200; 1 = pure expert)',ylabel='Macro development F2',title=k+' policy')
        ax.set_xticks([0,.25,.5,.75,1]);ax.legend(fontsize=7)
        legends[k]=ax.get_legend_handles_labels()[1]
    fig.tight_layout();fig.savefig(dest/'conditional_experts_operating.png',dpi=160);plt.close(fig)
    summary[['candidate','policy','f2','precision','recall']].to_csv(dest/'plotted_operating_points.csv',index=False)
    curves.to_csv(dest/'plotted_alpha_curves.csv',index=False);capacity.to_csv(dest/'plotted_capacity_lines.csv',index=False)
    write_json(dest/'figure_legends.json',legends);audit_figure(summary,dest)


def audit_figure(summary,dest):
    dest=Path(dest);points=pd.read_csv(dest/'plotted_operating_points.csv')
    curves=pd.read_csv(dest/'plotted_alpha_curves.csv');caps=pd.read_csv(dest/'plotted_capacity_lines.csv')
    keys(points,['candidate','policy'],product(independent_strategy_ids(),POLICIES))
    keys(curves,['partition','weighting','policy','alpha'],product(PARTITIONS,['balanced','unweighted'],POLICIES,[0.,.25,.5,.75,1.]))
    keys(caps,['weighting','policy'],product(['balanced','unweighted'],POLICIES))
    index=summary.set_index(['candidate','policy'])
    for r in points.to_dict('records'):
        close(r,index.loc[(r['candidate'],r['policy'])][['f2','precision','recall']].to_dict())
    for r in curves.to_dict('records'):
        p,w,k,a=[r[v] for v in ['partition','weighting','policy','alpha']]
        n=('baseline_balanced' if w=='balanced' else 'baseline_lgbm') if a==0 else (
            'expert_'+p+'_'+w if a==1 else f'expert_blend_{p}_{w}_a{round(a*100)}')
        if r['candidate']!=n:raise ValueError('Canonical alpha endpoint differs')
        close(r,dict(f2=float(index.loc[(n,k),'f2'])))
    for r in caps.to_dict('records'):
        n='capacity400_'+r['weighting']
        if r['candidate']!=n:raise ValueError('Canonical capacity endpoint differs')
        close(r,dict(f2=float(index.loc[(n,r['policy']),'f2'])))
    expected=[p+' / '+w for p,w in product(PARTITIONS,['balanced','unweighted'])]+['global400 / '+w for w in ['balanced','unweighted']]
    if read_config(dest/'figure_legends.json')!={'operating':POLICIES,**{k:expected for k in POLICIES}}:
        raise ValueError('Figure legends differ')
    return dict(operating_points=len(points),curve_rows=len(curves),capacity_lines=len(caps))


def report(config_path=run.CONFIG):
    c,s=run.load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir'])
    if any((dest/n).exists() for n in ['report_hashes.json','generated_report_hashes.json']):
        raise ValueError('Completed scientific report cannot be overwritten')
    audit=audit_development(config_path,write=False)
    frame=load_data(c);base=features(frame);records,data,masks,final=audit_final(c,s,frame,base)
    cf=pd.read_csv(out/'crossfit.csv');summary=pd.read_csv(out/'summary.csv');h=run.history(c)
    dint,ddraws,fint,fdraws=uncertainty(c,s,cf,h,data,masks)
    for name,table in [('development_intervals',dint),('development_draws',ddraws),('consumed_intervals',fint),('consumed_draws',fdraws)]:
        table.to_csv(dest/(name+'.csv'),index=False)
    for name in run.TABLES+run.WORK_TABLES+['validation_complete.json','independent_development_audit.json','selection.json',
            'exploratory_access.json','transfer_complete.json','exploratory.csv','declaration.json','checks.json',
            'preparation.json','resource_preflight.json','source_miniature_gate.json','older_miniature_audit.json','miniature_gate.json']:
        shutil.copyfile(out/name,dest/name)
    shutil.copyfile(config_path,dest/'config.json');shutil.copyfile(c['protocol_path'],dest/'protocol.md')
    operating_figure(summary,dest)
    with tempfile.TemporaryDirectory(prefix='conditional-figure-') as temporary:
        operating_figure(summary,Path(temporary))
        if digest(Path(temporary)/'conditional_experts_operating.png')!=digest(dest/'conditional_experts_operating.png'):
            raise ValueError('Figure pixel regeneration differs')
    figure=audit_figure(summary,dest)
    index=summary.set_index(['candidate','policy']);winner=summary.iloc[0];final_index=final.set_index('role')
    gains={a:float(winner.f2-index.loc[tuple(nk),'f2']) for a,nk in run.scores.REFERENCE_ROLES.items()}
    consumed={a:float(final_index.loc['selected','f2']-final_index.loc[a,'f2']) for a in gains}
    measured=all(gains[a]>0 and consumed[a]>0 for a in gains)
    result=dict(fingerprint=s['fingerprint'],selection_sha256=digest(out/'selection.json'),
        development_gain=gains,consumed_gain=consumed,measured_gain_on_both_scopes=measured,
        direct_draws_checked=1500,saved_development_models_checked=audit['saved_model_checks'],
        saved_final_models_checked=len(records),visible_roles=len(s['choices']),
        development_intervals=len(dint),consumed_intervals=len(fint),figure=figure,
        conditional_clear_development_gain=all(float(dint[(dint.scope=='macro')&(dint.reference==a)].iloc[0].ci_low)>0 for a in gains),
        conditional_clear_consumed_gain=all(float(fint[fint.reference==a].iloc[0].ci_low)>0 for a in gains),
        figure_regeneration_identical=True,figure_visually_inspected=False,
        adaptive_partition_and_repeated_search=True,month7_previously_consumed=True)
    write_json(dest/'audit.json',result)
    contrasts=pd.read_csv(out/'contrasts.csv')
    effects=contrasts.groupby('contrast',as_index=False).agg(rows=('f2_difference','size'),
        minimum=('f2_difference','min'),maximum=('f2_difference','max'),
        positive=('f2_difference',lambda x:int((x>0).sum())),negative=('f2_difference',lambda x:int((x<0).sum())),
        tied=('f2_difference',lambda x:int((x==0).sum())))
    effects.to_csv(dest/'matched_effect_summary.csv',index=False)
    ref_table=pd.DataFrame([dict(reference=a,development_difference=gains[a],consumed_difference=consumed[a],
        development_pass=gains[a]>0,consumed_pass=consumed[a]>0) for a in gains])
    ref_table.to_csv(dest/'required_reference_comparisons.csv',index=False)
    from .spline_report import markdown
    prep=read_config(out/'preparation.json');transfer=read_config(out/'transfer_complete.json')
    text=f'''# Conditional LightGBM expert study

Selected **{winner.candidate}/{winner.policy}**, macro chronological development F2 **{winner.f2:.9f}**.
Consumed month-7 F2 is **{float(final_index.loc['selected','f2']):.9f}**. Positive-class F2 beta=2 has no review quota or precision floor.

Measured improvements against all six references on both scopes: **{measured}**. Finite study completion alone does not establish the broader objective. Clear gains additionally require positive conditional interval lower bounds in the named scope.

## Required comparisons

{markdown(ref_table)}

## Operating results

{markdown(summary.head(15))}

{markdown(final)}

## Matched effects

{markdown(effects)}

All negative effects and ties are retained. Raw group counts, routing and errors are exported in eligibility.csv, routing.csv and stratum_errors.csv; these diagnostics did not select new subgroups or thresholds. Global400 controls change nominal round capacity only. They do not equalize runtime, stored trees or histogram costs. Partitioning changes population and total capacity together, so effects do not isolate a pure causal population benefit.

## Actual work

Development audited {prep['base_results']} model results, including {prep['reference_reuses']} original-base reuses, {prep['local_fits']} local fits and {prep['capacity_fits']} global400 fits. Exact absent-source reproduction: bagged {prep['reproduced_bagged_fits']}, spline {prep['reproduced_spline_fits']}, other trees {prep['reproduced_tree_fits']}. New conditional training row-rounds {prep['conditional_train_row_rounds']}, stored trees {prep['conditional_stored_trees']}, original fit seconds {prep['conditional_original_fit_seconds']:.3f}. Sixty new-method fits are separate from sixteen production miniature fits and current/historical methodological test calls.

Shared pure/blend computation uses {audit['measured_work']['shared_computation_row_rounds']} actual prediction row-rounds across the declared conditional methods; per-bucket global/local work is in conditional_inference_work.csv. Shared source scores are not charged again to each blend. Original fit times and process high-water RSS are measured, not isolated incremental resource costs.

Transfer retained {len(s['choices'])} visible roles, {transfer['base_results']} model results, {transfer['total_scores']} full scores and {transfer['subset_scores']} local subset artifacts under frozen bounds. No losing grid was evaluated on month 7. See preparation.json, transfer_complete.json and model provenance for deduplicated work and aliases.

## Conditional uncertainty and limitations

All 1500 paired fixed-decision row-bootstrap draws were independently reconstructed. Intervals omit retraining, adaptive source/partition/policy/selection search, temporal and generative variation, and multiplicity. Seventeen earlier studies informed this exploratory search; partition choices are adaptive analyst choices. Month 7 was already consumed and supplies no independent confirmation. Synthetic BAF and two development evaluation months limit generalization. Settings are analyst-selected, not group decisions.

Codex assisted with design, implementation, experiments, audits and this report. Original source recipes and official API references are retained in the protocol; no public notebook was copied. Preserve frozen execution/access identities and all predecessor studies. Further tuning requires a separate study.
'''
    (dest/'findings.md').write_text(text)
    run.guard(c)
    write_json(dest/'generated_report_hashes.json',{str(p):digest(p) for p in dest.rglob('*') if p.is_file() and p.name not in ['generated_report_hashes.json','report_hashes.json']})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['audit-development','report']);parser.add_argument('--config',default=run.CONFIG)
    args=parser.parse_args();print(json.dumps(globals()[args.action.replace('-','_')](args.config),indent=2))
