"""Finite focal-loss score study with immutable chronology and matched controls."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from .common import digest,features,load_data,read_config,write_json
from .engineering import NativeInputs,engineer
from .improvements import paths,prediction
from .recall_weighted import ids_digest
from .recall_weighted_metrics import POLICIES,aligned,learn,measure,decisions
from .recall_extension import audit_bases,expected_population,order_summary
from .calibrated_policy import (SOURCES as OLD_SOURCES,fingerprint as source_fingerprint,guard as source_guard,
    ensure_base,verify_table)
from .calibrated_policy_metrics import evaluate
from .focal_loss import estimator,scores,integration_gate

CONFIG='configs/focal-loss.json'
SOURCES=OLD_SOURCES+['baf/focal_loss.py','baf/focal_study.py','baf/focal_report.py']


def now():return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    result=source_fingerprint(path);result['sources']={p:digest(p) for p in SOURCES};return result


def reference_config(config):
    return {**read_config(config['source_config']),'output_dir':config['output_dir']}


def declared_presets():
    return [dict(id=f'custom_g{gamma}_w{weight}',kind='custom',gamma=gamma,positive_weight=weight)
            for gamma in [0,1,2] for weight in [1,3]]+[
            dict(id=f'builtin_w{weight}',kind='builtin',gamma=0,positive_weight=weight) for weight in [1,3]]


def guard(config):
    source=read_config(config['source_config']);source_guard(source)
    if source_fingerprint(config['source_config'])!=config['source_fingerprint']:raise ValueError('Sixth study identity changed')
    for p,sha in config['preserve_sha256'].items():
        if digest(p)!=sha:raise ValueError('Preserved artifact changed: '+p)
    if config['presets']!=declared_presets() or config['policies']!=POLICIES or config['beta']!=2:
        raise ValueError('Declared loss grid or objective changed')
    if config['validation_months']!=[4,5,6] or config['test_month']!=7:raise ValueError('Declared chronology changed')
    if config['data_path']!=source['data_path'] or config['data_sha256']!=source['data_sha256']:raise ValueError('Data recipe changed')
    if [s['id'] for s in config['references']]!=['baseline_balanced','baseline_lgbm','native31']:
        raise ValueError('Reference grid changed')
    if any(s not in source['candidates'] for s in config['references']):raise ValueError('Reference recipe changed')
    dest=Path(config['output_dir']).resolve()
    for path in [config['source_config']]+[source[k] for k in ['stack_config','extension_config','previous_f2_config','prior_config','original_config']]:
        origin=Path(read_config(path)['output_dir']).resolve()
        if dest==origin or dest in origin.parents or origin in dest.parents:raise ValueError('Destination overlaps prior study')


def model_paths(config,phase,month,name):
    stem=f'{phase}-m{month}-{name}';out=Path(config['output_dir'])
    return out/'models'/(stem+'.joblib'),out/'numeric'/(stem+'.json')


def expected_record(frame,spec,month,phase,identity):
    train=frame.index[frame.month<month].to_numpy();ev=frame.index[frame.month==month].to_numpy()
    if not len(train) or not len(ev):raise ValueError('Empty chronological population')
    return dict(candidate=spec['id'],spec=spec,phase=phase,eval_month=month,fingerprint=identity,
        train_rows=len(train),train_fraud=int(frame.loc[train,'fraud_bool'].sum()),train_month_max=int(frame.loc[train,'month'].max()),
        train_month_counts={str(k):int(v) for k,v in frame.loc[train,'month'].value_counts().sort_index().items()},
        train_ids_sha256=ids_digest(train),evaluation_ids_sha256=ids_digest(ev))


def audit_loss(config,identity,frame,base,spec,month,phase='validation',adapter=None):
    rp,pp=paths(config,phase,month,spec['id']);mp,tp=model_paths(config,phase,month,spec['id'])
    record=read_config(rp);expected=expected_record(frame,spec,month,phase,identity)
    if any(record[k]!=v for k,v in expected.items()):raise ValueError('Loss training/identity changed')
    for p,key in [(pp,'prediction_sha256'),(mp,'model_sha256'),(tp,'numeric_sha256')]:
        if not p.exists() or digest(p)!=record[key]:raise ValueError('Loss artifact changed: '+str(p))
    if record['warnings']:raise ValueError('Fit warnings need inspection')
    data=prediction(config,phase,month,spec['id']);aligned(data,expected_population(frame,month))
    artifact=joblib.load(mp)
    if artifact['spec']!=spec or artifact['tree_params']!=config['tree_params']:raise ValueError('Saved recipe changed')
    train=frame.index[frame.month<month];ev=frame.index[frame.month==month]
    expected_adapter=NativeInputs().fit(base.loc[train]) if adapter is None else adapter
    actual=artifact['adapter']
    if actual.columns!=expected_adapter.columns or actual.levels!=expected_adapter.levels or not actual.medians.equals(expected_adapter.medians):
        raise ValueError('Preprocessing training population changed')
    raw,p=scores(artifact['booster'],actual.transform(base.loc[ev]))
    with np.load(pp) as saved:
        if not np.allclose(raw,saved['raw_margin'],atol=1e-12,rtol=1e-12):raise ValueError('Raw margin reconstruction differs')
    if not np.allclose(p,data['scores'],atol=1e-12,rtol=1e-12):raise ValueError('Score reconstruction differs')
    if record['feature_names']!=actual.columns:raise ValueError('Recorded feature order changed')
    trace=read_config(tp)
    if spec['kind']=='custom':
        if len(trace)!=config['tree_params']['n_estimators'] or not trace or artifact['booster'].current_iteration()>len(trace):raise ValueError('Numeric trace incomplete')
        expected_first=(len(train)-expected['train_fraud']+spec['positive_weight']*expected['train_fraud'])*np.log(2)*.5**spec['gamma']/len(train)
        if not np.isclose(trace[0]['mean_loss'],expected_first,atol=1e-12):raise ValueError('Loss initialization or training weight differs')
        for i,r in enumerate(trace):
            if r['iteration']!=i+1 or r['n']!=len(train) or not 0<=r['replaced_count']<=len(train):raise ValueError('Numeric trace population differs')
            if not np.isclose(r['replaced_fraction'],r['replaced_count']/len(train)):raise ValueError('Numeric fraction differs')
            if spec['gamma']==0 and r['replaced_count']!=0:raise ValueError('Ordinary loss was floored')
            if spec['gamma']>0 and r['supplied_hessian_min']<config['hessian_floor']:raise ValueError('Hessian floor violated')
    elif trace:raise ValueError('Built-in control has a custom trace')
    if record['fitted_tree_count']!=artifact['booster'].current_iteration() or record['objective_calls']!=len(trace):raise ValueError('Fitted tree/callback count differs')
    return record


def fit_loss(config,identity,frame,base,spec,month,phase='validation'):
    rp,pp=paths(config,phase,month,spec['id']);mp,tp=model_paths(config,phase,month,spec['id'])
    present=[p.exists() for p in [rp,pp,mp,tp]]
    if any(present):
        if not all(present):raise ValueError('Partial loss cache')
        return audit_loss(config,identity,frame,base,spec,month,phase)
    expected=expected_record(frame,spec,month,phase,identity);train=frame.index[frame.month<month];ev=frame.index[frame.month==month]
    adapter=NativeInputs().fit(base.loc[train]);x=adapter.transform(base.loc[train]);xe=adapter.transform(base.loc[ev])
    trace=[];model=estimator(spec,config,trace);start=time.perf_counter()
    print('Fitting',rp.stem,f'rows={len(train):,}',flush=True)
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=config['threads']):
        warnings.simplefilter('always');model.fit(x,frame.loc[train,'fraud_bool'].to_numpy());raw,p=scores(model.booster_,xe)
    messages=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    if messages:raise ValueError('Inspect loss fit warnings: '+str(messages))
    mp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(dict(booster=model.booster_,adapter=adapter,spec=spec,tree_params=config['tree_params']),mp)
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=p,raw_margin=raw)
    write_json(tp,trace)
    record={**expected,'model_path':str(mp),'model_sha256':digest(mp),'numeric_sha256':digest(tp),'prediction_sha256':digest(pp),
        'fit_seconds':time.perf_counter()-start,'warnings':messages,'feature_names':adapter.columns,'reused':False,
        'fitted_tree_count':model.booster_.current_iteration(),'objective_calls':len(trace)}
    write_json(rp,record)
    print('Finished',rp.stem,f"seconds={record['fit_seconds']:.1f}",flush=True)
    return audit_loss(config,identity,frame,base,spec,month,phase,adapter)


def prepare_references(config,identity,frame,base,phase='validation',specs=None,allow_missing=True):
    specs=config['references'] if specs is None else specs;ref=reference_config(config);results=[];eng=None
    for month in config['validation_months'] if phase=='validation' else [7]:
        for spec in specs:
            if not allow_missing and not all(p.exists() for p in paths(ref,phase,month,spec['id'])):raise ValueError('Reference missing at audit')
            # The old recipes are exact; ignored caches are optional in a new destination.
            if eng is None:
                old=read_config(ref['stack_config'])
                if not paths(ref,phase,month,spec['id'])[0].exists() and not paths(old,phase,month,spec['id'])[0].exists():eng=engineer(base)
            results.append(ensure_base(frame,base,eng,spec,month,phase,ref,identity))
    return results


def all_specs(config):return config['presets']+config['references']


def history(config):
    return {s['id']:{m:prediction(config,'validation',m,s['id']) for m in config['validation_months']} for s in all_specs(config)}


def crossfit(config):
    h=history(config);rows=[]
    for month in [5,6]:
        for name,items in h.items():
            aligned(items[month],h['baseline_balanced'][month]);past={m:items[m] for m in config['validation_months'] if m<month}
            for kind in POLICIES:
                rule=learn(past,kind,month);mask=decisions(items[month],rule)
                rows.append(dict(candidate=name,policy=kind,eval_month=month,source_months=','.join(map(str,rule['source_months'])),
                                 parameters=json.dumps(rule,sort_keys=True),**evaluate(items[month],mask)))
    cf=pd.DataFrame(rows)
    if len(cf)!=66:raise ValueError('Incomplete forward loss grid')
    return cf


def control_names(config,name):
    spec=next((s for s in config['presets'] if s['id']==name),None)
    if spec is None:return []
    weight=spec['positive_weight'];ordinary=f'custom_g0_w{weight}';builtin=f'builtin_w{weight}'
    return [n for n in [ordinary,builtin] if n!=name]


def export_comparisons(config,cf):
    out=Path(config['output_dir']);summary=order_summary(cf);lookup=summary.set_index(['candidate','policy']);rows=[]
    def add(label,name,ref,kind,refkind=None):
        a=lookup.loc[(name,kind)];b=lookup.loc[(ref,refkind or kind)]
        rows.append(dict(contrast=label,candidate=name,policy=kind,reference=ref,reference_policy=refkind or kind,
            **{m+'_difference':float(a[m]-b[m]) for m in ['f2','precision','recall','flagged_fraction','average_precision']}))
    for spec in config['presets']:
        for kind in POLICIES:
            if spec['gamma']>0:add('focal minus custom logistic',spec['id'],f"custom_g0_w{spec['positive_weight']}",kind)
            if spec['kind']=='custom' and spec['gamma']==0:add('custom minus built-in logistic',spec['id'],f"builtin_w{spec['positive_weight']}",kind)
    for name in summary.candidate.unique():
        for kind in POLICIES:
            for ref in ['baseline_balanced','baseline_lgbm']:add('old pooled reference',name,ref,kind,'pooled')
            if kind!='pooled':add('policy minus own pooled',name,name,kind,'pooled')
    pd.DataFrame(rows).to_csv(out/'contrasts.csv',index=False)
    pooled=[]
    from .improvement_metrics import confusion_metrics
    for (name,kind),group in cf.groupby(['candidate','policy']):
        total=group[['tn','fp','fn','tp']].sum();metrics=confusion_metrics(*[total[k] for k in ['tn','fp','fn','tp']])
        pooled.append(dict(candidate=name,policy=kind,**metrics,flagged_fraction=(total.tp+total.fp)/group.n.sum()))
    pd.DataFrame(pooled).to_csv(out/'pooled.csv',index=False)
    discrepancies=[]
    for month in [4,5,6]:
        for weight in [1,3]:
            arrays={}
            for kind,name in [('custom',f'custom_g0_w{weight}'),('builtin',f'builtin_w{weight}')]:
                with np.load(paths(config,'validation',month,name)[1]) as z:arrays[kind]={k:z[k] for k in z.files}
            aligned(arrays['custom'],arrays['builtin']);a,b=arrays['custom'],arrays['builtin']
            discrepancies.append(dict(eval_month=month,positive_weight=weight,max_raw_difference=float(np.max(np.abs(a['raw_margin']-b['raw_margin']))),
                mean_raw_difference=float(np.mean(np.abs(a['raw_margin']-b['raw_margin']))),max_probability_difference=float(np.max(np.abs(a['scores']-b['scores']))),
                mean_probability_difference=float(np.mean(np.abs(a['scores']-b['scores']))),spearman_rank=float(spearmanr(a['scores'],b['scores']).statistic)))
    pd.DataFrame(discrepancies).to_csv(out/'ordinary_score_diagnostics.csv',index=False)
    return summary


def audit_grid(config,identity,frame,base):
    results=prepare_references(config,identity,frame,base,allow_missing=False)
    for month in config['validation_months']:
        adapter=NativeInputs().fit(base.loc[frame.index[frame.month<month]])
        for spec in config['presets']:results.append(audit_loss(config,identity,frame,base,spec,month,adapter=adapter))
    if len(results)!=33:raise ValueError('Incomplete prediction audit')
    return results


def validate(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():raise ValueError('Frozen study cannot be retuned')
    gate=integration_gate(c);write_json(out/'integration_gate.json',dict(fingerprint=identity,rows=gate,miniature_fits=4))
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
    refs=prepare_references(c,identity,frame,base)
    for month in c['validation_months']:
        for spec in c['presets']:fit_loss(c,identity,frame,base,spec,month)
    cf=crossfit(c);summary=export_comparisons(c,cf)
    for name,value in [('baseline_balanced',.32780157809618937),('baseline_lgbm',.3226536849)]:
        actual=summary[(summary.candidate==name)&(summary.policy=='pooled')].iloc[0].f2
        if not np.isclose(actual,value,atol=1e-10,rtol=0):raise ValueError('Old F2 baseline mismatch')
    cf.to_csv(out/'crossfit.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
    files=['integration_gate.json','crossfit.csv','summary.csv','contrasts.csv','pooled.csv','ordinary_score_diagnostics.csv']
    write_json(out/'validation_complete.json',dict(fingerprint=identity,created_at_utc=now(),files={p:digest(out/p) for p in files},
        new_development_fits=24,reference_results=9,reference_fits=sum(not r.get('reused',False) for r in refs),
        reference_reuses=sum(r.get('reused',False) for r in refs),miniature_integration_fits=4,forward_rows=66))
    guard(c);print(summary.head(12).to_string(index=False),flush=True)


def freeze(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    complete=read_config(out/'validation_complete.json')
    if complete['fingerprint']!=identity or any(digest(out/p)!=sha for p,sha in complete['files'].items()):raise ValueError('Stale development outputs')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);audit_grid(c,identity,frame,base)
    cf=crossfit(c);verify_table(cf,out/'crossfit.csv');summary=order_summary(cf);verify_table(summary.reset_index(drop=True),out/'summary.csv')
    winner=summary.iloc[0];name=str(winner.candidate);kind=str(winner.policy);h=history(c)
    choices=dict(selected=[name,kind],balanced=['baseline_balanced','pooled'],unweighted=['baseline_lgbm','pooled'],same_model=[name,'pooled'])
    for ctrl in control_names(c,name):choices[ctrl]=[ctrl,kind]
    rules={alias:learn(h[n],k,7) for alias,(n,k) in choices.items()}
    needed={n for n,k in choices.values()};specs=[s for s in all_specs(c) if s['id'] in needed]
    frozen=dict(created_at_utc=now(),fingerprint=identity,config=c,selected_candidate=name,selected_policy=kind,
        selected_development_f2=float(winner.f2),choices=choices,rules=rules,fit_specs=specs,validation_files=complete['files'],
        artifacts={str(p):digest(p) for folder in ['runs','predictions','models','numeric'] for p in (out/folder).glob('*') if p.is_file()},
        month7_previously_consumed=True,exploratory=True)
    write_json(out/'selection.json',frozen);print('Frozen',name,kind,float(winner.f2),flush=True)


def load_selection(config_path=CONFIG):
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['config']!=c or s['fingerprint']!=fingerprint(config_path):raise ValueError('Frozen study identity changed')
    for p,sha in s['artifacts'].items():
        if not Path(p).exists() or digest(p)!=sha:raise ValueError('Frozen loss artifact changed: '+p)
    for p,sha in s['validation_files'].items():
        if digest(out/p)!=sha:raise ValueError('Frozen validation output changed')
    return c,s


def final_data(config,selected):
    return {alias:prediction(config,'exploratory',7,n) for alias,(n,k) in selected['choices'].items()}


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);ap=out/'exploratory_access.json'
    if not ap.exists():write_json(ap,dict(access_at_utc=now(),selection_sha256=digest(out/'selection.json'),month7_previously_consumed=True))
    access=read_config(ap)
    if access['selection_sha256']!=digest(out/'selection.json') or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Selection/access mismatch')
    frame=load_data(c);base=features(frame);refids={r['id'] for r in c['references']};records=[]
    refs=[r for r in s['fit_specs'] if r['id'] in refids]
    if refs:records+=prepare_references(c,s['fingerprint'],frame,base,'exploratory',refs)
    for spec in s['fit_specs']:
        if spec['id'] not in refids:records.append(fit_loss(c,s['fingerprint'],frame,base,spec,7,'exploratory'))
    data=final_data(c,s);rows=[]
    for alias,d in data.items():
        aligned(d,data['balanced']);rule=s['rules'][alias];mask=decisions(d,rule)
        rows.append(dict(role=alias,candidate=s['choices'][alias][0],policy=s['choices'][alias][1],eval_month=7,
                         parameters=json.dumps(rule,sort_keys=True),**evaluate(d,mask)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    write_json(out/'transfer_complete.json',dict(selection_sha256=digest(out/'selection.json'),table_sha256=digest(out/'exploratory.csv'),
        base_results=len(records),new_loss_fits=sum(r['candidate'] not in refids for r in records),
        reference_fits=sum(r['candidate'] in refids and not r.get('reused',False) for r in records),
        reference_reuses=sum(r.get('reused',False) for r in records)))
    guard(c);print(table[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','freeze','exploratory']);p.add_argument('--config',default=CONFIG)
    a=p.parse_args();globals()[a.command](a.config)
