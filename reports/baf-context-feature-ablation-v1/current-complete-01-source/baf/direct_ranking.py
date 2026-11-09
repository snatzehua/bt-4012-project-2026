"""Finite direct-ranking study with frozen chronological artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .common import digest,features,load_data,read_config,write_json
from .engineering import NativeInputs
from .improvements import paths,prediction
from .recall_weighted_metrics import POLICIES,aligned,learn,measure,decisions
from .recall_extension import expected_population,order_summary
from .improvement_metrics import confusion_metrics
from .calibrated_policy import verify_table
from . import focal_study as old
from .direct_grouping import grouping,describe,parameters,estimator,raw_scores,fit_checked,integration_gate

CONFIG='configs/direct-ranking.json'
SOURCES=old.SOURCES+['baf/direct_grouping.py','baf/direct_ranking.py','baf/direct_ranking_report.py']


def declared_presets():
    return [dict(id=f'rank_g{g}_l{str(l).replace(".","p")}',kind='rank',group_size=g,reg_lambda=l)
            for g in [256,1024] for l in [.1,1.]]+[
            dict(id=f'binary_l{str(l).replace(".","p")}',kind='binary',group_size=None,reg_lambda=l) for l in [.1,1.]]


def fingerprint(path):
    result=old.fingerprint(path);result['sources']={p:digest(p) for p in SOURCES};return result


def guard(c):
    source=read_config(c['source_config']);old.guard(source)
    if old.fingerprint(c['source_config'])!=c['source_fingerprint']:raise ValueError('Seventh study identity changed')
    for path,sha in c['preserve_sha256'].items():
        if digest(path)!=sha:raise ValueError('Preserved artifact changed: '+path)
    if c['reference_source_config']!=source['source_config'] or c['references']!=source['references']:raise ValueError('Reference recipe changed')
    if c['presets']!=declared_presets() or c['policies']!=POLICIES or c['beta']!=2:raise ValueError('Declared ranking grid changed')
    if c['validation_months']!=[4,5,6] or c['test_month']!=7 or c['seed']!=2026 or c['group_seed']!=2026 or c['threads']!=6:
        raise ValueError('Declared chronology or seed changed')
    if any(c[k]!=source[k] for k in ['data_path','data_sha256','bootstrap_seed','bootstrap_replicates']):raise ValueError('Data or uncertainty recipe changed')
    if c['tree_params']!=dict(n_estimators=800,learning_rate=.03,num_leaves=31,min_child_samples=100,min_child_weight=.001,max_bin=255,
            colsample_bytree=.9,subsample=1,subsample_freq=0,deterministic=True,force_col_wise=True):raise ValueError('Declared tree settings changed')
    ref=read_config(c['reference_source_config']);dest=Path(c['output_dir']).resolve()
    for path in [c['source_config'],c['reference_source_config']]+[ref[k] for k in ['stack_config','extension_config','previous_f2_config','prior_config','original_config']]:
        origin=Path(read_config(path)['output_dir']).resolve()
        if dest==origin or dest in origin.parents or origin in dest.parents:raise ValueError('Destination overlaps preserved study')


def prepare_references(c,identity,frame,base,phase='validation',specs=None,allow_missing=True):
    return old.prepare_references({**c,'source_config':c['reference_source_config']},identity,frame,base,phase,specs,allow_missing)


def model_paths(c,phase,month,name):
    out=Path(c['output_dir']);stem=f'{phase}-m{month}-{name}'
    return out/'models'/(stem+'.joblib'),out/'groups'/(stem+'.json')


def training(c,spec,frame,month):
    train=frame.index[frame.month<month];order,g,gm=grouping(train.to_numpy(),frame.loc[train,'month'].to_numpy(),spec['group_size'] or 256,c['group_seed'])
    meta=describe(order,g,gm,frame.loc[order,'fraud_bool'].to_numpy())
    return train,order,g,{**meta,'used_by_objective':spec['kind']=='rank','construction_size':spec['group_size'] or 256}


def expected_record(c,identity,frame,spec,month,phase,meta):
    return {**old.expected_record(frame,spec,month,phase,identity),'fit_params':parameters(spec,c),
            'ordered_train_ids_sha256':meta['ordered_ids_sha256'],'groups_sha256':meta['groups_sha256'],
            'group_months_sha256':meta['group_months_sha256'],'score_type':'finite raw margin; not probability'}


def audit_model(c,identity,frame,base,spec,month,phase='validation',adapter=None):
    rp,pp=paths(c,phase,month,spec['id']);mp,gp=model_paths(c,phase,month,spec['id']);r=read_config(rp)
    train,order,g,meta=training(c,spec,frame,month);expected=expected_record(c,identity,frame,spec,month,phase,meta)
    if any(r[k]!=v for k,v in expected.items()):raise ValueError('Ranking training/identity changed')
    for path,key in [(pp,'prediction_sha256'),(mp,'model_sha256'),(gp,'groups_file_sha256')]:
        if not path.exists() or digest(path)!=r[key]:raise ValueError('Ranking artifact changed: '+str(path))
    if read_config(gp)!=meta:raise ValueError('Grouping reconstruction differs')
    if r['warnings']:raise ValueError('Ranking warnings need inspection')
    data=prediction(c,phase,month,spec['id']);aligned(data,expected_population(frame,month))
    artifact=joblib.load(mp)
    if artifact['spec']!=spec or artifact['fit_params']!=parameters(spec,c):raise ValueError('Saved ranking recipe changed')
    expected_adapter=NativeInputs().fit(base.loc[train]) if adapter is None else adapter;a=artifact['adapter']
    if a.columns!=expected_adapter.columns or a.levels!=expected_adapter.levels or not a.medians.equals(expected_adapter.medians):
        raise ValueError('Preprocessing population changed')
    raw=raw_scores(artifact['booster'],a.transform(base.loc[data['row_ids']]))
    if not np.allclose(raw,data['scores'],atol=1e-12,rtol=1e-12):raise ValueError('Raw score reconstruction differs')
    with np.load(pp) as z:
        if not np.array_equal(z['raw_margin'],z['scores']):raise ValueError('Raw score transformed')
    if r['feature_names']!=a.columns or r['fitted_tree_count']!=artifact['booster'].current_iteration():raise ValueError('Saved fit dimensions changed')
    if not 1<=r['fitted_tree_count']<=c['tree_params']['n_estimators']:raise ValueError('Invalid fitted tree count')
    return r


def fit_model(c,identity,frame,base,spec,month,phase='validation'):
    rp,pp=paths(c,phase,month,spec['id']);mp,gp=model_paths(c,phase,month,spec['id'])
    present=[p.exists() for p in [rp,pp,mp,gp]]
    if any(present):
        if not all(present):raise ValueError('Partial ranking cache')
        return audit_model(c,identity,frame,base,spec,month,phase)
    train,order,g,meta=training(c,spec,frame,month);ev=frame.index[frame.month==month]
    adapter=NativeInputs().fit(base.loc[train]);x=adapter.transform(base.loc[order]);xe=adapter.transform(base.loc[ev]);model=estimator(spec,c)
    expected=expected_record(c,identity,frame,spec,month,phase,meta);start=time.perf_counter()
    print('Fitting',rp.stem,f'rows={len(train):,}',flush=True)
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=c['threads']):
        warnings.simplefilter('always');fit_checked(model,x,frame.loc[order,'fraud_bool'].to_numpy(),g)
        raw=raw_scores(model.booster_,xe)
    messages=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    if messages:raise ValueError('Inspect ranking fit warnings: '+str(messages))
    mp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(dict(booster=model.booster_,adapter=adapter,spec=spec,fit_params=parameters(spec,c)),mp)
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=raw,raw_margin=raw)
    write_json(gp,meta)
    record={**expected,'model_path':str(mp),'model_sha256':digest(mp),'groups_file_sha256':digest(gp),'prediction_sha256':digest(pp),
            'fit_seconds':time.perf_counter()-start,'warnings':messages,'feature_names':adapter.columns,'reused':False,
            'fitted_tree_count':model.booster_.current_iteration()}
    write_json(rp,record);print('Finished',rp.stem,f"seconds={record['fit_seconds']:.1f}",flush=True)
    return audit_model(c,identity,frame,base,spec,month,phase,adapter)


def all_specs(c):return c['presets']+c['references']


def history(c):return {s['id']:{m:prediction(c,'validation',m,s['id']) for m in c['validation_months']} for s in all_specs(c)}


def crossfit(c):
    h=history(c);rows=[]
    for month in [5,6]:
        for name,items in h.items():
            aligned(items[month],h['baseline_balanced'][month]);past={m:d for m,d in items.items() if m<month}
            for kind in POLICIES:
                rule=learn(past,kind,month)
                rows.append(dict(candidate=name,policy=kind,eval_month=month,source_months=','.join(map(str,rule['source_months'])),
                                 parameters=json.dumps(rule,sort_keys=True),**measure(items[month],rule)))
    table=pd.DataFrame(rows)
    if len(table)!=54:raise ValueError('Incomplete ranking policy grid')
    return table


def control_names(c,name):
    spec=next((s for s in c['presets'] if s['id']==name),None)
    if spec is None:return []
    return [s['id'] for s in c['presets'] if s['reg_lambda']==spec['reg_lambda'] and s['kind']!=spec['kind']]


def export_comparisons(c,cf):
    out=Path(c['output_dir']);summary=order_summary(cf);lookup=summary.set_index(['candidate','policy']);rows=[]
    def add(label,name,ref,kind,refkind=None):
        a=lookup.loc[(name,kind)];b=lookup.loc[(ref,refkind or kind)]
        rows.append(dict(contrast=label,candidate=name,policy=kind,reference=ref,reference_policy=refkind or kind,
            **{m+'_difference':float(a[m]-b[m]) for m in ['f2','precision','recall','flagged_fraction','average_precision']}))
    for spec in c['presets']:
        for kind in POLICIES:
            if spec['kind']=='rank':add('rank minus binary',spec['id'],control_names(c,spec['id'])[0],kind)
            if spec['kind']=='rank' and spec['group_size']==1024:
                ref=next(s['id'] for s in c['presets'] if s['kind']=='rank' and s['group_size']==256 and s['reg_lambda']==spec['reg_lambda'])
                add('group 1024 minus 256',spec['id'],ref,kind)
            if spec['reg_lambda']==1.:
                ref=next(s['id'] for s in c['presets'] if s['kind']==spec['kind'] and s['group_size']==spec['group_size'] and s['reg_lambda']==.1)
                add('lambda 1 minus 0.1',spec['id'],ref,kind)
    for name in summary.candidate.unique():
        for kind in POLICIES:
            for ref in ['baseline_balanced','baseline_lgbm']:add('old pooled reference',name,ref,kind,'pooled')
            if kind!='pooled':add('policy minus own pooled',name,name,kind,'pooled')
    pd.DataFrame(rows).to_csv(out/'contrasts.csv',index=False)
    pooled=[]
    for (name,kind),group in cf.groupby(['candidate','policy']):
        total=group[['tn','fp','fn','tp']].sum();metrics=confusion_metrics(*[total[k] for k in ['tn','fp','fn','tp']])
        pooled.append(dict(candidate=name,policy=kind,**metrics,flagged_fraction=(total.tp+total.fp)/group.n.sum()))
    pd.DataFrame(pooled).to_csv(out/'pooled.csv',index=False)
    score_rows=[]
    for name,items in history(c).items():
        for month,d in items.items():
            raw=d['scores'];score_rows.append(dict(candidate=name,eval_month=month,n=len(raw),score_min=float(raw.min()),score_max=float(raw.max()),
                score_mean=float(raw.mean()),unique_scores=len(np.unique(raw)),below_zero=int((raw<0).sum()),above_one=int((raw>1).sum())))
    pd.DataFrame(score_rows).to_csv(out/'score_diagnostics.csv',index=False)
    return summary


def audit_grid(c,identity,frame,base):
    rows=prepare_references(c,identity,frame,base,allow_missing=False)
    for month in c['validation_months']:
        adapter=NativeInputs().fit(base.loc[frame.index[frame.month<month]])
        for spec in c['presets']:rows.append(audit_model(c,identity,frame,base,spec,month,adapter=adapter))
    if len(rows)!=27:raise ValueError('Incomplete ranking audit')
    return rows


def validate(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():raise ValueError('Frozen ranking study cannot be retuned')
    write_json(out/'integration_gate.json',dict(fingerprint=identity,rows=integration_gate(c),miniature_fits=4))
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);refs=prepare_references(c,identity,frame,base)
    for month in c['validation_months']:
        for spec in c['presets']:fit_model(c,identity,frame,base,spec,month)
    cf=crossfit(c);summary=export_comparisons(c,cf)
    for name,value in [('baseline_balanced',.32780157809618937),('baseline_lgbm',.3226536849212962)]:
        if not np.isclose(summary[(summary.candidate==name)&(summary.policy=='pooled')].iloc[0].f2,value,atol=1e-12,rtol=0):raise ValueError('Old F2 reference mismatch')
    cf.to_csv(out/'crossfit.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
    files=['integration_gate.json','crossfit.csv','summary.csv','contrasts.csv','pooled.csv','score_diagnostics.csv']
    write_json(out/'validation_complete.json',dict(fingerprint=identity,created_at_utc=old.now(),files={p:digest(out/p) for p in files},
        new_development_fits=18,reference_results=9,reference_fits=sum(not r.get('reused',False) for r in refs),reference_reuses=sum(r.get('reused',False) for r in refs),
        miniature_integration_fits=4,forward_rows=54))
    guard(c);print(summary.head(12).to_string(index=False),flush=True)


def freeze(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    complete=read_config(out/'validation_complete.json')
    if complete['fingerprint']!=identity or any(digest(out/p)!=sha for p,sha in complete['files'].items()):raise ValueError('Stale ranking development output')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);audit_grid(c,identity,frame,base)
    cf=crossfit(c);verify_table(cf,out/'crossfit.csv');summary=order_summary(cf);verify_table(summary.reset_index(drop=True),out/'summary.csv')
    winner=summary.iloc[0];name=str(winner.candidate);kind=str(winner.policy);h=history(c)
    choices=dict(selected=[name,kind],balanced=['baseline_balanced','pooled'],unweighted=['baseline_lgbm','pooled'],same_model=[name,'pooled'])
    for ctrl in control_names(c,name):choices[ctrl]=[ctrl,kind]
    rules={alias:learn(h[n],k,7) for alias,(n,k) in choices.items()};needed={n for n,k in choices.values()}
    frozen=dict(created_at_utc=old.now(),fingerprint=identity,config=c,selected_candidate=name,selected_policy=kind,selected_development_f2=float(winner.f2),
        choices=choices,rules=rules,fit_specs=[s for s in all_specs(c) if s['id'] in needed],validation_files=complete['files'],
        artifacts={str(p):digest(p) for folder in ['runs','predictions','models','groups'] for p in (out/folder).glob('*') if p.is_file()},
        month7_previously_consumed=True,exploratory=True)
    write_json(out/'selection.json',frozen);print('Frozen',name,kind,float(winner.f2),flush=True)


def load_selection(config_path=CONFIG):
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['config']!=c or s['fingerprint']!=fingerprint(config_path):raise ValueError('Frozen ranking identity changed')
    for p,sha in s['artifacts'].items():
        if not Path(p).exists() or digest(p)!=sha:raise ValueError('Frozen ranking artifact changed: '+p)
    for p,sha in s['validation_files'].items():
        if digest(out/p)!=sha:raise ValueError('Frozen ranking validation output changed')
    return c,s


def final_data(c,s):return {alias:prediction(c,'exploratory',7,n) for alias,(n,k) in s['choices'].items()}


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);ap=out/'exploratory_access.json'
    if not ap.exists():write_json(ap,dict(access_at_utc=old.now(),selection_sha256=digest(out/'selection.json'),month7_previously_consumed=True))
    access=read_config(ap)
    if access['selection_sha256']!=digest(out/'selection.json') or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Freeze/access mismatch')
    frame=load_data(c);base=features(frame);refids={r['id'] for r in c['references']};records=[];refs=[r for r in s['fit_specs'] if r['id'] in refids]
    if refs:records+=prepare_references(c,s['fingerprint'],frame,base,'exploratory',refs)
    for spec in s['fit_specs']:
        if spec['id'] not in refids:records.append(fit_model(c,s['fingerprint'],frame,base,spec,7,'exploratory'))
    data=final_data(c,s);rows=[]
    for alias,d in data.items():
        aligned(d,data['balanced']);rule=s['rules'][alias]
        rows.append(dict(role=alias,candidate=s['choices'][alias][0],policy=s['choices'][alias][1],eval_month=7,parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    write_json(out/'transfer_complete.json',dict(selection_sha256=digest(out/'selection.json'),table_sha256=digest(out/'exploratory.csv'),base_results=len(records),
        new_model_fits=sum(r['candidate'] not in refids for r in records),reference_fits=sum(r['candidate'] in refids and not r.get('reused',False) for r in records),
        reference_reuses=sum(r.get('reused',False) for r in records)))
    guard(c);print(table[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','freeze','exploratory']);p.add_argument('--config',default=CONFIG)
    a=p.parse_args();globals()[a.command](a.config)
