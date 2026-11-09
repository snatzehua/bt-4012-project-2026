"""Chronological score stacking with preserved studies and immutable selection."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import digest,features,load_data,read_config,write_json
from .engineering import engineer
from .improvements import paths,prediction,save_prediction
from .recall_extension import (SOURCES as OLD_SOURCES,audit_bases,
    fingerprint as extension_fingerprint,guard as extension_guard,run_base)
from .recall_weighted import ids_digest
from .recall_weighted_metrics import POLICIES,aligned,learn,measure
from .recall_extension import order_summary
from .stacking_features import matched_control,policy_history,score_matrix,training_matrix

SOURCES=OLD_SOURCES+["baf/stacking.py","baf/stacking_features.py"]


def fingerprint(path):
    result=extension_fingerprint(path);result['sources']={p:digest(p) for p in SOURCES}
    return result


def guard(config):
    extension_guard(config)
    if extension_fingerprint(config['extension_config'])!=config['extension_fingerprint']:
        raise ValueError('Preserved extension identity changed')
    if config['early_months']!=[1,2,3] or config['validation_months']!=[4,5,6] or config['test_month']!=7:
        raise ValueError('Declared chronology changed')


def base_pairs(config):
    return [(s,m) for m in range(1,7) for s in config['candidates']
            if s['id'] in config['members'] or m in config['validation_months']]


def base_history(config):
    history={s['id']:{} for s in config['candidates']};records={}
    for spec,month in base_pairs(config):
        name=spec['id'];history[name][month]=prediction(config,'validation',month,name)
        records[(name,month)]=read_config(paths(config,'validation',month,name)[0])
    return history,records


def audit_basegrid(config,identity,frame,phase='validation',specs=None):
    if phase!='validation':return audit_bases(config,identity,frame,phase,specs)
    return [audit_bases({**config,'validation_months':[m]},identity,frame,specs=[s])[0] for s,m in base_pairs(config)]


def meta_paths(config,phase,month,name):
    stem=f'{phase}-m{month}-{name}';out=Path(config['output_dir'])
    return out/'meta'/(stem+'.json'),out/'models'/(stem+'.joblib')


def source_hashes(config,spec,months):
    values={}
    for month in months:
        for name in spec['members']:
            for p in paths(config,'validation',month,name):values[str(p)]=digest(p)
    return values


def fit_meta(config,spec,month,identity,history,base_records,phase='validation'):
    rp,mp=meta_paths(config,phase,month,spec['id'])
    x,columns,pop,months=training_matrix(history,base_records,spec,month,config['clip'])
    expected=dict(candidate=spec['id'],phase=phase,spec=spec,fingerprint=identity,eval_month=month,train_months=months,train_month_max=max(months),
                  train_rows=len(pop['labels']),train_fraud=int(pop['labels'].sum()),train_ids_sha256=ids_digest(pop['row_ids']),
                  feature_names=columns,source_hashes=source_hashes(config,spec,months),in_sample=False)
    if rp.exists():
        record=read_config(rp)
        for key,value in expected.items():
            if record[key]!=value:raise ValueError('Stale combiner provenance: '+key)
        if not mp.exists() or record['model_sha256']!=digest(mp):raise ValueError('Combiner artifact changed')
        model=joblib.load(mp)
    else:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=spec['C'],solver='lbfgs',max_iter=1000,tol=1e-6,
                            class_weight={0:1,1:spec['pos_weight']},random_state=config['seed']))
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=config['threads']):
            warnings.simplefilter('always');model.fit(x,pop['labels'])
        warning_text=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if warning_text:raise ValueError('Combiner fit warnings require inspection: '+str(warning_text))
        mp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,mp)
        record={**expected,'candidate':spec['id'],'phase':phase,'model_path':str(mp),'model_sha256':digest(mp),
                'fit_seconds':time.perf_counter()-start,'warnings':warning_text}
        write_json(rp,record)
        print('Fitted',rp.stem,f"rows={len(x):,}",flush=True)
    if not np.allclose(model.named_steps['standardscaler'].mean_,x.mean(axis=0),atol=1e-12):
        raise ValueError('Scaler is not fitted on eligible score rows')
    if phase=='validation':
        current={n:history[n][month] for n in spec['members']}
        data=predict_meta(config,record,model,current)
        pp=paths(config,'validation',month,spec['id'])[1]
        if 'prediction_sha256' in record:
            saved=prediction(config,'validation',month,spec['id']);aligned(saved,data)
            if digest(pp)!=record['prediction_sha256'] or not np.allclose(saved['scores'],data['scores'],atol=1e-12,rtol=1e-12):
                raise ValueError('Cached combiner predictions changed')
        else:
            save_prediction(pp,data['row_ids'],data['labels'],data['scores'])
            record.update(prediction_sha256=digest(pp),evaluation_ids_sha256=ids_digest(data['row_ids']))
            write_json(rp,record)
    return record


def predict_meta(config,record,model,items):
    spec=record['spec'];x,cols,ref=score_matrix(items,spec['members'],spec['representation'],config['clip'])
    if cols!=record['feature_names']:raise ValueError('Frozen feature order changed')
    with threadpool_limits(limits=config['threads']):scores=model.predict_proba(x)[:,1]
    return dict(row_ids=ref['row_ids'],labels=ref['labels'],scores=scores)


def histories(config):
    history={};records={}
    for name in ['baseline_balanced','baseline_lgbm']+[s['id'] for s in config['combiners']]:
        history[name]={}
        for month in config['validation_months']:
            history[name][month]=prediction(config,'validation',month,name)
            rp=paths(config,'validation',month,name)[0] if name.startswith('baseline_') else meta_paths(config,'validation',month,name)[0]
            records[(name,month)]=read_config(rp)
    return history,records


def crossfit(config):
    history,records=histories(config);rows=[]
    for month in config['validation_months'][1:]:
        for name,items in history.items():
            aligned(items[month],history['baseline_balanced'][month])
            past=policy_history(history,records,name,month,config['validation_months'])
            for kind in POLICIES:
                rule=learn(past,kind,month)
                rows.append(dict(candidate=name,policy=kind,eval_month=month,source_months=','.join(map(str,rule['source_months'])),
                                 parameters=json.dumps(rule,sort_keys=True),**measure(items[month],rule)))
    cf=pd.DataFrame(rows)
    if len(cf)!=config['expected_crossfit_rows'] or cf.candidate.nunique()!=config['expected_strategy_count']:
        raise ValueError('Incomplete forward evaluation grid')
    return cf


def validate(config_path):
    config=read_config(config_path);guard(config);identity=fingerprint(config_path);out=Path(config['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Frozen study cannot be retuned')
    frame=load_data(config);frame=frame[frame.month<config['test_month']];base=features(frame);eng=engineer(base)
    for spec,month in base_pairs(config):run_base(frame,base,eng,spec,month,'validation',config,identity)
    history,records=base_history(config)
    for month in config['validation_months']:
        for spec in config['combiners']:fit_meta(config,spec,month,identity,history,records)
    cf=crossfit(config);summary=order_summary(cf)
    prior=pd.read_csv('reports/baf-recall-weighted-v1/summary.csv').set_index(['candidate','policy'])
    for name in ['baseline_balanced','baseline_lgbm']:
        actual=summary[(summary.candidate==name)&(summary.policy=='pooled')].iloc[0].f2
        if not np.isclose(actual,prior.loc[(name,'pooled'),'f2'],atol=1e-12):raise ValueError('Prior F2 baseline mismatch')
    cf.to_csv(out/'crossfit.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
    write_json(out/'validation_complete.json',dict(fingerprint=identity,crossfit_sha256=digest(out/'crossfit.csv'),summary_sha256=digest(out/'summary.csv')))
    guard(config);print(summary.head(12).to_string(index=False),flush=True)


def verify_tables(config,cf):
    stored=pd.read_csv(Path(config['output_dir'])/'crossfit.csv')
    if cf.shape!=stored.shape:raise ValueError('Forward table shape changed')
    for col in cf:
        if pd.api.types.is_numeric_dtype(cf[col]):
            if not np.allclose(cf[col],stored[col],atol=1e-12,rtol=1e-10):raise ValueError('Forward metric changed')
        elif list(cf[col].astype(str))!=list(stored[col].astype(str)):raise ValueError('Forward policy provenance changed')


def audit_metas(config,identity,history,records):
    values=[]
    for month in config['validation_months']:
        for spec in config['combiners']:
            rp,mp=meta_paths(config,'validation',month,spec['id'])
            pp=paths(config,'validation',month,spec['id'])[1]
            if not all(p.exists() for p in [rp,mp,pp]) or 'prediction_sha256' not in read_config(rp):
                raise ValueError('Required combiner audit artifact is missing')
            values.append(fit_meta(config,spec,month,identity,history,records))
    if len(values)!=config['expected_meta_results']:raise ValueError('Incomplete combiner grid')
    return values


def freeze(config_path):
    config=read_config(config_path);guard(config);identity=fingerprint(config_path);out=Path(config['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    complete=read_config(out/'validation_complete.json')
    if complete['fingerprint']!=identity or any(complete[k+'_sha256']!=digest(out/(k+'.csv')) for k in ['crossfit','summary']):
        raise ValueError('Stale development outputs')
    frame=load_data(config);frame=frame[frame.month<config['test_month']]
    audit_basegrid(config,identity,frame);history,records=base_history(config);audit_metas(config,identity,history,records)
    cf=crossfit(config);verify_tables(config,cf);winner=order_summary(cf).iloc[0]
    specs={s['id']:s for s in config['combiners']};name=str(winner.candidate);learned=[]
    if name in specs:
        learned=[specs[name]];control=matched_control(specs[name],config['combiners'])
        if control:learned.append(control)
    final_fits={s['id']:fit_meta(config,s,7,identity,history,records,'frozen') for s in learned}
    names=sorted({name,'baseline_balanced','baseline_lgbm'}|{s['id'] for s in learned})
    h,r=histories(config)
    rules={n:{kind:learn(policy_history(h,r,n,7,config['validation_months']),kind,7) for kind in POLICIES} for n in names}
    needed={'baseline_balanced','baseline_lgbm'}|{n for s in learned for n in s['members']}
    selection=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),fingerprint=identity,config=config,
                   selected_candidate=name,selected_policy=str(winner.policy),selected_development_f2=float(winner.f2),
                   evaluate_names=names,learned_specs=learned,final_fits=final_fits,rules=rules,
                   fit_specs=[s for s in config['candidates'] if s['id'] in needed],month7_previously_consumed=True,exploratory=True)
    write_json(out/'selection.json',selection);print('Frozen',name,winner.policy,winner.f2,flush=True)


def load_selection(config_path):
    config=read_config(config_path);guard(config);s=read_config(Path(config['output_dir'])/'selection.json')
    if s['config']!=config or s['fingerprint']!=fingerprint(config_path):raise ValueError('Frozen identity changed')
    for name,r in s['final_fits'].items():
        rp,mp=meta_paths(config,'frozen',7,name)
        if r!=read_config(rp) or r['model_sha256']!=digest(mp):raise ValueError('Frozen combiner changed')
    return config,s


def final_predictions(config,selected):
    data={s['id']:prediction(config,'exploratory',7,s['id']) for s in selected['fit_specs']}
    for spec in selected['learned_specs']:
        record=selected['final_fits'][spec['id']];model=joblib.load(record['model_path'])
        data[spec['id']]=predict_meta(config,record,model,{n:data[n] for n in spec['members']})
    return {name:data[name] for name in selected['evaluate_names']}


def exploratory(config_path):
    config,selected=load_selection(config_path);out=Path(config['output_dir']);ap=out/'exploratory_access.json'
    if not ap.exists():write_json(ap,dict(access_at_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=digest(out/'selection.json'),month7_previously_consumed=True))
    access=read_config(ap)
    if access['selection_sha256']!=digest(out/'selection.json') or not selected['created_at_utc']<access['access_at_utc']:
        raise ValueError('Selection/access mismatch')
    frame=load_data(config);base=features(frame);eng=engineer(base)
    for spec in selected['fit_specs']:run_base(frame,base,eng,spec,7,'exploratory',config,selected['fingerprint'])
    data=final_predictions(config,selected);rows=[]
    learned={s['id'] for s in selected['learned_specs']}
    for name,d in data.items():
        aligned(d,data['baseline_balanced'])
        if name in learned:save_prediction(paths(config,'exploratory',7,name)[1],d['row_ids'],d['labels'],d['scores'])
        for kind,rule in selected['rules'][name].items():
            rows.append(dict(candidate=name,policy=kind,selected=name==selected['selected_candidate'] and kind==selected['selected_policy'],
                             parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    print(table[['candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']].to_string(index=False),flush=True)
    guard(config)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','freeze','exploratory']);p.add_argument('--config',default='configs/chronological-stacking.json')
    args=p.parse_args();globals()[args.command](args.config)
