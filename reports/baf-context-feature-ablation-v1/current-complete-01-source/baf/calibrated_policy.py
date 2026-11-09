"""Separate finite calibrated-F2 study with immutable, chronological provenance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from .common import digest, features, load_data, read_config, write_json
from .engineering import engineer
from .improvements import paths, prediction, save_prediction
from .recall_extension import audit_bases, run_base, expected_population, order_summary
from .recall_weighted import ids_digest
from .recall_weighted_metrics import POLICIES, aligned, decisions, learn
from .stacking import (SOURCES as STACK_SOURCES, guard as stack_guard, fingerprint as stack_fingerprint,
    base_pairs, base_history, fit_meta, meta_paths, source_hashes, predict_meta)
from .stacking_features import policy_history, training_matrix, verify_source
from .calibrated_policy_metrics import (CALIBRATION_POLICIES, fit_calibrator, calibrated, calibration_population,
    batch_decisions, evaluate, reliability, hindsight_prefix)

CONFIG='configs/calibrated-f2-policy.json'
SOURCES=STACK_SOURCES+['baf/calibrated_policy_metrics.py','baf/calibrated_policy.py','baf/calibrated_policy_report.py']


def now(): return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    result=stack_fingerprint(path); result['sources']={p:digest(p) for p in SOURCES}; return result


def guard(config):
    stack_guard(config)
    if stack_fingerprint(config['stack_config'])!=config['stack_fingerprint']:
        raise ValueError('Preserved stacking identity changed')
    if config['calibration_policies']!=CALIBRATION_POLICIES or config['score_sources']!=[
        'baseline_balanced','baseline_lgbm','stack_logit_percentile_c0p01_w1']:
        raise ValueError('Declared policy grid changed')
    old=read_config(config['stack_config'])
    for field in ['candidates','members','clip','seed','threads','early_months','validation_months','test_month']:
        if config[field]!=old[field]: raise ValueError('Frozen source recipe changed: '+field)
    chosen=[s for s in old['combiners'] if s['id']==config['score_sources'][-1]]
    if config['combiners']!=chosen: raise ValueError('Fixed combiner recipe changed')
    dest=Path(config['output_dir']).resolve()
    prior=[config[k] for k in ['stack_config','extension_config','previous_f2_config','prior_config','original_config']]
    for path in prior:
        origin=Path(read_config(path)['output_dir']).resolve()
        if dest==origin or origin in dest.parents or dest in origin.parents:
            raise ValueError('Destination overlaps a preserved study')


def source_record(config,phase,month,name):
    return paths(config,phase,month,name)[0] if name.startswith('baseline_') else meta_paths(config,phase,month,name)[0]


def ensure_base(frame,base,eng,spec,month,phase,config,identity):
    rp,pp=paths(config,phase,month,spec['id']);old=read_config(config['stack_config'])
    old_rp,old_pp=paths(old,phase,month,spec['id'])
    if rp.exists()!=pp.exists(): raise ValueError('Partial own base cache')
    if rp.exists():
        return audit_bases({**config,'validation_months':[month]},identity,frame,phase,[spec])[0]
    if old_rp.exists()!=old_pp.exists(): raise ValueError('Partial preserved base cache')
    if old_rp.exists():
        raw=audit_bases({**old,'validation_months':[month]},config['stack_fingerprint'],frame,phase,[spec])[0]
        data=prediction(old,phase,month,spec['id'])
        save_prediction(pp,data['row_ids'],data['labels'],data['scores'])
        record={**raw,'fingerprint':identity,'reused':True,'prediction_sha256':digest(pp),
                'provenance':dict(record=str(old_rp),record_sha256=digest(old_rp),prediction=str(old_pp),prediction_sha256=digest(old_pp))}
        write_json(rp,record)
        return audit_bases({**config,'validation_months':[month]},identity,frame,phase,[spec])[0]
    return run_base(frame,base,eng,spec,month,phase,config,identity)


def ensure_meta(config,spec,month,identity,history,records):
    rp,mp=meta_paths(config,'validation',month,spec['id']);pp=paths(config,'validation',month,spec['id'])[1]
    if rp.exists():
        if not mp.exists() or not pp.exists(): raise ValueError('Partial own combiner cache')
        record=fit_meta(config,spec,month,identity,history,records)
        if 'provenance' in record:
            for p,sha in record['provenance'].items():
                if digest(p)!=sha: raise ValueError('Reused combiner source changed')
        return record
    if mp.exists() or pp.exists(): raise ValueError('Partial own combiner cache')
    old=read_config(config['stack_config']);orp,omp=meta_paths(old,'validation',month,spec['id']);opp=paths(old,'validation',month,spec['id'])[1]
    present=[p.exists() for p in [orp,omp,opp]]
    if any(present) and not all(present): raise ValueError('Partial preserved combiner cache')
    if all(present):
        raw=read_config(orp)
        available=all(Path(p).exists() for p in raw['source_hashes'])
        if available:
            for p,sha in raw['source_hashes'].items():
                if digest(p)!=sha: raise ValueError('Preserved combiner dependency changed')
            # Exact existing artifacts only: this read-only audit cannot fit or save old files.
            before={str(p):digest(p) for p in [orp,omp,opp]}
            old_records={(name,m):read_config(paths(old,'validation',m,name)[0]) for name,m in records
                         if paths(old,'validation',m,name)[0].exists()}
            raw=fit_meta(old,spec,month,config['stack_fingerprint'],history,old_records)
            if any(digest(p)!=sha for p,sha in before.items()): raise ValueError('Preserved combiner was modified')
            mp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(omp,mp)
            data=prediction(old,'validation',month,spec['id']);save_prediction(pp,data['row_ids'],data['labels'],data['scores'])
            record={**raw,'fingerprint':identity,'model_path':str(mp),'model_sha256':digest(mp),
                    'source_hashes':source_hashes(config,spec,raw['train_months']),'prediction_sha256':digest(pp),
                    'reused':True,'provenance':before}
            write_json(rp,record)
            return fit_meta(config,spec,month,identity,history,records)
    record=fit_meta(config,spec,month,identity,history,records);record['reused']=False;write_json(rp,record);return record


def prepare_sources(config,identity,frame,allow_missing=True):
    base=eng=None;results=[]
    for spec,month in base_pairs(config):
        if not allow_missing and not all(p.exists() for p in paths(config,'validation',month,spec['id'])):
            raise ValueError('Required own source missing at audit')
        if base is None:
            # Features are used only if a frozen recipe actually needs refitting.
            old=read_config(config['stack_config'])
            if not paths(config,'validation',month,spec['id'])[0].exists() and not paths(old,'validation',month,spec['id'])[0].exists():
                base=features(frame);eng=engineer(base)
        results.append(ensure_base(frame,base,eng,spec,month,'validation',config,identity))
    history,records=base_history(config)
    meta=[]
    for month in config['validation_months']:
        for spec in config['combiners']:
            if not allow_missing and not meta_paths(config,'validation',month,spec['id'])[0].exists():
                raise ValueError('Required own combiner missing at audit')
            meta.append(ensure_meta(config,spec,month,identity,history,records))
    return results,meta


def histories(config):
    h={n:{} for n in config['score_sources']};r={}
    for name in h:
        for month in config['validation_months']:
            h[name][month]=prediction(config,'validation',month,name)
            record=read_config(source_record(config,'validation',month,name));verify_source(record,month)
            if record['prediction_sha256']!=digest(paths(config,'validation',month,name)[1]): raise ValueError('Source prediction changed')
            aligned(h[name][month],h['baseline_balanced'].get(month,h[name][month]));r[(name,month)]=record
    return h,r


def cal_paths(config,name,kind,evaluation):
    stem=f'm{evaluation}-{name}-{kind}';out=Path(config['output_dir'])/'calibration'
    return out/(stem+'.json'),out/(stem+'.joblib')


def fit_calibration(config,identity,history,records,name,kind,evaluation,allow_fit=True):
    method,mode=kind.split('_');past=policy_history(history,records,name,evaluation,config['validation_months'])
    data,months=calibration_population(past,evaluation,mode)
    expected=dict(candidate=name,policy=kind,eval_month=evaluation,source_months=months,
                  train_rows=len(data['labels']),train_fraud=int(data['labels'].sum()),train_ids_sha256=ids_digest(data['row_ids']),
                  fingerprint=identity,source_hashes={str(p):digest(p) for m in months for p in
                  [source_record(config,'validation',m,name),paths(config,'validation',m,name)[1]]})
    rp,mp=cal_paths(config,name,kind,evaluation)
    if rp.exists():
        record=read_config(rp)
        if any(record[k]!=v for k,v in expected.items()): raise ValueError('Calibration provenance changed')
        if not mp.exists() or digest(mp)!=record['model_sha256']: raise ValueError('Calibration artifact changed')
        model=joblib.load(mp)
    else:
        if not allow_fit: raise ValueError('Calibration artifact missing at audit')
        if mp.exists(): raise ValueError('Partial calibration cache')
        model=fit_calibrator(data,method,config['clip'],config['seed'],config['threads'])
        mp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,mp)
        record={**expected,'created_at_utc':now(),'model_path':str(mp),'model_sha256':digest(mp),'warnings':model['warnings']}
        write_json(rp,record)
    if model['method']=='sigmoid':
        from .calibrated_policy_metrics import logit
        if not np.allclose(model['model'].named_steps['standardscaler'].mean_,logit(data['scores'],config['clip']).mean(axis=0),atol=1e-12):
            raise ValueError('Calibration scaler population differs')
    return model,record


def apply_rule(config,data,rule):
    if rule['kind'] in POLICIES: return decisions(data,rule),data['scores'],{}
    if rule['kind']!='batch': raise ValueError('Unknown rule')
    if digest(rule['model_path'])!=rule['model_sha256']: raise ValueError('Calibrator changed')
    p=calibrated(joblib.load(rule['model_path']),data['scores'])
    mask,info=batch_decisions(p,data['scores'],data['row_ids'])
    return mask,p,info


def crossfit(config,identity,allow_fit=True):
    h,r=histories(config);rows=[];bins=[]
    for month in config['validation_months'][1:]:
        for name in config['score_sources']:
            data=h[name][month];past=policy_history(h,r,name,month,config['validation_months'])
            for kind in POLICIES+CALIBRATION_POLICIES:
                if kind in POLICIES: rule=learn(past,kind,month)
                else:
                    _,rec=fit_calibration(config,identity,h,r,name,kind,month,allow_fit)
                    rule=dict(kind='batch',policy=kind,source_months=rec['source_months'],model_path=rec['model_path'],model_sha256=rec['model_sha256'])
                mask,p,info=apply_rule(config,data,rule)
                rows.append(dict(candidate=name,policy=kind,eval_month=month,source_months=','.join(map(str,rule['source_months'])),
                                 parameters=json.dumps(rule,sort_keys=True),k=int(mask.sum()),predicted_total_positives=info.get('predicted_total_positives',float(p.sum())),
                                 estimated_tp=info.get('estimated_tp',float(p[mask].sum())),surrogate=info.get('surrogate',None),**evaluate(data,mask,p)))
                for flavor,scores in [('raw',data['scores']),('calibrated',p)]:
                    bins.extend(dict(candidate=name,policy=kind,eval_month=month,score_type=flavor,**b) for b in reliability(data,scores))
    cf=pd.DataFrame(rows)
    if len(cf)!=42: raise ValueError('Incomplete declared grid')
    return cf,pd.DataFrame(bins)


def verify_table(actual,path):
    stored=pd.read_csv(path)
    if list(actual.columns)!=list(stored.columns) or actual.shape!=stored.shape: raise ValueError('Table shape changed')
    for key in actual:
        if pd.api.types.is_numeric_dtype(actual[key]):
            if not np.allclose(actual[key],stored[key],atol=1e-12,rtol=1e-10,equal_nan=True): raise ValueError('Metric changed: '+key)
        elif not actual[key].fillna('').astype(str).equals(stored[key].fillna('').astype(str)):
            raise ValueError('Provenance changed: '+key)


def validate(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists(): raise ValueError('Frozen study cannot be retuned')
    frame=load_data(c);frame=frame[frame.month<7]
    bases,metas=prepare_sources(c,identity,frame)
    cf,bins=crossfit(c,identity);summary=order_summary(cf)
    for name,value in [('baseline_balanced',.32780157809618937),('baseline_lgbm',.3226536849)]:
        actual=summary[(summary.candidate==name)&(summary.policy=='pooled')].iloc[0].f2
        if not np.isclose(actual,value,atol=1e-10,rtol=0): raise ValueError('Prior F2 baseline mismatch')
    for label,table in [('crossfit',cf),('summary',summary),('reliability',bins)]:table.to_csv(out/(label+'.csv'),index=False)
    write_json(out/'validation_complete.json',dict(fingerprint=identity,created_at_utc=now(),
        files={name:digest(out/(name+'.csv')) for name in ['crossfit','summary','reliability']},
        base_results=len(bases),base_fits=sum(not x.get('reused',False) for x in bases),
        base_reuses=sum(x.get('reused',False) for x in bases),meta_results=len(metas),meta_fits=sum(not x.get('reused',False) for x in metas),
        meta_reuses=sum(x.get('reused',False) for x in metas),score_sources=9,calibration_fits=24,forward_rows=42))
    guard(c);print(summary.head(12).to_string(index=False),flush=True)


def freeze(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists(): raise ValueError('Selection is immutable')
    complete=read_config(out/'validation_complete.json')
    if complete['fingerprint']!=identity or any(digest(out/(n+'.csv'))!=sha for n,sha in complete['files'].items()):
        raise ValueError('Stale development outputs')
    frame=load_data(c);frame=frame[frame.month<7];prepare_sources(c,identity,frame,False)
    cf,bins=crossfit(c,identity,False);verify_table(cf,out/'crossfit.csv');verify_table(bins,out/'reliability.csv')
    summary=order_summary(cf);verify_table(summary.reset_index(drop=True),out/'summary.csv');winner=summary.iloc[0]
    name=str(winner.candidate);kind=str(winner.policy);h,r=histories(c)
    choices=dict(selected=(name,kind),balanced=('baseline_balanced','pooled'),unweighted=('baseline_lgbm','pooled'),same_model=(name,'pooled'))
    rules={}
    for alias,(n,k) in choices.items():
        if k in POLICIES:rules[alias]=learn(policy_history(h,r,n,7,c['validation_months']),k,7)
        else:
            _,record=fit_calibration(c,identity,h,r,n,k,7)
            rules[alias]=dict(kind='batch',policy=k,source_months=record['source_months'],model_path=record['model_path'],model_sha256=record['model_sha256'])
    spec=next((s for s in c['combiners'] if s['id']==name),None);final_meta=None
    if spec:
        bh,br=base_history(c);final_meta=fit_meta(c,spec,7,identity,bh,br,'frozen')
    needed={'baseline_balanced','baseline_lgbm'}|set(spec['members'] if spec else [])
    frozen=dict(created_at_utc=now(),fingerprint=identity,config=c,selected_candidate=name,selected_policy=kind,
        selected_development_f2=float(winner.f2),choices={n:list(v) for n,v in choices.items()},rules=rules,final_meta=final_meta,
        fit_specs=[s for s in c['candidates'] if s['id'] in needed],validation_files=complete['files'],
        development_artifacts={str(p):digest(p) for folder in ['runs','predictions','models','meta','calibration']
            for p in (out/folder).glob('*') if p.is_file()},month7_previously_consumed=True,exploratory=True)
    write_json(out/'selection.json',frozen);print('Frozen',name,kind,float(winner.f2),flush=True)


def load_selection(config_path=CONFIG):
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['config']!=c or s['fingerprint']!=fingerprint(config_path): raise ValueError('Frozen identity changed')
    for p,sha in s['development_artifacts'].items():
        if not Path(p).exists() or digest(p)!=sha: raise ValueError('Frozen artifact changed: '+p)
    for name,sha in s['validation_files'].items():
        if digest(out/(name+'.csv'))!=sha: raise ValueError('Frozen validation table changed')
    return c,s


def diagnostics(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);h,_=histories(c);sp=out/'selection.json';sha=digest(sp)
    rp=out/'diagnostic_provenance.json'
    if not rp.exists(): write_json(rp,dict(created_at_utc=now(),selection_sha256=sha,excluded_from_selection=True,development_only=True))
    rec=read_config(rp)
    if rec['selection_sha256']!=sha or not s['created_at_utc']<rec['created_at_utc']: raise ValueError('Diagnostic boundary violated')
    table=pd.DataFrame([dict(candidate=n,eval_month=m,**hindsight_prefix(h[n][m])) for n in c['score_sources'] for m in [5,6]])
    if (out/'hindsight.csv').exists(): verify_table(table,out/'hindsight.csv')
    else:table.to_csv(out/'hindsight.csv',index=False)
    if digest(sp)!=sha: raise ValueError('Diagnostic modified selection')
    print(table.to_string(index=False),flush=True)


def final_data(config,selected):
    items={s['id']:prediction(config,'exploratory',7,s['id']) for s in selected['fit_specs']}
    if selected['final_meta']:
        record=selected['final_meta'];name=record['spec']['id']
        items[name]=predict_meta(config,record,joblib.load(record['model_path']),{n:items[n] for n in record['spec']['members']})
    return {alias:items[name] for alias,(name,kind) in selected['choices'].items()}


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);ap=out/'exploratory_access.json'
    if not ap.exists(): write_json(ap,dict(access_at_utc=now(),selection_sha256=digest(out/'selection.json'),month7_previously_consumed=True))
    access=read_config(ap)
    if access['selection_sha256']!=digest(out/'selection.json') or not s['created_at_utc']<access['access_at_utc']:
        raise ValueError('Selection/access mismatch')
    frame=load_data(c);base=features(frame);eng=engineer(base);records=[]
    for spec in s['fit_specs']:records.append(ensure_base(frame,base,eng,spec,7,'exploratory',c,s['fingerprint']))
    data=final_data(c,s);rows=[]
    for alias,d in data.items():
        aligned(d,data['balanced']);mask,p,info=apply_rule(c,d,s['rules'][alias])
        rows.append(dict(role=alias,candidate=s['choices'][alias][0],policy=s['choices'][alias][1],eval_month=7,
                         parameters=json.dumps(s['rules'][alias],sort_keys=True),**{**info,'k':int(mask.sum())},**evaluate(d,mask,p)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    write_json(out/'transfer_complete.json',dict(selection_sha256=digest(out/'selection.json'),table_sha256=digest(out/'exploratory.csv'),
                base_results=len(records),base_fits=sum(not r.get('reused',False) for r in records),
                base_reuses=sum(r.get('reused',False) for r in records),final_combiner_fits=int(s['final_meta'] is not None),
                final_calibration_fits=int(s['rules']['selected']['kind']=='batch')))
    guard(c);print(table[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','freeze','diagnostics','exploratory']);p.add_argument('--config',default=CONFIG)
    a=p.parse_args();globals()[a.command](a.config)
