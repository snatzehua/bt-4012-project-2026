"""Fixed score-fusion study with verified frozen recipes and chronology."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import shutil
import warnings
from threadpoolctl import threadpool_limits
from .common import digest,features,load_data,read_config,write_json
from .engineering import NativeInputs,make_estimator
from .improvements import paths,prediction
from .focal_study import expected_record,now
from .recall_weighted_metrics import POLICIES,aligned,learn,measure
from .recall_extension import expected_population,order_summary
from .calibrated_policy import verify_table
from .improvement_metrics import confusion_metrics
from . import xgboost_ensemble as old
from .score_fusion import strategies,compose,choices_for,clipping,EPSILON

CONFIG='configs/xgboost-score-fusion.json'
SOURCES=old.SOURCES+['baf/score_fusion.py','baf/score_fusion_study.py','baf/score_fusion_report.py']


def fingerprint(path):
    result=old.fingerprint(path);result['sources']={p:digest(p) for p in SOURCES}
    result['software']['xgboost']=version('xgboost');return result


def declared_strategies(c):return strategies([s['id'] for s in c['presets']],[s['id'] for s in c['references']])


def guard(c):
    source=read_config(c['source_config']);old.guard(source)
    if c['source_config']!=old.CONFIG or old.fingerprint(c['source_config'])!=c['source_fingerprint']:
        raise ValueError('Ninth study identity changed')
    protected={**source['preserve_sha256'],c['source_config']:digest(c['source_config'])}
    for k in ['output_dir','report_dir']:
        p=str(Path(source[k])/'selection.json');protected[p]=digest(p)
    if c['preserve_sha256']!=protected:raise ValueError('Preservation contract changed')
    for p,h in protected.items():
        if digest(p)!=h:raise ValueError('Preserved artifact changed: '+p)
    for key in ['presets','tree_params','references','reference_source_config','validation_months','test_month',
                'beta','seed','threads','policies','bootstrap_seed','bootstrap_replicates','data_path','data_sha256']:
        if c[key]!=source[key]:raise ValueError('Frozen recipe changed: '+key)
    if c['epsilon']!=EPSILON or c['strategies']!=declared_strategies(c):raise ValueError('Declared fusion grid changed')
    if any(k in c for k in ['primary_fraction','precision_floor','review_quota']):raise ValueError('Undeclared constraint')
    roots=sorted({str(Path(p).parent) for p in protected if p.endswith('/selection.json')})
    a,b=[Path(c[k]).resolve() for k in ['output_dir','report_dir']]
    for dest in [a,b]:
        for root in roots:
            origin=Path(root).resolve()
            if dest==origin or dest in origin.parents or origin in dest.parents:raise ValueError('Destination overlaps preserved study')
    if a==b or a in b.parents or b in a.parents:raise ValueError('Bulk/report destinations overlap')


def reference_config(c):
    return {**read_config(c['reference_source_config']),'output_dir':c['output_dir']}


def reference_model_path(c,phase,month,name):
    return Path(c['output_dir'])/'models'/f'{phase}-m{month}-{name}-reference.joblib'


def parameter_signature(model):
    return {k:repr(v) for k,v in sorted(model.get_params(deep=True).items())}


def verify_origin(record):
    for p,h in record.get('origin_artifacts',{}).items():
        if not Path(p).exists() or digest(p)!=h:raise ValueError('Reuse origin changed: '+p)


def reference_inference(c,frame,base,spec,month,phase,record):
    mp=reference_model_path(c,phase,month,spec['id'])
    if not mp.exists() or digest(mp)!=record['reference_model_sha256']:raise ValueError('Reference model changed')
    artifact=joblib.load(mp);train=frame.index[frame.month<month];data=prediction(c,phase,month,spec['id'])
    factory,expected_adapter=make_estimator(spec,reference_config(c),base.loc[train])
    if artifact['spec']!=spec or parameter_signature(artifact['model'])!=parameter_signature(factory):raise ValueError('Reference parameters changed')
    adapter=artifact['adapter']
    if (adapter is None)!=(expected_adapter is None):raise ValueError('Reference adapter kind changed')
    if adapter is not None:
        if adapter.columns!=expected_adapter.columns or adapter.levels!=expected_adapter.levels or not adapter.medians.equals(expected_adapter.medians):
            raise ValueError('Reference adapter training changed')
    x=base.loc[data['row_ids']];x=adapter.transform(x) if adapter is not None else x
    if list(artifact['model'].classes_)!=[0,1]:raise ValueError('Reference class orientation changed')
    with threadpool_limits(limits=c['threads']):actual=artifact['model'].predict_proba(x)[:,1]
    if not np.allclose(actual,data['scores'],atol=1e-12,rtol=1e-12):raise ValueError('Reference saved-model scores differ')
    verify_origin(record)
    return record


def ensure_reference_model(c,identity,frame,base,spec,month,phase,record,allow_missing):
    mp=reference_model_path(c,phase,month,spec['id']);rp,_=paths(c,phase,month,spec['id'])
    if 'reference_model_sha256' in record:
        return reference_inference(c,frame,base,spec,month,phase,record)
    if mp.exists():raise ValueError('Partial own reference model cache')
    if not allow_missing:raise ValueError('Reference model missing at audit')
    # Older development caches contain predictions, but no saved model contract.
    # Reproduce that wholly absent model separately; never silently replace scores.
    train=frame.index[frame.month<month];data=prediction(c,phase,month,spec['id'])
    factory,adapter=make_estimator(spec,reference_config(c),base.loc[train]);source_model=None
    if phase=='exploratory':
        prior=read_config(read_config(c['reference_source_config'])['prior_config'])
        candidate=Path(prior['output_dir'])/'models'/(spec['id']+'.joblib')
        if candidate.exists():source_model=candidate
    start=time.perf_counter();messages=[]
    if source_model is not None:
        artifact=joblib.load(source_model)
        if artifact['spec']!=spec or parameter_signature(artifact['model'])!=parameter_signature(factory):raise ValueError('Frozen final reference recipe changed')
    else:
        x=adapter.transform(base.loc[train]) if adapter is not None else base.loc[train]
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=c['threads']):
            warnings.simplefilter('always');factory.fit(x,frame.loc[train,'fraud_bool'].to_numpy())
        messages=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if messages:raise ValueError('Reference verification fit warnings: '+str(messages))
        artifact=dict(model=factory,adapter=adapter,spec=spec)
    x=base.loc[data['row_ids']];x=artifact['adapter'].transform(x) if artifact['adapter'] is not None else x
    with threadpool_limits(limits=c['threads']):actual=artifact['model'].predict_proba(x)[:,1]
    if not np.allclose(actual,data['scores'],atol=1e-12,rtol=1e-12):raise ValueError('Exact reference reproduction differs from frozen prediction')
    mp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,mp)
    r={**record,'reference_model_path':str(mp),'reference_model_sha256':digest(mp),
       'reference_model_new_fit':source_model is None,'reference_model_verification_seconds':time.perf_counter()-start,
       'reference_model_warnings':messages}
    if source_model is not None:r['origin_artifacts']={str(source_model):digest(source_model)}
    write_json(rp,r)
    return reference_inference(c,frame,base,spec,month,phase,r)


def prepare_references(c,identity,frame,base,phase='validation',specs=None,allow_missing=True):
    specs=c['references'] if specs is None else specs
    ref=reference_config(c)
    for month in c['validation_months'] if phase=='validation' else [7]:
        for spec in specs:
            rp,pp=paths(c,phase,month,spec['id'])
            if rp.exists()!=pp.exists():raise ValueError('Partial own reference prediction cache')
            possible=[read_config(ref[k]) for k in ['stack_config','previous_f2_config','prior_config']]
            available=False
            for prior in possible:
                sr,sp=paths(prior,phase,month,spec['id'])
                if sr.exists()!=sp.exists():raise ValueError('Partial preserved reference prediction cache')
                available|=sr.exists()
            if not rp.exists() and not available and allow_missing:
                # Fresh clone: fit once, saving both the prediction and model.
                train=frame.index[frame.month<month];ev=frame.index[frame.month==month]
                model,adapter=make_estimator(spec,ref,base.loc[train]);x=base.loc[train]
                if adapter is not None:x=adapter.transform(x)
                start=time.perf_counter()
                with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=c['threads']):
                    warnings.simplefilter('always');model.fit(x,frame.loc[train,'fraud_bool'].to_numpy())
                    z=base.loc[ev];z=adapter.transform(z) if adapter is not None else z;p=model.predict_proba(z)[:,1]
                messages=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
                if messages:raise ValueError('Reference fit warnings: '+str(messages))
                mp=reference_model_path(c,phase,month,spec['id'])
                if mp.exists():raise ValueError('Partial own reference model cache')
                mp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(dict(model=model,adapter=adapter,spec=spec),mp)
                d=dict(row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=p)
                pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**d)
                r={**expected_record(frame,spec,month,phase,identity),**measure(d,dict(kind='pooled',threshold=.5)),
                   'family':spec['family'],'feature_count':base.shape[1],'fit_seconds':time.perf_counter()-start,'warnings':messages,
                   'reused':False,'prediction_sha256':digest(pp),'reference_model_path':str(mp),'reference_model_sha256':digest(mp),
                   'reference_model_new_fit':True,'reference_model_verification_seconds':0,'reference_model_warnings':[]}
                write_json(rp,r)
    records=old.prepare_references(c,identity,frame,base,phase,specs,allow_missing)
    return [ensure_reference_model(c,identity,frame,base,next(s for s in specs if s['id']==r['candidate']),
            r['eval_month'],phase,r,allow_missing) for r in records]


def model_paths(c,phase,month,name):
    stem=f'{phase}-m{month}-{name}';out=Path(c['output_dir'])/'models'
    return out/(stem+'.ubj'),out/(stem+'.joblib')


def audit_model(c,identity,frame,base,spec,month,phase='validation',adapter=None):
    r=old.audit_model(c,identity,frame,base,spec,month,phase,adapter);verify_origin(r);return r


def cache_state(paths_):
    present=[p.exists() for p in paths_]
    if any(present) and not all(present):raise ValueError('Partial model cache')
    return all(present)


def fit_model(c,identity,frame,base,spec,month,phase='validation',allow_missing=True):
    rp,pp=paths(c,phase,month,spec['id']);mp,ap=model_paths(c,phase,month,spec['id'])
    if cache_state([rp,pp,mp,ap]):return audit_model(c,identity,frame,base,spec,month,phase)
    if not allow_missing:raise ValueError('Model cache missing at audit')
    source=read_config(c['source_config']);sr,sp=paths(source,phase,month,spec['id']);sm,sa=model_paths(source,phase,month,spec['id'])
    if cache_state([sr,sp,sm,sa]):
        raw=old.audit_model(source,c['source_fingerprint'],frame,base,spec,month,phase)
        for src,dst in [(sp,pp),(sm,mp),(sa,ap)]:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        record={**raw,'fingerprint':identity,'reused':True,'model_path':str(mp),
                'origin_artifacts':{str(p):digest(p) for p in [sr,sp,sm,sa]}}
        write_json(rp,record)
        return audit_model(c,identity,frame,base,spec,month,phase)
    record=old.fit_model(c,identity,frame,base,spec,month,phase)
    return record


def ensure_score(c,identity,spec,month,phase='validation',allow_missing=True):
    if spec['kind']=='raw':return read_config(paths(c,phase,month,spec['id'])[0])
    members={n:prediction(c,phase,month,n) for n in spec['members']};d=compose(spec,members,c['epsilon'])
    dependencies={n:{'record':digest(paths(c,phase,month,n)[0]),'prediction':digest(paths(c,phase,month,n)[1])} for n in spec['members']}
    rp,pp=paths(c,phase,month,spec['id']);present=[rp.exists(),pp.exists()]
    expected=dict(candidate=spec['id'],phase=phase,eval_month=month,spec=spec,fingerprint=identity,
                  dependencies=dependencies,transform='fixed row-wise raw/logit mean; v1',epsilon=c['epsilon'])
    if any(present):
        if not all(present):raise ValueError('Partial composed-score cache')
        r=read_config(rp)
        if any(r[k]!=v for k,v in expected.items()) or digest(pp)!=r['prediction_sha256']:raise ValueError('Composed-score identity changed')
        actual=prediction(c,phase,month,spec['id']);aligned(actual,d)
        if not np.array_equal(actual['scores'],d['scores']):raise ValueError('Composed-score reconstruction differs')
        return r
    if not allow_missing:raise ValueError('Composed-score cache missing')
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**d)
    r={**expected,'prediction_sha256':digest(pp)};write_json(rp,r);return r


def history(c):return {s['id']:{m:prediction(c,'validation',m,s['id']) for m in c['validation_months']} for s in c['strategies']}


def crossfit(c):
    h=history(c);rows=[]
    for month in [5,6]:
        for name,items in h.items():
            aligned(items[month],h['baseline_balanced'][month]);past={m:d for m,d in items.items() if m<month}
            for kind in POLICIES:
                rule=learn(past,kind,month)
                rows.append(dict(candidate=name,policy=kind,eval_month=month,source_months=','.join(map(str,rule['source_months'])),
                                 parameters=json.dumps(rule,sort_keys=True),**measure(items[month],rule)))
    if len(rows)!=366:raise ValueError('Incomplete policy grid')
    return pd.DataFrame(rows)


def export_comparisons(c,cf):
    out=Path(c['output_dir']);summary=order_summary(cf);lookup=summary.set_index(['candidate','policy']);rows=[]
    def add(label,name,ref,kind,refkind=None):
        a=lookup.loc[(name,kind)];b=lookup.loc[(ref,refkind or kind)]
        rows.append(dict(contrast=label,candidate=name,policy=kind,reference=ref,reference_policy=refkind or kind,
            **{m+'_difference':float(a[m]-b[m]) for m in ['f2','precision','recall','flagged_fraction','average_precision']}))
    for s in c['strategies']:
        for kind in POLICIES:
            for ref in ['baseline_balanced','baseline_lgbm']:add('old pooled reference',s['id'],ref,kind,'pooled')
            if kind!='pooled':add('policy minus own pooled',s['id'],s['id'],kind,'pooled')
            if s['kind']=='logit':add('logit minus raw',s['id'],s['members'][0],kind)
            if s['kind'] in ['raw_mean','logit_mean']:
                prefix='logit_' if s['kind']=='logit_mean' else ''
                for n in s['members']:add('fusion minus matched member',s['id'],prefix+n,kind)
                if prefix:add('logit fusion minus raw fusion',s['id'],s['id'].replace('logit_mean_','raw_mean_',1),kind)
    if len(rows)!=866:raise ValueError('Incomplete contrasts')
    pd.DataFrame(rows).to_csv(out/'contrasts.csv',index=False);pooled=[]
    for (name,kind),group in cf.groupby(['candidate','policy']):
        t=group[['tn','fp','fn','tp']].sum();metrics=confusion_metrics(*[t[k] for k in ['tn','fp','fn','tp']])
        pooled.append(dict(candidate=name,policy=kind,**metrics,flagged_fraction=(t.tp+t.fp)/group.n.sum()))
    pd.DataFrame(pooled).to_csv(out/'pooled.csv',index=False);h=history(c);diagnostics=[];clips=[]
    for name,items in h.items():
        for month,d in items.items():
            raw=d['scores'];diagnostics.append(dict(candidate=name,eval_month=month,n=len(raw),score_min=float(raw.min()),score_max=float(raw.max()),
                score_mean=float(raw.mean()),unique_scores=len(np.unique(raw)),tied_fraction=1-len(np.unique(raw))/len(raw)))
    for name in [s['members'][0] for s in c['strategies'] if s['kind']=='logit']:
        for month,d in h[name].items():clips.append(dict(candidate=name,eval_month=month,**clipping(d['scores'],c['epsilon'])))
    pd.DataFrame(diagnostics).to_csv(out/'score_diagnostics.csv',index=False);pd.DataFrame(clips).to_csv(out/'clipping.csv',index=False)
    return summary


def audit_grid(c,identity,frame,base):
    rows=prepare_references(c,identity,frame,base,allow_missing=False)
    for month in c['validation_months']:
        a=NativeInputs().fit(base.loc[frame.index[frame.month<month]])
        for spec in c['presets']:rows.append(audit_model(c,identity,frame,base,spec,month,adapter=a))
        for spec in c['strategies']:ensure_score(c,identity,spec,month,allow_missing=False)
    if len(rows)!=21:raise ValueError('Incomplete base audit')
    return rows


def validate(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():raise ValueError('Frozen study cannot be retuned')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);refs=prepare_references(c,identity,frame,base)
    models=[]
    for month in c['validation_months']:
        for spec in c['presets']:models.append(fit_model(c,identity,frame,base,spec,month))
        for spec in c['strategies']:ensure_score(c,identity,spec,month)
    cf=crossfit(c);summary=export_comparisons(c,cf)
    for name,value in [('baseline_balanced',.32780157809618937),('baseline_lgbm',.3226536849212962)]:
        if not np.isclose(summary[(summary.candidate==name)&(summary.policy=='pooled')].iloc[0].f2,value,atol=1e-12,rtol=0):raise ValueError('Old pooled benchmark mismatch')
    cf.to_csv(out/'crossfit.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
    files=['crossfit.csv','summary.csv','contrasts.csv','pooled.csv','score_diagnostics.csv','clipping.csv']
    write_json(out/'validation_complete.json',dict(fingerprint=identity,created_at_utc=now(),files={p:digest(out/p) for p in files},
        new_xgboost_fits=sum(not r.get('reused',False) for r in models),xgboost_reuses=sum(r.get('reused',False) for r in models),
        reference_prediction_reuses=sum(r.get('reused',False) for r in refs),reference_prediction_refits=sum(not r.get('reused',False) for r in refs),
        reference_model_verification_fits=sum(r['reference_model_new_fit'] and r.get('reused',False) for r in refs),reference_results=len(refs),
        miniature_integration_fits=0,forward_rows=366,score_results=183,base_results=21))
    guard(c);print(summary.head(12).to_string(index=False),flush=True)


def freeze(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    complete=read_config(out/'validation_complete.json')
    if complete['fingerprint']!=identity or any(digest(out/p)!=sha for p,sha in complete['files'].items()):raise ValueError('Stale development outputs')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);audit_grid(c,identity,frame,base)
    cf=crossfit(c);verify_table(cf,out/'crossfit.csv');summary=order_summary(cf);verify_table(summary.reset_index(drop=True),out/'summary.csv')
    winner=summary.iloc[0];name=str(winner.candidate);kind=str(winner.policy);h=history(c)
    spec=next(s for s in c['strategies'] if s['id']==name);choices=choices_for(spec,kind)
    rules={alias:learn(h[n],k,7) for alias,(n,k) in choices.items()};needed={n for n,k in choices.values()}
    selected_strategies=[s for s in c['strategies'] if s['id'] in needed];deps={n for s in selected_strategies for n in s['members']}
    frozen=dict(created_at_utc=now(),fingerprint=identity,config=c,selected_candidate=name,selected_policy=kind,selected_development_f2=float(winner.f2),
        choices=choices,rules=rules,score_specs=selected_strategies,fit_specs=[s for s in c['presets']+c['references'] if s['id'] in deps],
        validation_files=complete['files'],artifacts={str(p):digest(p) for folder in ['runs','predictions','models'] for p in (out/folder).glob('*') if p.is_file()},
        month7_previously_consumed=True,exploratory=True)
    write_json(out/'selection.json',frozen);print('Frozen',name,kind,float(winner.f2),flush=True)


def load_selection(config_path=CONFIG):
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['config']!=c or s['fingerprint']!=fingerprint(config_path):raise ValueError('Frozen study identity changed')
    for p,sha in s['artifacts'].items():
        if not Path(p).exists() or digest(p)!=sha:raise ValueError('Frozen score/model artifact changed: '+p)
    for p,sha in s['validation_files'].items():
        if digest(out/p)!=sha:raise ValueError('Frozen validation output changed')
    winner=next(x for x in c['strategies'] if x['id']==s['selected_candidate'])
    if s['choices']!=choices_for(winner,s['selected_policy']):raise ValueError('Frozen control choices differ')
    needed={n for n,k in s['choices'].values()};score_specs=[x for x in c['strategies'] if x['id'] in needed]
    deps={n for x in score_specs for n in x['members']}
    if s['score_specs']!=score_specs or s['fit_specs']!=[x for x in c['presets']+c['references'] if x['id'] in deps]:raise ValueError('Frozen dependency inclusion differs')
    table=order_summary(pd.read_csv(out/'crossfit.csv'));best=table.iloc[0]
    if [str(best.candidate),str(best.policy)]!=s['choices']['selected'] or not np.isclose(best.f2,s['selected_development_f2'],atol=1e-12,rtol=0):raise ValueError('Frozen selection differs from development')
    h=history(c)
    if any(s['rules'][a]!=learn(h[n],k,7) for a,(n,k) in s['choices'].items()):raise ValueError('Frozen earlier-only rule differs')
    return c,s


def final_data(c,s):return {alias:prediction(c,'exploratory',7,n) for alias,(n,k) in s['choices'].items()}


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);ap=out/'exploratory_access.json'
    if not ap.exists():write_json(ap,dict(access_at_utc=now(),selection_sha256=digest(out/'selection.json'),month7_previously_consumed=True))
    access=read_config(ap)
    if access['selection_sha256']!=digest(out/'selection.json') or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Freeze/access mismatch')
    frame=load_data(c);base=features(frame);refids={r['id'] for r in c['references']};records=[];refs=[r for r in s['fit_specs'] if r['id'] in refids]
    if refs:records+=prepare_references(c,s['fingerprint'],frame,base,'exploratory',refs)
    for spec in s['fit_specs']:
        if spec['id'] not in refids:records.append(fit_model(c,s['fingerprint'],frame,base,spec,7,'exploratory'))
    for spec in s['score_specs']:ensure_score(c,s['fingerprint'],spec,7,'exploratory')
    data=final_data(c,s);rows=[]
    for alias,d in data.items():
        aligned(d,data['balanced']);rule=s['rules'][alias]
        rows.append(dict(role=alias,candidate=s['choices'][alias][0],policy=s['choices'][alias][1],eval_month=7,parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    write_json(out/'transfer_complete.json',dict(selection_sha256=digest(out/'selection.json'),table_sha256=digest(out/'exploratory.csv'),base_results=len(records),
        new_model_fits=sum(r['candidate'] not in refids and not r.get('reused',False) for r in records),reference_fits=sum(r['candidate'] in refids and not r.get('reused',False) for r in records),
        reference_reuses=sum(r['candidate'] in refids and r.get('reused',False) for r in records),
        reference_model_verification_fits=sum(r.get('reference_model_new_fit',False) and r.get('reused',False) for r in records)))
    guard(c);print(table[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','freeze','exploratory']);p.add_argument('--config',default=CONFIG)
    a=p.parse_args();globals()[a.command](a.config)
