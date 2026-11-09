"""Finite chronological categorical models and rank ensembles with frozen identities."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .common import digest,features,load_data,read_config,write_json
from .engineering import NativeInputs
from .improvements import paths,prediction
from .focal_study import expected_record,now
from .recall_weighted_metrics import POLICIES,aligned,learn,measure
from .recall_extension import expected_population,order_summary
from .calibrated_policy import verify_table
from .improvement_metrics import confusion_metrics
from . import direct_ranking as old
from .xgboost_inputs import presets,tree_parameters,parameters,fit_checked,scores,load_booster,integration_gate
from .rank_ensemble_scores import strategies,compose,choices_for

CONFIG='configs/xgboost-rank-ensembles.json'
SOURCES=old.SOURCES+['baf/xgboost_inputs.py','baf/rank_ensemble_scores.py','baf/xgboost_ensemble.py',
    'baf/xgboost_ensemble_report.py','requirements.txt','requirements-improvements.txt','requirements-xgboost.txt']


def fingerprint(path):
    result=old.fingerprint(path);result['sources']={p:digest(p) for p in SOURCES}
    result['software']['xgboost']=version('xgboost');return result


def declared_strategies(c):return strategies([s['id'] for s in c['presets']],[s['id'] for s in c['references']])


def guard(c):
    source=read_config(c['source_config']);old.guard(source)
    if old.fingerprint(c['source_config'])!=c['source_fingerprint']:raise ValueError('Eighth study identity changed')
    for p,h in c['preserve_sha256'].items():
        if digest(p)!=h:raise ValueError('Preserved artifact changed: '+p)
    if version('xgboost')!='3.4.1':raise ValueError('Pinned XGBoost version changed')
    if c['presets']!=presets() or c['tree_params']!=tree_parameters() or c['strategies']!=declared_strategies(c):raise ValueError('Declared model/score grid changed')
    if c['references']!=source['references'] or c['reference_source_config']!=source['reference_source_config']:raise ValueError('Reference recipe changed')
    for k,v in dict(validation_months=[4,5,6],test_month=7,beta=2,seed=2026,threads=6,policies=POLICIES,bootstrap_seed=1027,bootstrap_replicates=500).items():
        if c[k]!=v:raise ValueError('Declared study setting changed: '+k)
    if any(c[k]!=source[k] for k in ['data_path','data_sha256']):raise ValueError('Data identity changed')
    roots=[read_config(c['reference_source_config'])[k] for k in ['stack_config','extension_config','previous_f2_config','prior_config','original_config']]
    roots+=[c['source_config'],source['source_config'],c['reference_source_config']]
    for key in ['output_dir','report_dir']:
        dest=Path(c[key]).resolve()
        for path in roots:
            prior=read_config(path);origin=Path(prior.get(key,prior['output_dir'].replace('outputs/','reports/',1))).resolve()
            if dest==origin or dest in origin.parents or origin in dest.parents:raise ValueError('Destination overlaps preserved study')
    if Path(c['output_dir']).resolve()==Path(c['report_dir']).resolve():raise ValueError('Report and bulk destination coincide')


def prepare_references(c,identity,frame,base,phase='validation',specs=None,allow_missing=True):
    return old.prepare_references(c,identity,frame,base,phase,specs,allow_missing)


def model_paths(c,phase,month,name):
    stem=f'{phase}-m{month}-{name}';out=Path(c['output_dir'])/'models'
    return out/(stem+'.ubj'),out/(stem+'.joblib')


def audit_model(c,identity,frame,base,spec,month,phase='validation',adapter=None):
    rp,pp=paths(c,phase,month,spec['id']);mp,ap=model_paths(c,phase,month,spec['id']);r=read_config(rp)
    expected={**expected_record(frame,spec,month,phase,identity),'fit_params':parameters(spec,c)}
    if any(r[k]!=v for k,v in expected.items()):raise ValueError('XGBoost training/identity changed')
    for p,k in [(pp,'prediction_sha256'),(mp,'model_sha256'),(ap,'adapter_sha256')]:
        if not p.exists() or digest(p)!=r[k]:raise ValueError('XGBoost artifact changed: '+str(p))
    if r['warnings']:raise ValueError('Fit warnings need inspection')
    d=prediction(c,phase,month,spec['id']);aligned(d,expected_population(frame,month))
    artifact=joblib.load(ap);a=artifact['adapter'];train=frame.index[frame.month<month]
    expected_adapter=NativeInputs().fit(base.loc[train]) if adapter is None else adapter
    if artifact['spec']!=spec or artifact['fit_params']!=parameters(spec,c):raise ValueError('Saved model recipe changed')
    if a.columns!=expected_adapter.columns or a.levels!=expected_adapter.levels or not a.medians.equals(expected_adapter.medians):raise ValueError('Training-fitted adapter changed')
    b=load_booster(mp,c['threads']);actual=scores(b,a.transform(base.loc[d['row_ids']]))
    if not np.allclose(actual,d['scores'],atol=1e-12,rtol=1e-12):raise ValueError('Saved categorical model prediction differs')
    if b.feature_names!=a.columns or b.feature_types!=r['feature_types'] or r['feature_names']!=a.columns:raise ValueError('Feature identity changed')
    if b.num_boosted_rounds()!=r['fitted_rounds'] or r['fitted_rounds']!=c['tree_params']['n_estimators']:raise ValueError('Fitted rounds changed')
    return r


def fit_model(c,identity,frame,base,spec,month,phase='validation'):
    rp,pp=paths(c,phase,month,spec['id']);mp,ap=model_paths(c,phase,month,spec['id']);present=[p.exists() for p in [rp,pp,mp,ap]]
    if any(present):
        if not all(present):raise ValueError('Partial XGBoost cache')
        return audit_model(c,identity,frame,base,spec,month,phase)
    train=frame.index[frame.month<month];ev=frame.index[frame.month==month];a=NativeInputs().fit(base.loc[train]);start=time.perf_counter()
    print('Fitting',rp.stem,f'rows={len(train):,}',flush=True)
    model,messages=fit_checked(a.transform(base.loc[train]),frame.loc[train,'fraud_bool'].to_numpy(),spec,c)
    b=model.get_booster();p=scores(b,a.transform(base.loc[ev]))
    mp.parent.mkdir(parents=True,exist_ok=True);model.save_model(mp);joblib.dump(dict(adapter=a,spec=spec,fit_params=parameters(spec,c)),ap)
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=p)
    record={**expected_record(frame,spec,month,phase,identity),'fit_params':parameters(spec,c),'model_path':str(mp),
        'model_sha256':digest(mp),'adapter_sha256':digest(ap),'prediction_sha256':digest(pp),'fit_seconds':time.perf_counter()-start,
        'warnings':messages,'feature_names':a.columns,'feature_types':b.feature_types,'fitted_rounds':b.num_boosted_rounds(),'reused':False}
    write_json(rp,record);print('Finished',rp.stem,f"seconds={record['fit_seconds']:.1f}",flush=True)
    return audit_model(c,identity,frame,base,spec,month,phase,a)


def ensure_score(c,identity,spec,month,phase='validation',allow_missing=True):
    if spec['kind']=='raw':return read_config(paths(c,phase,month,spec['id'])[0])
    members={n:prediction(c,phase,month,n) for n in spec['members']};d=compose(spec,members)
    dependencies={n:{'record':digest(paths(c,phase,month,n)[0]),'prediction':digest(paths(c,phase,month,n)[1])} for n in spec['members']}
    rp,pp=paths(c,phase,month,spec['id']);present=[rp.exists(),pp.exists()]
    expected=dict(candidate=spec['id'],phase=phase,eval_month=month,spec=spec,fingerprint=identity,
                  dependencies=dependencies,transform='average ascending rank/n; v1; current unlabeled batch')
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
    if len(rows)!=150:raise ValueError('Incomplete policy grid')
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
            if s['kind']=='percentile':add('percentile minus raw',s['id'],s['members'][0],kind)
            if s['kind']=='blend':
                for n in s['members']:add('blend minus percentile member',s['id'],'pct_'+n,kind)
    for spec in c['presets']:
        for kind in POLICIES:
            if spec['max_depth']==6:add('depth 6 minus 4',spec['id'],f"xgb_d4_w{spec['scale_pos_weight']}",kind)
            if spec['scale_pos_weight']==5:add('weight 5 minus 1',spec['id'],f"xgb_d{spec['max_depth']}_w1",kind)
    if len(rows)!=302:raise ValueError('Incomplete contrasts')
    pd.DataFrame(rows).to_csv(out/'contrasts.csv',index=False);pooled=[]
    for (name,kind),group in cf.groupby(['candidate','policy']):
        t=group[['tn','fp','fn','tp']].sum();metrics=confusion_metrics(*[t[k] for k in ['tn','fp','fn','tp']])
        pooled.append(dict(candidate=name,policy=kind,**metrics,flagged_fraction=(t.tp+t.fp)/group.n.sum()))
    pd.DataFrame(pooled).to_csv(out/'pooled.csv',index=False);h=history(c);diagnostics=[];diversity=[]
    for name,items in h.items():
        for month,d in items.items():
            raw=d['scores'];diagnostics.append(dict(candidate=name,eval_month=month,n=len(raw),score_min=float(raw.min()),score_max=float(raw.max()),
                score_mean=float(raw.mean()),unique_scores=len(np.unique(raw)),tied_fraction=1-len(np.unique(raw))/len(raw)))
            if name in [s['id'] for s in c['presets']]:
                other=h['baseline_balanced'][month]['scores'];rho=float(spearmanr(raw,other).statistic)
                diversity.append(dict(candidate=name,eval_month=month,n=len(raw),spearman_with_balanced=rho,
                    tied_fraction=1-len(np.unique(raw))/len(raw),balanced_tied_fraction=1-len(np.unique(other))/len(other)))
    pd.DataFrame(diagnostics).to_csv(out/'score_diagnostics.csv',index=False);pd.DataFrame(diversity).to_csv(out/'diversity.csv',index=False)
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
    gate_path=out/'integration_gate.json'
    if gate_path.exists():
        gate=read_config(gate_path)
        if gate['fingerprint']!=identity or gate['miniature_fits']!=8:raise ValueError('Integration checkpoint identity changed')
    else:write_json(gate_path,dict(fingerprint=identity,rows=integration_gate(c),miniature_fits=8))
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);refs=prepare_references(c,identity,frame,base)
    for month in c['validation_months']:
        for spec in c['presets']:fit_model(c,identity,frame,base,spec,month)
        for spec in c['strategies']:ensure_score(c,identity,spec,month)
    cf=crossfit(c);summary=export_comparisons(c,cf)
    for name,value in [('baseline_balanced',.32780157809618937),('baseline_lgbm',.3226536849212962)]:
        if not np.isclose(summary[(summary.candidate==name)&(summary.policy=='pooled')].iloc[0].f2,value,atol=1e-12,rtol=0):raise ValueError('Old pooled benchmark mismatch')
    cf.to_csv(out/'crossfit.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
    files=['integration_gate.json','crossfit.csv','summary.csv','contrasts.csv','pooled.csv','score_diagnostics.csv','diversity.csv']
    write_json(out/'validation_complete.json',dict(fingerprint=identity,created_at_utc=now(),files={p:digest(out/p) for p in files},
        new_development_fits=12,reference_results=9,reference_fits=sum(not r.get('reused',False) for r in refs),
        reference_reuses=sum(r.get('reused',False) for r in refs),miniature_integration_fits=8,forward_rows=150,score_results=75))
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
        new_model_fits=sum(r['candidate'] not in refids for r in records),reference_fits=sum(r['candidate'] in refids and not r.get('reused',False) for r in records),
        reference_reuses=sum(r.get('reused',False) for r in records)))
    guard(c);print(table[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','freeze','exploratory']);p.add_argument('--config',default=CONFIG)
    a=p.parse_args();globals()[a.command](a.config)
