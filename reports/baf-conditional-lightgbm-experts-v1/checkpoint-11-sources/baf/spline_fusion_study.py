"""Preserved Newton/tree score reuse and chronological fixed F2 composition."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import warnings
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import digest,features,load_data,read_config,write_json
from .improvements import paths,prediction
from .focal_study import expected_record,now
from .recall_extension import expected_population,order_summary
from .recall_weighted_metrics import POLICIES,aligned,learn,measure,decisions
from .improvement_metrics import confusion_metrics
from .calibrated_policy import verify_table
from .engineering import make_estimator
from . import newton_spline_study as newton
from . import newton_spline_audit as ancestor
from . import score_fusion_study as tree
from .spline_fusion_scores import (FUSION,REFERENCES,MEMBERS,ALPHAS,EPSILON,
    strategies,compose,choices_for,contrast_factors,diversity,fixtures,clipping)

CONFIG='configs/spline-score-fusion.json'
SOURCES=ancestor.ANALYSIS_SOURCES+['baf/spline_fusion_scores.py','baf/spline_fusion_study.py','baf/spline_fusion_report.py']
TABLES=['crossfit.csv','summary.csv','pooled.csv','contrasts.csv','score_diagnostics.csv',
        'clipping.csv','diversity.csv','joint_errors.csv','preservation.json','declaration.json','fixtures.json']


def fingerprint(path=CONFIG):
    fp=newton.fingerprint(path);fp['sources']={p:digest(p) for p in SOURCES};return fp


def verify_hashes(hashes):
    for p,h in hashes.items():
        if not Path(p).is_file() or digest(p)!=h:raise ValueError('Preserved artifact changed: '+p)


def guard(c):
    source=read_config(c['newton_source_config'])
    if newton.fingerprint(c['newton_source_config'])!=c['newton_fit_fingerprint'] or ancestor.analysis_fingerprint(c['newton_source_config'])!=c['newton_analysis_fingerprint']:
        raise ValueError('Frozen Newton identities changed')
    if tree.fingerprint(c['source_config'])!=c['source_fingerprint']:raise ValueError('Frozen fusion identity changed')
    verify_hashes(c['preserve_sha256'])
    if c['candidates']!=[s for s in source['candidates'] if s['id'] in MEMBERS]:raise ValueError('Newton recipes changed')
    if c['strategies']!=strategies():raise ValueError('Strategy declaration changed')
    for key in ['data_path','data_sha256','tree_params','reference_source_config','epsilon','references',
                'presets','fusion_spec','solver','gradient_limit','feature_contract','validation_months',
                'test_month','beta','seed','threads','policies','bootstrap_replicates','bootstrap_seed']:
        if c[key]!=source[key]:raise ValueError('Frozen setting changed: '+key)
    if c['alphas']!=ALPHAS or c['output_dir']!='outputs/baf-spline-score-fusion-v1' or c['report_dir']!='reports/baf-spline-score-fusion-v1':
        raise ValueError('Study declaration changed')
    if any(k in c for k in ['primary_fraction','review_quota','precision_floor']):raise ValueError('Undeclared constraint')
    presence=Path(c['output_dir'])/'preservation.json'
    if presence.exists():
        r=read_config(presence)
        if r['mandatory']!=c['preserve_sha256']:raise ValueError('Mandatory snapshot changed')
        verify_hashes(r['optional_present'])
        if sorted(r['roots'])!=sorted(c['historical_output_roots']):raise ValueError('Historical roots changed')


def initialize(config_path=CONFIG):
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);fp=fingerprint(config_path)
    p=out/'preservation.json'
    if not p.exists():
        present={str(p):digest(p) for root in c['historical_output_roots'] for p in Path(root).rglob('*') if p.is_file()}
        write_json(p,dict(mandatory=c['preserve_sha256'],optional_present=present,
                         roots=c['historical_output_roots'],created_at_utc=now()))
    p=out/'declaration.json'
    expected=dict(fingerprint=fp,config_path=str(config_path),protocol_sha256=digest(c['protocol_path']),
                  preservation_sha256=digest(out/'preservation.json'),strategies=c['strategies'],
                  month7_previously_consumed=True)
    if p.exists():
        r=read_config(p)
        if any(r[k]!=v for k,v in expected.items()):raise ValueError('Prospective identity changed')
    else:write_json(p,dict(**expected,created_at_utc=now()))
    guard(c);return c,fp


def check_expected_record(record,frame,spec,month,phase,identity):
    if any(record.get(k)!=v for k,v in expected_record(frame,spec,month,phase,identity).items()):
        raise ValueError('Recipe/population/fingerprint differs')


def cache_paths(c,spec,month,phase):
    rp,pp=paths(c,phase,month,spec['id'])
    if spec['id'] in MEMBERS:return [rp,pp,newton.model_path(c,phase,month,spec['id'])]
    if spec['id']=='xgb_d4_w1':return [rp,pp,*tree.model_paths(c,phase,month,spec['id'])]
    return [rp,pp,tree.reference_model_path(c,phase,month,spec['id'])]


def audit_base(c,fp,frame,base,spec,month,phase):
    ps=cache_paths(c,spec,month,phase)
    if not tree.cache_state(ps):raise ValueError('Required base missing')
    r=read_config(ps[0]);check_expected_record(r,frame,spec,month,phase,fp)
    if digest(ps[1])!=r['prediction_sha256']:raise ValueError('Saved prediction changed')
    aligned(prediction(c,phase,month,spec['id']),expected_population(frame,month))
    tree.verify_origin(r)
    if spec['id'] in MEMBERS:
        return newton.audit_new(c,fp,frame,base,spec,month,phase)
    if spec['id']=='xgb_d4_w1':return tree.audit_model(c,fp,frame,base,spec,month,phase)
    return tree.reference_inference(c,frame,base,spec,month,phase,r)


def reproduce_reference(c,fp,frame,base,spec,month,phase):
    """Exact single missing recipe, without traversing unrelated legacy caches."""
    train=frame.index[frame.month<month];ev=frame.index[frame.month==month]
    model,adapter=make_estimator(spec,tree.reference_config(c),base.loc[train]);x=base.loc[train]
    if adapter is not None:x=adapter.transform(x)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=c['threads']):
        warnings.simplefilter('always');model.fit(x,frame.loc[train,'fraud_bool'].to_numpy())
        z=base.loc[ev];z=adapter.transform(z) if adapter is not None else z;scores=model.predict_proba(z)[:,1]
    messages=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    if messages:raise ValueError('Reference fit warning: '+str(messages))
    rp,pp,mp=cache_paths(c,spec,month,phase)
    mp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(dict(model=model,adapter=adapter,spec=spec),mp)
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=scores)
    write_json(rp,dict(**expected_record(frame,spec,month,phase,fp),prediction_sha256=digest(pp),
        reference_model_sha256=digest(mp),reference_model_path=str(mp),reference_model_new_fit=True,
        reference_model_warnings=[],warnings=messages,reused=False,fit_seconds=time.perf_counter()-start))


def prepare_base(c,fp,frame,base,spec,month,phase='validation',allow_missing=True):
    own=cache_paths(c,spec,month,phase)
    if tree.cache_state(own):return audit_base(c,fp,frame,base,spec,month,phase)
    if not allow_missing:raise ValueError('Required cache missing at audit')
    source=read_config(c['newton_source_config'] if spec['id'] in MEMBERS else c['source_config'])
    origin=cache_paths(source,spec,month,phase)
    if tree.cache_state(origin):
        original_fp=c['newton_fit_fingerprint'] if spec['id'] in MEMBERS else c['source_fingerprint']
        r=audit_base(source,original_fp,frame,base,spec,month,phase)
        for src,dest in zip(origin[1:],own[1:]):dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        record={**r,'fingerprint':fp,'reused':True,'origin_artifacts':{str(p):digest(p) for p in origin}}
        if spec['id'] in MEMBERS:
            record.update(original_fit_fingerprint=c['newton_fit_fingerprint'],original_analysis_fingerprint=c['newton_analysis_fingerprint'])
        elif spec['id']=='xgb_d4_w1':record['model_path']=str(own[2])
        else:record.update(reference_model_path=str(own[2]),reference_model_new_fit=False)
        write_json(own[0],record)
    elif spec['id'] in MEMBERS:newton.fit_new(c,fp,frame,base,spec,month,phase)
    elif spec['id']=='xgb_d4_w1':tree.old.fit_model(c,fp,frame,base,spec,month,phase)
    else:reproduce_reference(c,fp,frame,base,spec,month,phase)
    return audit_base(c,fp,frame,base,spec,month,phase)


def derived(c,fp,spec,month,phase='validation',allow_missing=True):
    rp,pp=paths(c,phase,month,spec['id']);names=spec['members']
    items={n:prediction(c,phase,month,n) for n in names}
    data=compose(spec,items)
    deps={n:dict(record=digest(paths(c,phase,month,n)[0]),prediction=digest(paths(c,phase,month,n)[1]),
                 model_identity=read_config(paths(c,phase,month,n)[0]).get('model_sha256',
                 read_config(paths(c,phase,month,n)[0]).get('reference_model_sha256'))) for n in names}
    expected=dict(candidate=spec['id'],phase=phase,eval_month=month,spec=spec,fingerprint=fp,
                  dependencies=deps,epsilon=EPSILON,transform='fixed common-margin composition v1')
    if tree.cache_state([rp,pp]):
        r=read_config(rp);aligned(data,prediction(c,phase,month,spec['id']))
        if any(r.get(k)!=v for k,v in expected.items()) or digest(pp)!=r['prediction_sha256'] or not np.array_equal(data['scores'],prediction(c,phase,month,spec['id'])['scores']):
            raise ValueError('Derived provenance/arithmetic changed')
        return r
    if not allow_missing:raise ValueError('Derived score missing')
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**data)
    r=dict(**expected,prediction_sha256=digest(pp));write_json(rp,r);return r


def required_newton(names):
    found=set()
    for n in names:
        spec=next(s for s in strategies() if s['id']==n)
        if spec['kind']=='raw':found.add(n)
        elif spec['kind']=='logit':found.add(spec['members'][0])
        elif spec['kind']=='margin_blend':found.add(spec['members'][0].removeprefix('logit_'))
    return found


def prepare(c,fp,frame,base,phase='validation',names=None,allow_missing=True):
    names=set(s['id'] for s in c['strategies']) if names is None else set(names)
    needed=required_newton(names);records=[]
    for month in c['validation_months'] if phase=='validation' else [7]:
        for spec in c['references']+c['presets']+[s for s in c['candidates'] if s['id'] in needed]:
            records.append(prepare_base(c,fp,frame,base,spec,month,phase,allow_missing))
        tree.ensure_score(c,fp,c['fusion_spec'],month,phase,allow_missing)
        derived_names=set(names)
        for s in c['strategies']:
            if s['id'] in names and s['kind']=='margin_blend':derived_names.add(s['members'][0])
        for s in c['strategies']:
            if s['id'] in derived_names and s['kind'] in ['logit','margin_blend']:
                derived(c,fp,s,month,phase,allow_missing)
    return records


def history(c):
    return {s['id']:{m:prediction(c,'validation',m,s['id']) for m in c['validation_months']} for s in c['strategies']}


def crossfit(c):
    h=history(c);rows=[]
    for m in [5,6]:
        for n,items in h.items():
            aligned(items[m],h[FUSION][m])
            for k in POLICIES:
                r=learn({t:d for t,d in items.items() if t<m},k,m)
                rows.append(dict(candidate=n,policy=k,eval_month=m,source_months=','.join(map(str,r['source_months'])),
                                 parameters=json.dumps(r,sort_keys=True),**measure(items[m],r)))
    if len(rows)!=138:raise ValueError('Incomplete forward grid')
    return pd.DataFrame(rows)


def tables(c,cf):
    h=history(c);summary=order_summary(cf).reset_index(drop=True);lookup=summary.set_index(['candidate','policy']);contrasts=[]
    for label,n,k,ref,rk in contrast_factors():
        a,b=lookup.loc[(n,k)],lookup.loc[(ref,rk)]
        contrasts.append(dict(contrast=label,candidate=n,policy=k,reference=ref,reference_policy=rk,
            **{m+'_difference':float(a[m]-b[m]) for m in ['f2','precision','recall','flagged_fraction','average_precision']}))
    pooled=[]
    for (n,k),g in cf.groupby(['candidate','policy']):
        t=g[['tn','fp','fn','tp']].sum();pooled.append(dict(candidate=n,policy=k,
            **confusion_metrics(*[t[v] for v in ['tn','fp','fn','tp']]),flagged_fraction=(t.tp+t.fp)/g.n.sum()))
    score=[];clip=[];div=[];joint=[]
    for n,items in h.items():
        for m,d in items.items():
            x=d['scores'];score.append(dict(candidate=n,eval_month=m,n=len(x),score_min=float(x.min()),
                score_max=float(x.max()),score_mean=float(x.mean()),unique_scores=len(np.unique(x)),tied_fraction=1-len(np.unique(x))/len(x)))
    for n in MEMBERS:
        for m in c['validation_months']:
            clip.append(dict(candidate=n,eval_month=m,**clipping(h[n][m]['scores'])))
            div.append(dict(candidate='logit_'+n,reference=FUSION,eval_month=m,
                            **diversity(h['logit_'+n][m]['scores'],h[FUSION][m]['scores'])))
        for m in [5,6]:
            for k in POLICIES:
                a,b=[decisions(h[name][m],json.loads(cf[(cf.candidate==name)&(cf.policy==k)&(cf.eval_month==m)].iloc[0].parameters)) for name in ['logit_'+n,FUSION]]
                y=h[n][m]['labels'].astype(bool);r=dict(candidate='logit_'+n,reference=FUSION,eval_month=m,policy=k,n=len(y))
                for label,mask in [('fraud',y),('legitimate',~y)]:
                    r.update({label+'_both_flagged':int((mask&a&b).sum()),label+'_newton_only':int((mask&a&~b).sum()),
                              label+'_fusion_only':int((mask&~a&b).sum()),label+'_neither':int((mask&~a&~b).sum())})
                joint.append(r)
    return dict(crossfit=cf,summary=summary,pooled=pd.DataFrame(pooled),contrasts=pd.DataFrame(contrasts),
                score_diagnostics=pd.DataFrame(score),clipping=pd.DataFrame(clip),diversity=pd.DataFrame(div),joint_errors=pd.DataFrame(joint))


def validate(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Frozen study cannot be retuned')
    write_json(out/'fixtures.json',dict(**fixtures(),fingerprint=fp))
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
    records=prepare(c,fp,frame,base);cf=crossfit(c)
    for n,t in tables(c,cf).items():t.to_csv(out/(n+'.csv'),index=False)
    from .spline_fusion_report import audit_development
    audit_development(c,fp,frame,base,check_models=False)
    write_json(out/'validation_complete.json',dict(fingerprint=fp,created_at_utc=now(),files={p:digest(out/p) for p in TABLES},
        base_results=len(records),base_reuses=sum(bool(r.get('reused')) for r in records),
        new_fits=sum(not r.get('reused',False) for r in records),derived_results=51,total_scores=72,
        selectable_scores=69,forward_rows=138,miniature_fits=0))
    guard(c);print(order_summary(cf).head(12).to_string(index=False),flush=True)


def freeze(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir']);done=read_config(out/'validation_complete.json')
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    if done['fingerprint']!=fp:raise ValueError('Development identity changed')
    verify_hashes({str(out/p):h for p,h in done['files'].items()})
    independent=read_config(out/'independent_development_audit.json')
    if independent['fingerprint']!=fp or independent['validation_sha256']!=digest(out/'validation_complete.json'):
        raise ValueError('Independent audit missing/stale')
    summary=pd.read_csv(out/'summary.csv');winner=summary.iloc[0];h=history(c)
    roles=choices_for(str(winner.candidate),str(winner.policy))
    write_json(out/'selection.json',dict(created_at_utc=now(),fingerprint=fp,config=c,
        selected_candidate=str(winner.candidate),selected_policy=str(winner.policy),selected_development_f2=float(winner.f2),
        choices=roles,rules={a:learn(h[n],k,7) for a,(n,k) in roles.items()},validation_files=done['files'],
        independent_audit_sha256=digest(out/'independent_development_audit.json'),
        artifacts={str(p):digest(p) for folder in ['runs','predictions','models'] for p in (out/folder).glob('*') if p.is_file()},
        month7_previously_consumed=True,exploratory=True))
    print('Frozen',winner.candidate,winner.policy,float(winner.f2),'roles',len(roles),flush=True)


def load_selection(config_path=CONFIG):
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['config']!=c or s['fingerprint']!=fingerprint(config_path):raise ValueError('Frozen identity changed')
    verify_hashes(s['artifacts']);verify_hashes({str(out/p):h for p,h in s['validation_files'].items()})
    if s['independent_audit_sha256']!=digest(out/'independent_development_audit.json'):raise ValueError('Frozen audit changed')
    best=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0]
    if s['selected_candidate']!=best.candidate or s['selected_policy']!=best.policy or not np.isclose(s['selected_development_f2'],best.f2,atol=1e-12,rtol=0):raise ValueError('Frozen winner changed')
    if s['choices']!=choices_for(str(best.candidate),str(best.policy)):raise ValueError('Frozen controls changed')
    h=history(c)
    if any(s['rules'][a]!=learn(h[n],k,7) for a,(n,k) in s['choices'].items()):raise ValueError('Frozen final rules changed')
    return c,s


def final_data(c,s):return {a:prediction(c,'exploratory',7,n) for a,(n,k) in s['choices'].items()}


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);sha=digest(out/'selection.json');p=out/'exploratory_access.json'
    if not p.exists():write_json(p,dict(access_at_utc=now(),selection_sha256=sha,month7_previously_consumed=True))
    r=read_config(p)
    if r['selection_sha256']!=sha or not s['created_at_utc']<r['access_at_utc']:raise ValueError('Freeze/access chronology changed')
    frame=load_data(c);base=features(frame);names={n for n,k in s['choices'].values()}
    records=prepare(c,s['fingerprint'],frame,base,'exploratory',names)
    rows=[];data=final_data(c,s)
    for a,d in data.items():
        aligned(d,data['selected']);rule=s['rules'][a]
        rows.append(dict(role=a,candidate=s['choices'][a][0],policy=s['choices'][a][1],eval_month=7,
                         parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    write_json(out/'transfer_complete.json',dict(selection_sha256=sha,table_sha256=digest(out/'exploratory.csv'),
        base_results=len(records),base_reuses=sum(bool(r.get('reused')) for r in records),new_fits=sum(not r.get('reused',False) for r in records),
        new_newton_fits=sum(r['candidate'] in MEMBERS and not r.get('reused',False) for r in records),role_count=len(rows)))
    guard(c);print(table[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['initialize','validate','freeze','exploratory']);parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args();globals()[args.command](args.config)
