"""Newton additive study with checked objectives and solver-path observation."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import resource
import shutil
import time
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import digest, features, load_data, read_config, write_json
from .improvements import paths, prediction
from .focal_study import expected_record, now
from .recall_extension import expected_population, order_summary
from .recall_weighted_metrics import POLICIES, aligned, learn, measure
from .improvement_metrics import confusion_metrics
from .calibrated_policy import verify_table
from . import score_fusion_study as old
from .spline_inputs import AdditiveInputs, recipes, checked_scores
from . import spline_study as stopped
import hashlib
import inspect
import sys
import warnings
from contextlib import contextmanager
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._glm._newton_solver import NewtonSolver

CONFIG='configs/newton-spline-logistic.json'
FUSION='logit_mean_xgb_d4_w1_baseline_balanced_a50'
REFERENCE_IDS=[FUSION,'baseline_balanced','baseline_lgbm']
SOURCES=stopped.SOURCES+['baf/newton_spline_study.py','baf/newton_spline_report.py']
TABLES=['crossfit.csv','summary.csv','pooled.csv','contrasts.csv','score_diagnostics.csv',
        'solver.csv','feature_blocks.json','feature_diagnostics.csv','coefficients.csv',
        'numerical.csv','old_solver_comparisons.csv','preservation.json']


def fingerprint(path):
    result=old.fingerprint(path)
    result['sources']={p:digest(p) for p in SOURCES}
    return result


def declared_strategies(c):
    return [dict(id=s['id'],kind='new',members=[s['id']]) for s in recipes()]+[
        dict(id=n,kind='reference',members=[n]) for n in REFERENCE_IDS]


def guard(c):
    source=read_config(c['source_config']);old.guard(source)
    if c['gradient_limit']!=1.1e-6:raise ValueError('Gradient gate changed')
    original=read_config(stopped.CONFIG)
    if any(c[k]!=v for k,v in original.items() if k not in ['solver','output_dir','report_dir','preserve_sha256']):
        raise ValueError('Original representation or dependency contract changed')
    if c['output_dir']!='outputs/baf-newton-spline-logistic-v1' or c['report_dir']!='reports/baf-newton-spline-logistic-v1':
        raise ValueError('New study destinations changed')
    record=read_config('reports/baf-spline-logistic-v1/stopped.json')
    expected_optional={**record['artifacts'],'outputs/baf-spline-logistic-v1/stopped.json':digest('reports/baf-spline-logistic-v1/stopped.json')}
    if c['stopped_optional_hashes']!=expected_optional:raise ValueError('Stopped artifact declaration changed')
    presence=Path(c['output_dir'])/'preservation.json'
    verify_stopped_presence(expected_optional,presence,c['preserve_sha256'])
    if c['source_config']!=old.CONFIG or old.fingerprint(c['source_config'])!=c['source_fingerprint']:
        raise ValueError('Tenth study identity changed')
    for p,h in c['preserve_sha256'].items():
        if not Path(p).is_file() or digest(p)!=h:raise ValueError('Preserved artifact changed: '+p)
    if c['candidates']!=recipes() or c['strategies']!=declared_strategies(c):
        raise ValueError('Declared representation grid changed')
    for k,v in dict(validation_months=[4,5,6],test_month=7,beta=2,seed=2026,threads=6,
                    policies=POLICIES,bootstrap_replicates=500,bootstrap_seed=1027).items():
        if c[k]!=v:raise ValueError('Declared setting changed: '+k)
    for k in ['data_path','data_sha256','tree_params','reference_source_config','epsilon']:
        if c[k]!=source[k]:raise ValueError('Frozen dependency recipe changed: '+k)
    if c['references']!=source['references'][:2] or c['presets']!=source['presets'][:1]:
        raise ValueError('Frozen base members changed')
    if c['fusion_spec']!=next(s for s in source['strategies'] if s['id']==FUSION):
        raise ValueError('Frozen fusion changed')
    if c['solver']!=solver_parameters(recipes()[0]) or c['feature_contract']!=dict(
            smooth_count=17,discrete_count=7,categorical_count=5,quantiles=[0,.25,.5,.75,1],
            degree=3,extrapolation='constant',include_bias=False,scale_center=False):
        raise ValueError('Representation/solver contract changed')
    if any(k in c for k in ['primary_fraction','review_quota','precision_floor']):
        raise ValueError('Undeclared operating constraint')
    roots={str(Path(p).parent) for p in c['preserve_sha256'] if p.endswith('/selection.json')}
    destinations=[Path(c[k]).resolve() for k in ['output_dir','report_dir']]
    for a in destinations:
        for root in roots:
            b=Path(root).resolve()
            if a==b or a in b.parents or b in a.parents:raise ValueError('Destination overlaps preserved study')
    a,b=destinations
    if a==b or a in b.parents or b in a.parents:raise ValueError('Destinations overlap')


def model_path(c,phase,month,name):
    return Path(c['output_dir'])/'models'/f'{phase}-m{month}-{name}.joblib'


def signature(model):return {k:repr(v) for k,v in sorted(model.get_params(deep=False).items())}


def audit_new(c,identity,frame,base,spec,month,phase='validation'):
    rp,pp=paths(c,phase,month,spec['id']);mp=model_path(c,phase,month,spec['id'])
    if not old.cache_state([rp,pp,mp]):raise ValueError('New model missing at audit')
    r=read_config(rp);expected=expected_record(frame,spec,month,phase,identity)
    if any(r[k]!=v for k,v in expected.items()):raise ValueError('Logistic population/identity changed')
    if digest(pp)!=r['prediction_sha256'] or digest(mp)!=r['model_sha256']:
        raise ValueError('Logistic model/prediction artifact changed')
    a=joblib.load(mp);inp=a['inputs'];model=a['model']
    if a['spec']!=spec or inp.representation!=spec['representation'] or inp.metadata()!=r['feature_metadata']:
        raise ValueError('Saved representation metadata changed')
    from sklearn.linear_model import LogisticRegression
    if signature(model)!=signature(LogisticRegression(**solver_parameters(spec,c['seed']))):
        raise ValueError('Saved logistic recipe changed')
    if r['solver_params']!=solver_parameters(spec,c['seed']):raise ValueError('Recorded solver recipe differs')
    if r['warnings'] or int(model.n_iter_.max())!=r['iterations'] or r['iterations']>=c['solver']['max_iter']:
        raise ValueError('Saved convergence gate changed')
    train=frame.index[frame.month<month]
    # Refit only the representation to audit its training boundary and statistics.
    expected_inputs=AdditiveInputs(spec['representation']).fit(base.loc[train])
    if expected_inputs.metadata()!=inp.metadata():raise ValueError('Training-only representation differs')
    d=prediction(c,phase,month,spec['id']);aligned(d,expected_population(frame,month))
    with threadpool_limits(limits=c['threads']):p=checked_scores(model,inp.transform(base.loc[d['row_ids']]))
    if not np.allclose(p,d['scores'],atol=1e-12,rtol=1e-12):raise ValueError('Saved logistic inference differs')
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError('Nonfinite saved coefficients')
    diagnostics=objective_gradient(model,inp.transform(base.loc[train]),frame.loc[train,'fraud_bool'].to_numpy(),spec)
    check_numerical(model,r['warnings'],r['numerical'],diagnostics)
    return r


def fit_new(c,identity,frame,base,spec,month,phase='validation',allow_missing=True,inputs=None,matrix=None):
    rp,pp=paths(c,phase,month,spec['id']);mp=model_path(c,phase,month,spec['id'])
    if old.cache_state([rp,pp,mp]):return audit_new(c,identity,frame,base,spec,month,phase)
    if not allow_missing:raise ValueError('New model missing at audit')
    train=frame.index[frame.month<month];ev=frame.index[frame.month==month]
    inp=inputs if inputs is not None else AdditiveInputs(spec['representation']).fit(base.loc[train])
    z=matrix if matrix is not None else inp.transform(base.loc[train])
    if inp.representation!=spec['representation'] or len(train)!=z.shape[0]:raise ValueError('Transform reuse mismatch')
    numerical={}
    start=time.perf_counter();print('Fitting',rp.stem,f'rows={len(train):,} dim={z.shape[1]}',flush=True)
    try:
        model,messages,numerical=fit_checked(z,frame.loc[train,'fraud_bool'].to_numpy(),spec,c['threads'],c['seed'])
        with threadpool_limits(limits=c['threads']):scores=checked_scores(model,inp.transform(base.loc[ev]))
    except Exception as exc:
        write_json(Path(c['output_dir'])/'failures'/(rp.stem+'.json'),dict(
            **expected_record(frame,spec,month,phase,identity),error=repr(exc),numerical=getattr(exc,'diagnostics',numerical),created_at_utc=now()))
        raise
    mp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(dict(model=model,inputs=inp,spec=spec),mp)
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,row_ids=ev.to_numpy(),
        labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=scores)
    d=prediction(c,phase,month,spec['id'])
    r={**expected_record(frame,spec,month,phase,identity),**measure(d,dict(kind='pooled',threshold=.5)),
       'model_sha256':digest(mp),'prediction_sha256':digest(pp),'warnings':messages,
       'iterations':int(model.n_iter_.max()),'feature_metadata':inp.metadata(),
       'transformed_dimension':z.shape[1],'matrix_nnz':int(z.nnz),'matrix_density':z.nnz/(z.shape[0]*z.shape[1]),
       'numerical':numerical,'fit_seconds':time.perf_counter()-start,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'reused':False,'solver_params':solver_parameters(spec,c['seed'])}
    write_json(rp,r);print('Finished',rp.stem,f"iterations={r['iterations']} seconds={r['fit_seconds']:.1f}",flush=True)
    return audit_new(c,identity,frame,base,spec,month,phase)


def reference_bases(c,identity,frame,base,phase='validation',allow_missing=True):
    source=read_config(c['source_config']);rows=[]
    for month in c['validation_months'] if phase=='validation' else [7]:
        for spec in c['references']:
            rp,pp=paths(c,phase,month,spec['id']);mp=old.reference_model_path(c,phase,month,spec['id'])
            if not old.cache_state([rp,pp,mp]):
                if not allow_missing:raise ValueError('Reference missing at audit')
                sr,sp=paths(source,phase,month,spec['id']);sm=old.reference_model_path(source,phase,month,spec['id'])
                if old.cache_state([sr,sp,sm]):
                    raw=read_config(sr)
                    expected=expected_record(frame,spec,month,phase,c['source_fingerprint'])
                    if any(raw[k]!=v for k,v in expected.items()):raise ValueError('Source reference identity differs')
                    if digest(sp)!=raw['prediction_sha256']:raise ValueError('Source reference prediction changed')
                    aligned(prediction(source,phase,month,spec['id']),expected_population(frame,month))
                    old.reference_inference(source,frame,base,spec,month,phase,raw)
                    for src,dst in [(sp,pp),(sm,mp)]:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
                    r={**raw,'fingerprint':identity,'reused':True,'reference_model_path':str(mp),
                       'reference_model_new_fit':False,'origin_artifacts':{str(p):digest(p) for p in [sr,sp,sm]}}
                    write_json(rp,r)
                else:
                    old.prepare_references(c,identity,frame,base,phase,[spec],True)
            r=read_config(rp);expected=expected_record(frame,spec,month,phase,identity)
            if any(r[k]!=v for k,v in expected.items()) or digest(pp)!=r['prediction_sha256']:
                raise ValueError('Own reference identity differs')
            aligned(prediction(c,phase,month,spec['id']),expected_population(frame,month))
            rows.append(old.reference_inference(c,frame,base,spec,month,phase,r))
        for spec in c['presets']:
            rows.append(old.fit_model(c,identity,frame,base,spec,month,phase,allow_missing))
        old.ensure_score(c,identity,c['fusion_spec'],month,phase,allow_missing)
    return rows


def history(c):
    return {s['id']:{m:prediction(c,'validation',m,s['id']) for m in c['validation_months']} for s in c['strategies']}


def crossfit(c):
    h=history(c);rows=[]
    for month in [5,6]:
        for name,items in h.items():
            aligned(items[month],h['baseline_balanced'][month])
            for kind in POLICIES:
                rule=learn({m:d for m,d in items.items() if m<month},kind,month)
                rows.append(dict(candidate=name,policy=kind,eval_month=month,
                    source_months=','.join(map(str,rule['source_months'])),parameters=json.dumps(rule,sort_keys=True),
                    **measure(items[month],rule)))
    if len(rows)!=90:raise ValueError('Incomplete forward policy grid')
    return pd.DataFrame(rows)


def contrast_factors(c):
    rows=[];lookup={(s['representation'],s['C'],s['class_weight']):s['id'] for s in c['candidates']}
    for s in c['strategies']:
        for k in POLICIES:
            rows += [('old pooled reference',s['id'],k,r,'pooled') for r in REFERENCE_IDS]
            if k!='pooled':rows.append(('policy minus own pooled',s['id'],k,s['id'],'pooled'))
    for C in [.01,.1,1.]:
        for k in POLICIES:
            for w in [None,'balanced']:
                rows.append(('spline minus linear',lookup['spline',C,w],k,lookup['linear',C,w],k))
            for rep in ['linear','spline']:
                rows.append(('balanced minus unweighted',lookup[rep,C,'balanced'],k,lookup[rep,C,None],k))
    for rep,w in itertools.product(['linear','spline'],[None,'balanced']):
        for lo,hi in itertools.combinations([.01,.1,1.],2):
            for k in POLICIES:rows.append(('larger C minus smaller C',lookup[rep,hi,w],k,lookup[rep,lo,w],k))
    if len(rows)!=237:raise ValueError('Incomplete contrast declaration')
    return rows


def export_tables(c,cf,frame,base):
    out=Path(c['output_dir']);summary=order_summary(cf);lookup=summary.set_index(['candidate','policy']);rows=[]
    for label,n,k,r,rk in contrast_factors(c):
        a,b=lookup.loc[(n,k)],lookup.loc[(r,rk)]
        rows.append(dict(contrast=label,candidate=n,policy=k,reference=r,reference_policy=rk,
            **{m+'_difference':float(a[m]-b[m]) for m in ['f2','precision','recall','flagged_fraction','average_precision']}))
    pd.DataFrame(rows).to_csv(out/'contrasts.csv',index=False);pooled=[]
    for (n,k),g in cf.groupby(['candidate','policy']):
        t=g[['tn','fp','fn','tp']].sum();pooled.append(dict(candidate=n,policy=k,
            **confusion_metrics(*[t[v] for v in ['tn','fp','fn','tp']]),flagged_fraction=(t.tp+t.fp)/g.n.sum()))
    pd.DataFrame(pooled).to_csv(out/'pooled.csv',index=False)
    diags=[]
    for n,items in history(c).items():
        for m,d in items.items():
            x=d['scores'];diags.append(dict(candidate=n,eval_month=m,n=len(x),score_min=float(x.min()),
                score_max=float(x.max()),score_mean=float(x.mean()),unique_scores=len(np.unique(x)),tied_fraction=1-len(np.unique(x))/len(x)))
    pd.DataFrame(diags).to_csv(out/'score_diagnostics.csv',index=False)
    solvers=[];blocks={};fd=[];coeff=[]
    for m in c['validation_months']:
        for s in c['candidates']:
            r=read_config(paths(c,'validation',m,s['id'])[0]);a=joblib.load(model_path(c,'validation',m,s['id']))
            solvers.append(dict(candidate=s['id'],eval_month=m,iterations=r['iterations'],warnings=json.dumps(r['warnings']),
                transformed_dimension=r['transformed_dimension'],matrix_nnz=r['matrix_nnz'],matrix_density=r['matrix_density'],
                fit_seconds=r['fit_seconds'],peak_process_rss_bytes=r['peak_process_rss_bytes'],solver_params=json.dumps(r['solver_params'],sort_keys=True)))
            blocks[f"m{m}-{s['id']}"]=r['feature_metadata']
            for phase,ids in [('training',frame.index[frame.month<m]),('prediction',frame.index[frame.month==m])]:
                fd += [dict(candidate=s['id'],eval_month=m,population=phase,**d) for d in a['inputs'].diagnostics(base.loc[ids])]
            coeff += [dict(candidate=s['id'],eval_month=m,feature=n,coefficient=float(v))
                      for n,v in zip(a['inputs'].names,a['model'].coef_[0])]
            coeff.append(dict(candidate=s['id'],eval_month=m,feature='__INTERCEPT__',coefficient=float(a['model'].intercept_[0])))
    pd.DataFrame(solvers).to_csv(out/'solver.csv',index=False);write_json(out/'feature_blocks.json',blocks)
    pd.DataFrame(fd).to_csv(out/'feature_diagnostics.csv',index=False);pd.DataFrame(coeff).to_csv(out/'coefficients.csv',index=False)
    cf.to_csv(out/'crossfit.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
    return summary


def audit_grid(c,identity,frame,base):
    records=reference_bases(c,identity,frame,base,allow_missing=False)
    for m in c['validation_months']:
        for spec in c['candidates']:records.append(audit_new(c,identity,frame,base,spec,m))
    if len(records)!=45:raise ValueError('Incomplete fitted base audit')
    return records


def validate(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Frozen study cannot be retuned')
    preflight=read_config(out/'miniature.json')
    if preflight['fingerprint']!=identity or preflight['fit_count']!=24 or len(preflight['results'])!=12:
        raise ValueError('Missing or stale miniature gate')
    if digest(out/'preservation.json')!=preflight['preservation_sha256']:raise ValueError('Preservation presence snapshot changed')
    out.mkdir(parents=True,exist_ok=True);frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
    refs=reference_bases(c,identity,frame,base);new=[]
    for m in c['validation_months']:
        train=frame.index[frame.month<m]
        for rep in ['linear','spline']:
            inp=AdditiveInputs(rep).fit(base.loc[train]);matrix=inp.transform(base.loc[train])
            for s in c['candidates']:
                if s['representation']==rep:new.append(fit_new(c,identity,frame,base,s,m,inputs=inp,matrix=matrix))
    cf=crossfit(c);summary=export_tables(c,cf,frame,base);export_numerical(c,frame,base)
    audit_development(c,identity,frame,base)
    for name,value in [(FUSION,.3289065701775199),('baseline_balanced',.32780157809618937),('baseline_lgbm',.3226536849212962)]:
        if not np.isclose(summary[(summary.candidate==name)&(summary.policy=='pooled')].iloc[0].f2,value,atol=1e-12,rtol=0):
            raise ValueError('Preserved pooled reference differs')
    write_json(out/'validation_complete.json',dict(fingerprint=identity,created_at_utc=now(),
        files={p:digest(out/p) for p in TABLES+['miniature.json']},new_logistic_fits=len(new),
        reference_reuses=sum(r.get('reused',False) for r in refs),reference_results=len(refs),
        reference_fits=sum(not r.get('reused',False) for r in refs),forward_rows=90,
        selectable_scores=45,all_score_artifacts=48,miniature_fits=24))
    guard(c);print(summary.head(12).to_string(index=False),flush=True)


def choices_for(c,name,kind):
    roles=dict(selected=[name,kind],fusion=[FUSION,'pooled'],balanced=['baseline_balanced','pooled'],
               unweighted=['baseline_lgbm','pooled'],same_model=[name,'pooled'])
    spec=next((s for s in c['candidates'] if s['id']==name),None)
    if spec:
        pairs={k:dict(spec) for k in ['matched_representation','matched_weight']}
        pairs['matched_representation']['representation']='linear' if spec['representation']=='spline' else 'spline'
        pairs['matched_weight']['class_weight']=None if spec['class_weight']=='balanced' else 'balanced'
        for role,target in pairs.items():
            match=next(s for s in c['candidates'] if all(s[k]==target[k] for k in ['representation','C','class_weight']))
            roles[role]=[match['id'],kind]
    return roles


def freeze(config_path=CONFIG):
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    complete=read_config(out/'validation_complete.json')
    if complete['fingerprint']!=identity or any(digest(out/p)!=h for p,h in complete['files'].items()):
        raise ValueError('Stale development evidence')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);audit_development(c,identity,frame,base)
    cf=crossfit(c);verify_table(cf,out/'crossfit.csv');summary=order_summary(cf).reset_index(drop=True)
    verify_table(summary,out/'summary.csv');winner=summary.iloc[0];choices=choices_for(c,str(winner.candidate),str(winner.policy));h=history(c)
    write_json(out/'selection.json',dict(created_at_utc=now(),fingerprint=identity,config=c,
        selected_candidate=str(winner.candidate),selected_policy=str(winner.policy),selected_development_f2=float(winner.f2),
        choices=choices,rules={a:learn(h[n],k,7) for a,(n,k) in choices.items()},
        validation_files=complete['files'],artifacts={str(p):digest(p) for folder in ['runs','predictions','models']
            for p in (out/folder).glob('*') if p.is_file()},month7_previously_consumed=True,exploratory=True))
    print('Frozen',winner.candidate,winner.policy,float(winner.f2),flush=True)


def load_selection(config_path=CONFIG):
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['config']!=c or s['fingerprint']!=fingerprint(config_path):raise ValueError('Frozen identity changed')
    if any(not Path(p).is_file() or digest(p)!=h for p,h in s['artifacts'].items()):raise ValueError('Frozen model/score changed')
    if any(digest(out/p)!=h for p,h in s['validation_files'].items()):raise ValueError('Frozen development table changed')
    best=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0]
    if s['choices']!=choices_for(c,str(best.candidate),str(best.policy)) or not np.isclose(best.f2,s['selected_development_f2'],atol=1e-12,rtol=0):
        raise ValueError('Frozen winner/controls differ')
    if str(best.candidate)!=s['selected_candidate'] or str(best.policy)!=s['selected_policy']:raise ValueError('Frozen selection differs')
    h=history(c)
    if any(rule!=learn(h[s['choices'][a][0]],s['choices'][a][1],7) for a,rule in s['rules'].items()):
        raise ValueError('Frozen rule differs')
    return c,s


def final_data(c,s):return {a:prediction(c,'exploratory',7,n) for a,(n,k) in s['choices'].items()}


def miniature_frame():
    from .spline_inputs import NUMERIC, CATEGORICAL
    rng=np.random.default_rng(1027);n=160
    x=pd.DataFrame({col:rng.normal(size=n) for col in NUMERIC})
    for col in CATEGORICAL:x[col]=np.where(np.arange(n)%2,'A','B')
    x.loc[::3,'prev_address_months_count']=np.nan
    x['intended_balcon_amount']=np.nan;x['zip_count_4w']=1.
    x['customer_age']=np.where(np.arange(n)%4,20.,40.)
    x['proposed_credit_limit']=np.where(np.arange(n)%5,200.,1500.)
    x.loc[0,'device_os']=None
    labels=(x.name_email_similarity.to_numpy()+rng.normal(size=n)>.9).astype(int)
    return x,labels


def miniature(config_path=CONFIG):
    c=read_config(config_path);guard(c);initialize_preservation(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    gate=out/'miniature.json'
    if gate.exists():
        r=read_config(gate)
        if r['fingerprint']!=identity or r['fit_count']!=24 or r['preservation_sha256']!=digest(out/'preservation.json'):
            raise ValueError('Stale miniature cache')
        print('Verified prior miniature gate: 24 fits',flush=True);return
    x,y=miniature_frame();future=x.iloc[:8].copy();future.loc[future.index[0],'device_os']='NEW'
    future.loc[future.index[1],'income']=np.nan;future.loc[future.index[2],'name_email_similarity']=100.
    rows=[]
    with tempfile.TemporaryDirectory() as temp:
        for spec in c['candidates']:
            inp=AdditiveInputs(spec['representation']).fit(x);matrix=inp.transform(x);runs=[]
            for repeat in range(2):
                model,messages,numerical=fit_checked(matrix,y,spec,c['threads'],c['seed'])
                p=checked_scores(model,inp.transform(future));mp=Path(temp)/(spec['id']+'.joblib')
                joblib.dump(dict(model=model,inputs=inp),mp);a=joblib.load(mp)
                if not np.allclose(p,checked_scores(a['model'],a['inputs'].transform(future)),atol=1e-12,rtol=1e-12):
                    raise ValueError('Miniature persisted inference differs')
                runs.append(dict(iterations=int(model.n_iter_.max()),warnings=messages,numerical=numerical,scores=p.tolist()))
            if runs[0]!=runs[1]:raise ValueError('Miniature repeat differs')
            rows.append(dict(spec=spec,dimension=matrix.shape[1],runs=runs))
    write_json(gate,dict(fingerprint=identity,fit_count=24,created_at_utc=now(),results=rows,
        preservation_sha256=digest(out/'preservation.json'),
        fixture='seed 1027; missing/constant/duplicate knots/unseen category/new missingness/outside range; no BAF fits'))
    print('Miniature gate passed: 24 converged deterministic persisted fits',flush=True)


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);ap=out/'exploratory_access.json';sha=digest(out/'selection.json')
    if not ap.exists():write_json(ap,dict(access_at_utc=now(),selection_sha256=sha,month7_previously_consumed=True))
    access=read_config(ap)
    if access['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Freeze/access chronology differs')
    frame=load_data(c);base=features(frame);refs=reference_bases(c,s['fingerprint'],frame,base,'exploratory');new=[]
    names={n for n,k in s['choices'].values()}
    for spec in c['candidates']:
        if spec['id'] in names:new.append(fit_new(c,s['fingerprint'],frame,base,spec,7,'exploratory'))
    data=final_data(c,s);rows=[]
    for a,d in data.items():
        aligned(d,data['balanced']);rule=s['rules'][a]
        rows.append(dict(role=a,candidate=s['choices'][a][0],policy=s['choices'][a][1],eval_month=7,
            parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    write_json(out/'transfer_complete.json',dict(selection_sha256=sha,table_sha256=digest(out/'exploratory.csv'),
        new_logistic_fits=len(new),reference_results=len(refs),reference_reuses=sum(r.get('reused',False) for r in refs),
        reference_fits=sum(not r.get('reused',False) for r in refs),role_count=len(rows)))
    guard(c);print(table[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']].to_string(index=False),flush=True)


def solver_parameters(spec,seed=2026):
    return dict(C=spec['C'],class_weight=spec['class_weight'],solver='newton-cholesky',
                l1_ratio=0.,tol=1e-6,max_iter=100,fit_intercept=True,warm_start=False,random_state=seed)


def objective_gradient(model,matrix,y,spec):
    y=np.asarray(y,dtype=float);coef=np.asarray(model.coef_[0]);intercept=float(model.intercept_[0])
    if matrix.shape!=(len(y),len(coef)) or set(np.unique(y))!={0,1}:raise ValueError('Invalid objective population')
    w=np.ones(len(y))
    if spec['class_weight']=='balanced':
        counts=np.bincount(y.astype(int),minlength=2);w=len(y)/(2*counts[y.astype(int)])
    elif spec['class_weight'] is not None:raise ValueError('Undeclared objective weights')
    z=np.asarray(matrix@coef).ravel()+intercept;S=float(w.sum());residual=w*(expit(z)-y)
    value=float(np.dot(w,np.logaddexp(0,z)-y*z)/S+np.dot(coef,coef)/(2*spec['C']*S))
    gradient=np.r_[np.asarray(matrix.T@residual).ravel()/S+coef/(spec['C']*S),residual.sum()/S]
    if not np.isfinite(value) or not np.isfinite(gradient).all():raise ValueError('Nonfinite independent objective/gradient')
    return dict(objective=value,gradient=gradient.tolist(),gradient_max_abs=float(np.max(np.abs(gradient))),weight_sum=S)


def fallback_identity():
    return hashlib.sha256(inspect.getsource(NewtonSolver.fallback_lbfgs_solve).encode()).hexdigest()


@contextmanager
def observe_fallback():
    previous=sys.getprofile()
    if previous is not None:raise ValueError('Competing profiler present')
    target=NewtonSolver.fallback_lbfgs_solve.__code__
    result=dict(fallback_count=0,fallback_function_sha256=fallback_identity())
    def observer(frame,event,arg):
        if event=='call' and frame.f_code is target:result['fallback_count']+=1
    sys.setprofile(observer)
    try:yield result
    finally:sys.setprofile(previous)


class NumericalGate(ValueError):
    def __init__(self,message,diagnostics):
        super().__init__(message);self.diagnostics=diagnostics


def check_numerical(model,messages,record,computed):
    if messages or record['fallback_count']!=0 or record['fallback_function_sha256']!=fallback_identity():
        raise ValueError('Newton warning/fallback gate failed')
    if list(model.classes_)!=[0,1] or not 0<int(model.n_iter_.max())<100:
        raise ValueError('Newton class/iteration gate failed')
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():raise ValueError('Nonfinite Newton coefficients')
    if computed['gradient_max_abs']>1.1e-6:raise ValueError('Independent gradient gate failed')
    if record['iterations']!=int(model.n_iter_.max()) or record['warnings']!=messages:
        raise ValueError('Recorded numerical execution differs')
    if record['estimated_hessian_bytes']!=8*(model.n_features_in_+1)**2:
        raise ValueError('Recorded Hessian dimension differs')
    for k in ['objective','gradient','gradient_max_abs','weight_sum']:
        if not np.allclose(record[k],computed[k],atol=1e-12,rtol=1e-12):raise ValueError('Independent objective record differs')


def fit_checked(matrix,y,spec,threads=6,seed=2026):
    y=np.asarray(y)
    if set(np.unique(y))!={0,1} or matrix.shape[0]!=len(y) or not np.isfinite(matrix.data).all():
        raise ValueError('Invalid fitting population')
    model=LogisticRegression(**solver_parameters(spec,seed));numerical={};messages=[]
    try:
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=threads),observe_fallback() as observed:
            warnings.simplefilter('always')
            try:model.fit(matrix,y)
            finally:
                messages=[f'{w.category.__name__}: {w.message}' for w in caught]
                numerical.update(observed,warnings=messages)
        computed=objective_gradient(model,matrix,y,spec);numerical.update(computed,iterations=int(model.n_iter_.max()),
            estimated_hessian_bytes=8*(matrix.shape[1]+1)**2)
        check_numerical(model,messages,numerical,computed)
    except Exception as exc:
        raise NumericalGate(str(exc),numerical) from exc
    return model,messages,numerical


def initialize_preservation(c):
    guard(c);p=Path(c['output_dir'])/'preservation.json'
    if not p.exists():
        present={n:h for n,h in c['stopped_optional_hashes'].items() if Path(n).is_file()}
        write_json(p,dict(mandatory=c['preserve_sha256'],optional_present=present,
                         optional_absent=sorted(set(c['stopped_optional_hashes'])-set(present))))
    guard(c)


def verify_stopped_presence(expected,presence,mandatory):
    for p,h in expected.items():
        if Path(p).exists() and digest(p)!=h:raise ValueError('Stopped artifact changed: '+p)
    if presence.exists():
        captured=read_config(presence)
        if captured['mandatory']!=mandatory:raise ValueError('Mandatory preservation snapshot changed')
        if set(captured['optional_present'])|set(captured['optional_absent'])!=set(expected):
            raise ValueError('Stopped presence snapshot incomplete')
        if set(captured['optional_present'])&set(captured['optional_absent']):raise ValueError('Ambiguous stopped presence')
        for p,h in captured['optional_present'].items():
            if expected.get(p)!=h or not Path(p).is_file() or digest(p)!=h:raise ValueError('Captured stopped artifact changed: '+p)


def numerical_tables(c,frame,base):
    out=Path(c['output_dir']);rows=[];comparisons=[];old_c=read_config(stopped.CONFIG)
    for m in c['validation_months']:
        train=frame.index[frame.month<m]
        for spec in c['candidates']:
            r=read_config(paths(c,'validation',m,spec['id'])[0]);n=r['numerical']
            rows.append(dict(candidate=spec['id'],eval_month=m,**n))
            if m!=4:continue
            rp,pp=paths(old_c,'validation',m,spec['id']);mp=model_path(old_c,'validation',m,spec['id'])
            present=all(Path(p).is_file() for p in [rp,pp,mp])
            row=dict(candidate=spec['id'],eval_month=m,available=present)
            if present:
                a=joblib.load(mp);old_r=read_config(rp);new=joblib.load(model_path(c,'validation',m,spec['id']))
                if a['spec']!=spec or a['inputs'].metadata()!=new['inputs'].metadata():raise ValueError('Old-solver objective features differ')
                if old_r['train_ids_sha256']!=r['train_ids_sha256'] or old_r['evaluation_ids_sha256']!=r['evaluation_ids_sha256']:
                    raise ValueError('Old-solver population differs')
                if any(a['model'].get_params()[k]!=v for k,v in stopped.solver_parameters(spec,c['seed']).items()):
                    raise ValueError('Old-solver recipe differs')
                v=objective_gradient(a['model'],a['inputs'].transform(base.loc[train]),frame.loc[train,'fraud_bool'].to_numpy(),spec)
                row.update(old_objective=v['objective'],new_objective=n['objective'],objective_difference=n['objective']-v['objective'],
                           old_gradient_max_abs=v['gradient_max_abs'],new_gradient_max_abs=n['gradient_max_abs'],old_model_sha256=digest(mp))
            comparisons.append(row)
    return pd.DataFrame(rows),pd.DataFrame(comparisons)


def export_numerical(c,frame,base):
    numerical,comparisons=numerical_tables(c,frame,base)
    numerical.to_csv(Path(c['output_dir'])/'numerical.csv',index=False)
    comparisons.to_csv(Path(c['output_dir'])/'old_solver_comparisons.csv',index=False)


def audit_development(c,identity,frame,base):
    records=audit_grid(c,identity,frame,base)
    from .spline_report import audit_tables
    cf=crossfit(c);verify_table(cf,Path(c['output_dir'])/'crossfit.csv')
    verify_table(order_summary(cf),Path(c['output_dir'])/'summary.csv');audit_tables(c,cf,history(c))
    numbers=pd.read_csv(Path(c['output_dir'])/'numerical.csv')
    expected=[]
    for m in c['validation_months']:
        for spec in c['candidates']:
            r=read_config(paths(c,'validation',m,spec['id'])[0]);expected.append(dict(candidate=spec['id'],eval_month=m,**r['numerical']))
    verify_table(pd.DataFrame(expected),Path(c['output_dir'])/'numerical.csv')
    if len(numbers)!=36 or len(read_config(Path(c['output_dir'])/'feature_blocks.json'))!=36:raise ValueError('Incomplete numerical/features evidence')
    _,comparisons=numerical_tables(c,frame,base)
    verify_table(comparisons,Path(c['output_dir'])/'old_solver_comparisons.csv')
    blocks=read_config(Path(c['output_dir'])/'feature_blocks.json');coeff=[];diagnostics=[]
    for m in c['validation_months']:
        for spec in c['candidates']:
            a=joblib.load(model_path(c,'validation',m,spec['id']));inp=a['inputs'];model=a['model']
            if blocks[f"m{m}-{spec['id']}"]!=inp.metadata():raise ValueError('Feature block export differs')
            for population,ids in [('training',frame.index[frame.month<m]),('prediction',frame.index[frame.month==m])]:
                diagnostics.extend(dict(candidate=spec['id'],eval_month=m,population=population,**d) for d in inp.diagnostics(base.loc[ids]))
            coeff.extend(dict(candidate=spec['id'],eval_month=m,feature=n,coefficient=float(v)) for n,v in zip(inp.names,model.coef_[0]))
            coeff.append(dict(candidate=spec['id'],eval_month=m,feature='__INTERCEPT__',coefficient=float(model.intercept_[0])))
    verify_table(pd.DataFrame(coeff),Path(c['output_dir'])/'coefficients.csv')
    verify_table(pd.DataFrame(diagnostics),Path(c['output_dir'])/'feature_diagnostics.csv')
    guard(c);return records



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['miniature','validate','freeze','exploratory']);parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args();globals()[args.command](args.config)
