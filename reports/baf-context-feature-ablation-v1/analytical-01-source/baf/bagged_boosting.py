"""Isolated chronological bagged LightGBM study; prior identities stay immutable."""
from __future__ import annotations
import argparse
from itertools import product, combinations
import json
from pathlib import Path
import os, resource, shutil, subprocess, sys, time, warnings
import xml.etree.ElementTree as ET
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .common import CATEGORICAL, digest, read_config, write_json, load_data, features, make_model
from .improvements import paths, prediction
from .focal_study import expected_record, now
from .recall_extension import expected_population, order_summary
from .recall_weighted_metrics import POLICIES, aligned, learn, measure, decisions
from .improvement_metrics import confusion_metrics
from .spline_inputs import AdditiveInputs
from .interaction_inputs import storage
from . import unweighted_fusion_study as previous
from .unweighted_fusion_scores import lookup as old_lookup, compose as old_compose
from .bagged_boosting_scores import (SEEDS, WEIGHTINGS, EPSILON, REFERENCE_ROLES,
    REFERENCE_POLICIES, REFERENCE_BASES, recipes, strategies, lookup, probability,
    compose, choices_for, dependencies, contrast_factors, correlations, overlap,
    clipping, declaration_counts)

CONFIG='configs/bagged-lightgbm-seeds.json'
SOURCES=previous.SOURCES+['baf/bagged_boosting.py','baf/bagged_boosting_scores.py','baf/bagged_boosting_report.py']
COUNTS=declaration_counts()
TABLES=['crossfit.csv','summary.csv','pooled.csv','contrasts.csv','score_diagnostics.csv',
        'clipping.csv','seed_diversity.csv','decision_overlap.csv','model_diagnostics.csv',
        'preprocessing.json','reference_diagnostics.json']
verify_hashes=previous.verify_hashes
cache_state=previous.tree.cache_state


def fingerprint(path=CONFIG):
    result=previous.fingerprint(path);result['sources']={p:digest(p) for p in SOURCES}
    return result


def mandatory_preservation(source):
    result=dict(source['preserve_sha256'])
    for p in previous.SOURCES[-3:]+[previous.CONFIG,source['protocol_path'],
             str(Path(source['output_dir'])/'selection.json'),str(Path(source['report_dir'])/'report_hashes.json')]:
        result[p]=digest(p)
    result.update(read_config(Path(source['report_dir'])/'report_hashes.json'))
    return result


def guard(c):
    source=read_config(c['parent_config']);previous.guard(source)
    if c['parent_config']!=previous.CONFIG or previous.fingerprint()!=c['parent_fingerprint']:
        raise ValueError('Previous study identity changed')
    preserved=mandatory_preservation(source)
    if c['preserve_sha256']!=preserved or len(preserved)!=634:raise ValueError('Preservation union changed')
    verify_hashes(preserved)
    selections=[p for p in preserved if p.startswith('outputs/') and p.endswith('/selection.json')]
    if len(selections)!=15 or any(preserved.get(p.replace('outputs/','reports/',1))!=preserved[p] for p in selections):
        raise ValueError('Fifteen selection pairs required')
    changed={'preserve_sha256','source_paths','strategies','expected_counts','output_dir','report_dir',
             'protocol_path','historical_output_roots','candidates','lambdas','anchors','added_members'}
    if any(c.get(k)!=v for k,v in source.items() if k not in changed):raise ValueError('Frozen reference contract changed')
    expected=dict(bagged_presets=recipes(),strategies=strategies(),source_paths=SOURCES,
        expected_counts=COUNTS,candidates=[s for s in source['candidates'] if s['id'] in REFERENCE_BASES],
        original_base_config='configs/study.json',bagging_fraction=.8,bagging_frequency=1,bagging_seeds=SEEDS,
        output_dir='outputs/baf-bagged-lightgbm-seeds-v1',report_dir='reports/baf-bagged-lightgbm-seeds-v1',
        protocol_path='docs/bagged-lightgbm-seeds-protocol.md',
        historical_output_roots=sorted(set(source['historical_output_roots']+[source['output_dir']])))
    if len(set(SOURCES))!=52 or any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Prospective finite declaration changed')
    if any(k in c for k in ['primary_fraction','precision_floor','review_quota']):raise ValueError('Undeclared operating constraint')
    dest=[Path(c[k]).resolve() for k in ['output_dir','report_dir']]
    for a,b in combinations(dest+[Path(p).resolve() for p in c['historical_output_roots']],2):
        if (a in dest or b in dest) and (a==b or a in b.parents or b in a.parents):raise ValueError('Destination overlap')
    for name in ['preimplementation_preservation.json','preservation.json']:
        p=Path(c['output_dir'])/name
        if p.exists():
            record=read_config(p);verify_hashes(record if name.startswith('preimplementation') else record['optional_present'])
            if name=='preservation.json' and (record['mandatory']!=preserved or record['roots']!=c['historical_output_roots']):
                raise ValueError('Preservation snapshot declaration changed')


def initialize(config_path=CONFIG):
    c=read_config(config_path);guard(c);fp=fingerprint(config_path);out=Path(c['output_dir'])
    p=out/'preservation.json'
    if not p.exists():
        initial=out/'preimplementation_preservation.json'
        present=read_config(initial) if initial.exists() else {str(p):digest(p) for root in c['historical_output_roots'] for p in Path(root).rglob('*') if p.is_file()}
        write_json(p,dict(mandatory=c['preserve_sha256'],optional_present=present,roots=c['historical_output_roots']))
    record=dict(fingerprint=fp,config_path=str(config_path),protocol_sha256=digest(c['protocol_path']),
                preservation_sha256=digest(p),strategies=c['strategies'],expected_counts=COUNTS,month7_previously_consumed=True)
    p=out/'declaration.json'
    if p.exists():
        if any(read_config(p).get(k)!=v for k,v in record.items()):raise ValueError('Frozen execution declaration changed')
    else:write_json(p,dict(**record,created_at_utc=now()))
    return c,fp


def reference_specs(c):return c['candidates']+c['references']+c['presets']


def source_origins(c,spec):
    source=read_config(c['parent_config'])
    return [(source,c['parent_fingerprint'])]+previous.source_origins(source,spec)


def prepare_reference(c,fp,frame,base,spec,month,phase='validation',allow_missing=True):
    if spec not in reference_specs(c):raise ValueError('Undeclared reference recipe')
    own=previous.cache_paths(c,spec,month,phase)
    if cache_state(own):return previous.audit_base(c,fp,frame,base,spec,month,phase)
    if not allow_missing:raise ValueError('Required reference cache missing')
    failure=Path(c['output_dir'])/'failures'/(own[0].stem+'.json')
    if failure.exists():raise ValueError('Prior failed reference gate retained')
    for source,identity in source_origins(c,spec):
        origin=previous.cache_paths(source,spec,month,phase)
        if not cache_state(origin):continue
        r=previous.audit_base(source,identity,frame,base,spec,month,phase)
        for src,dst in zip(origin[1:],own[1:]):
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        r=dict(r,fingerprint=fp,reused=True,origin_artifacts={str(p):digest(p) for p in origin})
        if spec.get('representation')=='spline':
            r.update(original_fit_fingerprint=c['newton_fit_fingerprint'],original_analysis_fingerprint=c['newton_analysis_fingerprint'])
        elif spec['id']=='xgb_d4_w1':r['model_path']=str(own[2])
        else:r.update(reference_model_path=str(own[2]),reference_model_new_fit=False)
        write_json(own[0],r)
        return previous.audit_base(c,fp,frame,base,spec,month,phase)
    try:
        if spec.get('representation')=='spline':previous.newton.fit_new(c,fp,frame,base,spec,month,phase)
        elif spec['id']=='xgb_d4_w1':previous.tree.old.fit_model(c,fp,frame,base,spec,month,phase)
        else:previous.additive.reproduce_reference(c,fp,frame,base,spec,month,phase)
        return previous.audit_base(c,fp,frame,base,spec,month,phase)
    except Exception as exc:
        if not failure.exists():write_json(failure,dict(**expected_record(frame,spec,month,phase,fp),error=repr(exc),created_at_utc=now()))
        raise


def model_path(c,phase,month,name):
    return Path(c['output_dir'])/'models'/f'{phase}-m{month}-{name}-bagged.joblib'


def bagged_paths(c,phase,month,name):return [*paths(c,phase,month,name),model_path(c,phase,month,name)]


def factory(c,spec,columns,*,active=True):
    if spec not in recipes():raise ValueError('Undeclared bagged recipe')
    source=read_config(c['original_base_config']);source=dict(source,threads=c['threads'])
    result=make_model('lightgbm',spec['weighting']=='balanced',source,columns)
    result.set_params(model__subsample=spec['fraction'] if active else 1.,
                      model__subsample_freq=spec['frequency'] if active else 0,
                      model__bagging_seed=spec['bagging_seed'])
    return result


def preprocessing_metadata(pipeline):
    pre=pipeline.named_steps['preprocess'];num=pre.named_transformers_['numeric'].named_steps['impute'];cat=pre.named_transformers_['categorical']
    return dict(input_columns=list(pre.feature_names_in_),numeric_statistics=num.statistics_.tolist(),
                numeric_missing_indicators=num.indicator_.features_.tolist(),
                categories=[x.tolist() for x in cat.categories_],feature_names=pre.get_feature_names_out().tolist(),
                transformed_dimension=len(pre.get_feature_names_out()))


def audit_preprocessing(pipeline,x):
    pre=pipeline.named_steps['preprocess'];numeric=[n for n in x.columns if n not in CATEGORICAL]
    if list(pre.feature_names_in_)!=list(x.columns) or pre.sparse_threshold!=1. or list(pre.transformers_[0][2])!=numeric or list(pre.transformers_[1][2])!=CATEGORICAL:
        raise ValueError('Original feature roles changed')
    num=pre.named_transformers_['numeric'];imp=num.named_steps['impute'];cat=pre.named_transformers_['categorical']
    if list(num.named_steps)!=['impute'] or imp.strategy!='median' or not imp.add_indicator or not imp.keep_empty_features:
        raise ValueError('Original numeric preprocessing changed')
    expected=np.array([0. if x[n].isna().all() else float(x[n].median()) for n in numeric])
    if not np.array_equal(expected,imp.statistics_) or not np.array_equal(np.flatnonzero(x[numeric].isna().any().to_numpy()),imp.indicator_.features_):
        raise ValueError('Train-only numeric statistics changed')
    if cat.handle_unknown!='ignore' or not cat.sparse_output or cat.drop is not None:
        raise ValueError('Original categorical handling changed')
    if len(cat.categories_)!=len(CATEGORICAL) or any(not np.array_equal(np.sort(x[n].unique()),actual) for n,actual in zip(CATEGORICAL,cat.categories_)):
        raise ValueError('Train-only categorical vocabulary changed')
    return preprocessing_metadata(pipeline)


def effective_parameters(model,spec,threads,*,active=True):
    p=model.booster_.params
    expected=dict(subsample=spec['fraction'] if active else 1.,subsample_freq=spec['frequency'] if active else 0,
                  bagging_seed=spec['bagging_seed'],random_state=2026,deterministic=True,force_col_wise=True,
                  num_threads=threads,num_iterations=200,num_leaves=15,learning_rate=.05,min_child_samples=100,
                  colsample_bytree=1.,objective='binary')
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Effective booster recipe changed')
    for primary,alias in [('bagging_fraction','subsample'),('bagging_freq','subsample_freq'),('feature_fraction','colsample_bytree')]:
        if primary in p and p[primary]!=expected[alias]:raise ValueError('Conflicting effective parameter aliases')
    if p.get('data_sample_strategy','bagging')!='bagging' or p.get('pos_bagging_fraction',1.)!=1. or p.get('neg_bagging_fraction',1.)!=1.:
        raise ValueError('Undeclared sampling strategy')
    return p


def audit_bagged(c,fp,frame,base,spec,month,phase='validation'):
    ps=bagged_paths(c,phase,month,spec['id'])
    if not cache_state(ps):raise ValueError('Required bagged cache missing')
    r=read_config(ps[0]);expected=expected_record(frame,spec,month,phase,fp)
    if any(r.get(k)!=v for k,v in expected.items()) or digest(ps[1])!=r['prediction_sha256'] or digest(ps[2])!=r['model_sha256']:
        raise ValueError('Bagged training/serialized identity changed')
    if r['warnings']:raise ValueError('Bagged warnings retained')
    artifact=joblib.load(ps[2]);pipeline=artifact['model'];train=frame.index[frame.month<month]
    signature=previous.tree.parameter_signature
    if artifact['spec']!=spec or signature(pipeline)!=signature(factory(c,spec,base.columns)):
        raise ValueError('Bagged estimator recipe changed')
    if r['requested_model_params']!=signature(pipeline.named_steps['model']):raise ValueError('Requested model parameters changed')
    if r['preprocessing']!=audit_preprocessing(pipeline,base.loc[train]):raise ValueError('Preprocessing record changed')
    model=pipeline.named_steps['model'];effective=effective_parameters(model,spec,c['threads'])
    if r['booster_params']!=effective or r['fitted_tree_count']!=model.booster_.current_iteration() or not 1<=r['fitted_tree_count']<=200:
        raise ValueError('Fitted booster metadata changed')
    y=frame.loc[train,'fraud_bool'].to_numpy();n=len(y);fraud=int(y.sum())
    weights={'0':n/(2*(n-fraud)),'1':n/(2*fraud)} if spec['weighting']=='balanced' else {'0':1.,'1':1.}
    if r['class_weights']!=weights:raise ValueError('Whole-training class weights changed')
    d=prediction(c,phase,month,spec['id']);aligned(d,expected_population(frame,month));probability(d['scores'])
    if list(pipeline.classes_)!=[0,1]:raise ValueError('Class orientation changed')
    with threadpool_limits(limits=c['threads']):
        actual=pipeline.predict_proba(base.loc[d['row_ids']])[:,1]
    if not np.allclose(actual,d['scores'],atol=1e-12,rtol=0):raise ValueError('Saved bagged inference changed')
    return r


def fit_bagged(c,fp,frame,base,spec,month,phase='validation',allow_missing=True):
    if spec not in recipes():raise ValueError('Undeclared source')
    ps=bagged_paths(c,phase,month,spec['id'])
    if cache_state(ps):return audit_bagged(c,fp,frame,base,spec,month,phase)
    if not allow_missing:raise ValueError('Required bagged cache absent')
    failure=Path(c['output_dir'])/'failures'/(ps[0].stem+'.json')
    if failure.exists():raise ValueError('Prior failed bagged gate retained; no automatic retry')
    record=expected_record(frame,spec,month,phase,fp);start=time.perf_counter()
    try:
        train=frame.index[frame.month<month];ev=frame.index[frame.month==month];pipeline=factory(c,spec,base.columns)
        y=frame.loc[train,'fraud_bool'].to_numpy()
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=c['threads']):
            warnings.simplefilter('always');pipeline.fit(base.loc[train],y)
            values=pipeline.predict_proba(base.loc[ev])[:,1]
        messages=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if messages:raise ValueError('Bagged fit warning: '+repr(messages))
        probability(values);model=pipeline.named_steps['model'];effective=effective_parameters(model,spec,c['threads'])
        meta=audit_preprocessing(pipeline,base.loc[train]);sample=pipeline.named_steps['preprocess'].transform(base.loc[train[:8192]])
        for p in ps:p.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(dict(spec=spec,model=pipeline),ps[2])
        np.savez_compressed(ps[1],row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=values)
        n=len(y);fraud=int(y.sum());class_weights={'0':n/(2*(n-fraud)),'1':n/(2*fraud)} if spec['weighting']=='balanced' else {'0':1.,'1':1.}
        write_json(ps[0],dict(**record,prediction_sha256=digest(ps[1]),model_sha256=digest(ps[2]),
            reused=False,warnings=messages,requested_model_params=previous.tree.parameter_signature(model),
            booster_params=effective,preprocessing=meta,class_weights=class_weights,
            fitted_tree_count=model.booster_.current_iteration(),sample_matrix=storage(sample),
            fit_seconds=time.perf_counter()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            created_at_utc=now()))
        return audit_bagged(c,fp,frame,base,spec,month,phase)
    except Exception as exc:
        if not failure.exists():write_json(failure,dict(**record,error=repr(exc),elapsed_seconds=time.perf_counter()-start,created_at_utc=now()))
        raise


def derived(c,fp,name,month,phase='validation',allow_missing=True):
    old=name in dependencies(list(REFERENCE_POLICIES))['derived']
    spec=old_lookup(name) if old else lookup(name)
    items={n:prediction(c,phase,month,n) for n in spec['members']}
    d=old_compose(spec,items) if old else compose(spec,items)
    deps={n:{str(p):digest(p) for p in paths(c,phase,month,n)} for n in spec['members']}
    expected=dict(candidate=name,phase=phase,eval_month=month,fingerprint=fp,spec=spec,
                  dependencies=deps,epsilon=EPSILON,arithmetic='preserved reference' if old else 'ordered float64 sum / 3')
    rp,pp=paths(c,phase,month,name)
    if cache_state([rp,pp]):
        r=read_config(rp);saved=prediction(c,phase,month,name);aligned(d,saved)
        if any(r.get(k)!=v for k,v in expected.items()) or digest(pp)!=r['prediction_sha256'] or not np.array_equal(d['scores'],saved['scores']):
            raise ValueError('Derived score identity/arithmetic changed')
        verify_hashes(r['origin_artifacts']);return r
    if not allow_missing:raise ValueError('Required derived cache missing')
    origin={}
    if old:
        source=read_config(c['parent_config']);sr,sp=paths(source,phase,month,name)
        if cache_state([sr,sp]):
            r=read_config(sr);original=prediction(source,phase,month,name);aligned(d,original)
            expected_deps={n:{str(p):digest(p) for p in paths(source,phase,month,n)} for n in spec['members']}
            if r['fingerprint']!=c['parent_fingerprint'] or r['spec']!=spec or r['dependencies']!=expected_deps or r['prediction_sha256']!=digest(sp) or not np.array_equal(d['scores'],original['scores']):
                raise ValueError('Preserved original reference arithmetic/provenance differs')
            verify_hashes(r.get('origin_artifacts',{}))
            origin={str(p):digest(p) for p in [sr,sp]}
            for member in expected_deps.values():origin.update(member)
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**d)
    result=dict(**expected,prediction_sha256=digest(pp),reused=bool(origin),origin_artifacts=origin)
    write_json(rp,result);return result


def prepare(c,fp,frame,base,phase='validation',names=None,allow_missing=True):
    names=[s['id'] for s in strategies()] if names is None else names;dag=dependencies(names)
    specs={s['id']:s for s in reference_specs(c)+recipes()};records=[]
    old_order=[s['id'] for s in previous.strategies() if s['id'] in dag['derived']]
    new_order=[s['id'] for s in strategies() if s['kind']=='aggregate' and s['id'] in dag['derived']]
    for m in c['validation_months'] if phase=='validation' else [7]:
        for n in sorted(set(dag['bases'])&set(REFERENCE_BASES))+dag['bagged_bases']:
            print('Preparing',phase,m,n,flush=True)
            method=fit_bagged if n.startswith('bagged_') else prepare_reference
            records.append(method(c,fp,frame,base,specs[n],m,phase,allow_missing))
        for n in old_order+new_order:derived(c,fp,n,m,phase,allow_missing)
    return records


def history(c):
    return {s['id']:{m:prediction(c,'validation',m,s['id']) for m in c['validation_months']} for s in strategies()}


def crossfit(c):
    h=history(c);rows=[]
    for m,n,k in product([5,6],h,POLICIES):
        aligned(h[n][m],h[next(iter(REFERENCE_POLICIES))][m])
        rule=learn({t:d for t,d in h[n].items() if t<m},k,m)
        rows.append(dict(candidate=n,policy=k,eval_month=m,source_months=','.join(map(str,rule['source_months'])),
                         parameters=json.dumps(rule,sort_keys=True),**measure(h[n][m],rule)))
    return pd.DataFrame(rows)


def tables(c,cf):
    summary=order_summary(cf).reset_index(drop=True)
    auc=cf.groupby(['candidate','policy'])['roc_auc'].mean()
    summary['roc_auc']=[float(auc.loc[(r.candidate,r.policy)]) for r in summary.itertuples()]
    index=summary.set_index(['candidate','policy']);contrasts=[];pooled=[]
    for label,n,k,ref,rk in contrast_factors():
        a,b=index.loc[(n,k)],index.loc[(ref,rk)]
        contrasts.append(dict(contrast=label,candidate=n,policy=k,reference=ref,reference_policy=rk,
            **{v+'_difference':float(a[v]-b[v]) for v in ['f2','recall','precision','flagged_fraction','average_precision']}))
    for (n,k),g in cf.groupby(['candidate','policy']):
        t=g[['tn','fp','fn','tp']].sum()
        pooled.append(dict(candidate=n,policy=k,**confusion_metrics(*[int(t[v]) for v in ['tn','fp','fn','tp']]),flagged_fraction=float((t.tp+t.fp)/g.n.sum())))
    h=history(c);score=[];clips=[];div=[];joint=[];models=[];pre={};refmeta={}
    for n,months in h.items():
        for m,d in months.items():
            p=d['scores'];score.append(dict(candidate=n,eval_month=m,n=len(p),score_min=float(p.min()),score_max=float(p.max()),score_mean=float(p.mean()),unique_scores=len(np.unique(p)),tied_fraction=1-len(np.unique(p))/len(p)))
    for m,n in product([4,5,6],dependencies(list(h))['bases']):
        r=read_config(paths(c,'validation',m,n)[0]);clips.append(dict(candidate=n,eval_month=m,**clipping(prediction(c,'validation',m,n)['scores'])))
        if n.startswith('bagged_'):
            pre[f'{m}:{n}']=r['preprocessing'];models.append(dict(candidate=n,eval_month=m,weighting=r['spec']['weighting'],bagging_seed=r['spec']['bagging_seed'],fitted_tree_count=r['fitted_tree_count'],fit_seconds=r['fit_seconds'],peak_process_rss_bytes=r['peak_process_rss_bytes'],transformed_dimension=r['preprocessing']['transformed_dimension'],train_rows=r['train_rows'],train_fraud=r['train_fraud'],class_weights=json.dumps(r['class_weights'],sort_keys=True),warnings=json.dumps(r['warnings']),requested_model_params=json.dumps(r['requested_model_params'],sort_keys=True),booster_params=json.dumps(r['booster_params'],sort_keys=True)))
        else:refmeta[f'{m}:{n}']=r
    for w,(s,t),m in product(WEIGHTINGS,combinations(SEEDS,2),[4,5,6]):
        a,b=h[f'bagged_{w}_s{s}'][m],h[f'bagged_{w}_s{t}'][m]
        div.append(dict(weighting=w,first_seed=s,second_seed=t,eval_month=m,**correlations(a['scores'],b['scores'])))
        if m>4:
            for k in POLICIES:
                first,second=f'bagged_{w}_s{s}',f'bagged_{w}_s{t}'
                pa=decisions(a,learn({i:d for i,d in h[first].items() if i<m},k,m));pb=decisions(b,learn({i:d for i,d in h[second].items() if i<m},k,m))
                joint.append(dict(weighting=w,first_seed=s,second_seed=t,eval_month=m,policy=k,**overlap(a,b,pa,pb)))
    result=dict(crossfit=cf,summary=summary,pooled=pd.DataFrame(pooled),contrasts=pd.DataFrame(contrasts),
                score_diagnostics=pd.DataFrame(score),clipping=pd.DataFrame(clips),seed_diversity=pd.DataFrame(div),
                decision_overlap=pd.DataFrame(joint),model_diagnostics=pd.DataFrame(models))
    return result,pre,refmeta


def record_checks(config_path=CONFIG):
    c=read_config(config_path);guard(c);dest=Path(c['report_dir']);xml=dest/'tests.xml'
    if (Path(c['output_dir'])/'selection.json').exists():raise ValueError('Frozen checks cannot be overwritten')
    root=ET.parse(xml).getroot();cases=list(root.iter('testcase'))
    if not cases or any(list(root.iter(n)) for n in ['failure','error','skipped']):raise ValueError('Full passing methodological suite required')
    own=[t.get('name','') for t in cases if t.get('classname','').endswith('test_bagged_boosting')]
    if sum(n.startswith('test_new_bagged_cache_format[') for n in own)!=2 or sum(n.startswith('test_every_winner_policy[') for n in own)!=45:
        raise ValueError('Two genuine formats and all45 winner/policy tests required')
    receipt=dict(fingerprint=fingerprint(config_path),junit_sha256=digest(xml),tests_passed=len(cases),own_tests=len(own),
                 new_cache_test_fits=2,historical_test_fits_separate=True,test_source_sha256=digest('tests/test_bagged_boosting.py'),created_at_utc=now())
    p=dest/'test_fit_counts.json';invocations=read_config(p)['invocations'] if p.exists() else []
    if not any(v['junit_sha256']==receipt['junit_sha256'] for v in invocations):invocations.append(receipt)
    write_json(p,dict(invocations=invocations,total_new_cache_test_fits=sum(v['new_cache_test_fits'] for v in invocations)))
    write_json(Path(c['output_dir'])/'checks.json',receipt)
    return receipt


def execution_gate(c,fp,*,miniature=True):
    out=Path(c['output_dir'])
    names=['checks.json','source_miniature_gate.json','resource_preflight.json']+(['miniature_gate.json'] if miniature else [])
    for n in names:
        if read_config(out/n)['fingerprint']!=fp:raise ValueError('Missing/stale gate '+n)
    checks=read_config(out/'checks.json')
    if checks['test_source_sha256']!=digest('tests/test_bagged_boosting.py') or checks['junit_sha256']!=digest(Path(c['report_dir'])/'tests.xml'):
        raise ValueError('Execution test evidence changed')


def preflight(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir']);checks=read_config(out/'checks.json')
    if checks['fingerprint']!=fp or checks['test_source_sha256']!=digest('tests/test_bagged_boosting.py') or checks['junit_sha256']!=digest(Path(c['report_dir'])/'tests.xml'):
        raise ValueError('Resource gate requires current full tests')
    previous.source_miniature_gate(c,fp)
    p=out/'resource_preflight.json'
    if p.exists():
        if read_config(p)['fingerprint']!=fp:raise ValueError('Resource identity changed')
        return read_config(p)
    failure=out/'failures'/'resource-preflight.json'
    if failure.exists():raise ValueError('Prior resource failure retained')
    start=time.perf_counter()
    try:
        frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
        inp=AdditiveInputs('spline').fit(base);sample=inp.transform(base.iloc[:8192]);n,d=len(base),len(inp.names)
        pipeline=factory(c,recipes()[0],base.columns);pre=pipeline.named_steps['preprocess'].fit(base)
        oldsample=pre.transform(base.iloc[:8192]);original_dimension=oldsample.shape[1]
        maximum=max(d,original_dimension);matrix=12*n*maximum+8*(n+1);workspace=4*matrix+8*(maximum+1)**2+4*1024**3
        ram=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)) if sys.platform=='darwin' else os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
        result=dict(fingerprint=fp,host_ram_bytes=ram,training_rows=n,training_months=list(range(7)),
            additive_dimension=d,original_tree_dimension=original_dimension,max_dimension=maximum,max_chunk_rows=8192,
            additive_sample=storage(sample),original_tree_sample=storage(oldsample),matrix_upper_bytes=matrix,
            workspace_upper_bytes=workspace,allowance_fraction=.7,predictor_only=True,
            elapsed_seconds=time.perf_counter()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if workspace>ram*.7:raise ValueError('Largest required allocation exceeds70% hostRAM')
        write_json(p,result);return result
    except Exception as exc:
        write_json(failure,dict(fingerprint=fp,error=repr(exc),created_at_utc=now()));raise


def synthetic_data(n=4500):
    rng=np.random.default_rng(808);x=rng.normal(size=n);z=rng.normal(size=n)
    base=pd.DataFrame({'signal':x,'noise':z})
    base.loc[np.arange(n)%23==0,'noise']=np.nan
    for i,nm in enumerate(CATEGORICAL):base[nm]=np.where(np.arange(n)%(i+3)==0,'a','b')
    labels=(x+z*.7+rng.normal(size=n)>1.9).astype(int)
    frame=pd.DataFrame({'month':np.repeat([0,1,2],n//3),'fraud_bool':labels},index=base.index)
    return frame,base


def miniature(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp,miniature=False);out=Path(c['output_dir']);p=out/'miniature_gate.json'
    if p.exists():
        r=read_config(p)
        if r['fingerprint']!=fp or r['fit_count']!=9:raise ValueError('Miniature gate changed')
        verify_hashes(r['files']);return r
    failure=out/'failures'/'miniature.json'
    if failure.exists():raise ValueError('Prior miniature failure retained')
    frame,base=synthetic_data();train=frame.index[frame.month<2];ev=frame.index[frame.month==2];results=[];predictions={}
    try:
        requests=[(s['id'],s,True) for s in recipes()]+[('repeat',recipes()[0],True),('inactive17',recipes()[0],False),('inactive43',recipes()[2],False)]
        for tag,spec,active in requests:
            model=factory(c,spec,base.columns,active=active)
            with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=c['threads']):
                warnings.simplefilter('always');model.fit(base.loc[train],frame.loc[train,'fraud_bool']);values=model.predict_proba(base.loc[ev])[:,1]
            messages=[str(w.message) for w in caught]
            if messages:raise ValueError('Miniature warning '+repr(messages))
            effective_parameters(model.named_steps['model'],spec,c['threads'],active=active)
            audit_preprocessing(model,base.loc[train]);probability(values)
            mp=out/'miniature'/(tag+'.joblib');pp=out/'miniature'/(tag+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,mp);np.savez_compressed(pp,scores=values)
            with threadpool_limits(limits=c['threads']):restored=joblib.load(mp).predict_proba(base.loc[ev])[:,1]
            if not np.allclose(values,restored,atol=1e-12,rtol=0):raise ValueError('Miniature persistence changed')
            predictions[tag]=values;results.append(dict(tag=tag,active=active,spec=spec,files={str(mp):digest(mp),str(pp):digest(pp)}))
            write_json(out/'miniature_progress.json',dict(fingerprint=fp,fit_count=len(results),results=results))
        if not np.array_equal(predictions['repeat'],predictions[recipes()[0]['id']]):raise ValueError('Same seed is not repeatable')
        if not np.array_equal(predictions['inactive17'],predictions['inactive43']):raise ValueError('Inactive sampling unexpectedly changes seeds')
        if any(all(np.array_equal(predictions[f'bagged_{w}_s17'],predictions[f'bagged_{w}_s{s}']) for s in [29,43]) for w in WEIGHTINGS):
            raise ValueError('Designed fixture has no active row-sampling diversity')
        result=dict(fingerprint=fp,fit_count=9,results=results,files={p:h for r in results for p,h in r['files'].items()},
                    same_seed_repeatable=True,inactive_seed_controls_equal=True,active_seed_diversity=True,created_at_utc=now())
        write_json(p,result);return result
    except Exception as exc:
        write_json(failure,dict(fingerprint=fp,error=repr(exc),completed_fits=len(results),results=results,created_at_utc=now()));raise


def prepare_development(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Development is frozen')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter();records=prepare(c,fp,frame,base)
    references=[r for r in records if r['candidate'] in REFERENCE_BASES];new=[r for r in records if r['candidate'].startswith('bagged_')]
    derived_records=[read_config(paths(c,'validation',m,n)[0]) for m,n in product([4,5,6],dependencies([s['id'] for s in strategies()])['derived'])]
    result=dict(fingerprint=fp,base_results=len(records),new_bagged_fits=len(new),reference_results=len(references),
                reference_reuses=sum(bool(r.get('reused')) for r in references),reference_reproduction_fits=sum(not r.get('reused',False) for r in references),
                derived_results=len(derived_records),derived_reuses=sum(r['reused'] for r in derived_records),
                new_derived_computations=sum(not r['reused'] for r in derived_records),
                elapsed_seconds=time.perf_counter()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if (len(records),len(new),len(references),len(derived_records))!=(33,18,15,21):raise ValueError('Complete preparation set differs')
    write_json(out/'preparation.json',result);return result


def validate(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Frozen validation cannot be overwritten')
    if read_config(out/'preparation.json')['fingerprint']!=fp:raise ValueError('Preparation missing/stale')
    cf=crossfit(c);generated,pre,refs=tables(c,cf)
    for n,table in generated.items():table.to_csv(out/(n+'.csv'),index=False)
    write_json(out/'preprocessing.json',pre);write_json(out/'reference_diagnostics.json',refs)
    write_json(out/'validation_complete.json',dict(fingerprint=fp,created_at_utc=now(),files={n:digest(out/n) for n in TABLES}))
    from .bagged_boosting_report import audit_development
    audit_development(config_path)
    print(generated['summary'].head(15).to_string(index=False),flush=True)


def final_fit_bounds(c,choices):
    dag=dependencies([n for n,k in choices.values()]);available={};missing={'spline':0,'tree':0}
    for spec in reference_specs(c):
        if spec['id'] not in dag['bases']:continue
        found=None
        for source,identity in source_origins(c,spec):
            ps=previous.cache_paths(source,spec,7,'exploratory')
            if cache_state(ps):found={str(p):digest(p) for p in ps};break
        available[spec['id']]=found
        if found is None:missing['spline' if spec.get('representation')=='spline' else 'tree']+=1
    return dict(unique_base_results=len(dag['bases']),physical_scores=len(dag['scores']),max_new_bagged_fits=len(dag['bagged_bases']),
                max_reproduced_additive_fits=missing['spline'],max_reproduced_tree_fits=missing['tree'],original_availability=available)


def freeze(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    done=read_config(out/'validation_complete.json');audit=read_config(out/'independent_development_audit.json')
    if done['fingerprint']!=fp or audit['fingerprint']!=fp or audit['validation_sha256']!=digest(out/'validation_complete.json'):
        raise ValueError('Independent prefreeze audit missing/stale')
    verify_hashes({str(out/p):h for p,h in done['files'].items()})
    winner=pd.read_csv(out/'summary.csv').iloc[0];choices=choices_for(str(winner.candidate),str(winner.policy));h=history(c)
    write_json(out/'selection.json',dict(created_at_utc=now(),fingerprint=fp,config=c,selected_candidate=str(winner.candidate),
        selected_policy=str(winner.policy),selected_development_f2=float(winner.f2),choices=choices,
        rules={a:learn(h[n],k,7) for a,(n,k) in choices.items()},score_dependencies=dependencies([n for n,k in choices.values()]),
        final_fit_bounds=final_fit_bounds(c,choices),validation_files=done['files'],
        independent_audit_sha256=digest(out/'independent_development_audit.json'),
        artifacts={str(p):digest(p) for folder in ['runs','predictions','models'] for p in (out/folder).glob('*') if p.is_file()},
        month7_previously_consumed=True,exploratory=True))
    print('Frozen',winner.candidate,winner.policy,float(winner.f2),'roles',len(choices),flush=True)


def load_selection(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['fingerprint']!=fp or s['config']!=c:raise ValueError('Frozen execution identity changed')
    verify_hashes(s['artifacts']);verify_hashes({str(out/p):h for p,h in s['validation_files'].items()})
    if s['independent_audit_sha256']!=digest(out/'independent_development_audit.json'):raise ValueError('Frozen audit changed')
    w=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0];roles=choices_for(str(w.candidate),str(w.policy));h=history(c)
    if (s['selected_candidate'],s['selected_policy'])!=(str(w.candidate),str(w.policy)) or not np.isclose(s['selected_development_f2'],w.f2,atol=1e-12,rtol=0):raise ValueError('Actual macro winner differs')
    if s['choices']!=roles or s['score_dependencies']!=dependencies([n for n,k in roles.values()]) or s['final_fit_bounds']!=final_fit_bounds(c,roles):raise ValueError('Frozen roles/DAG/bounds differ')
    if any(s['rules'][a]!=learn(h[n],k,7) for a,(n,k) in roles.items()):raise ValueError('Final earlier-only rules differ')
    return c,s


def final_data(c,s):return {a:prediction(c,'exploratory',7,n) for a,(n,k) in s['choices'].items()}


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);sha=digest(out/'selection.json');p=out/'exploratory_access.json'
    if not p.exists():write_json(p,dict(access_at_utc=now(),selection_sha256=sha,month7_previously_consumed=True))
    access=read_config(p)
    if access['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Freeze/access chronology differs')
    if (out/'transfer_complete.json').exists():raise ValueError('Completed transfer cannot be overwritten')
    frame=load_data(c);base=features(frame);records=prepare(c,s['fingerprint'],frame,base,'exploratory',{n for n,k in s['choices'].values()})
    data=final_data(c,s);rows=[]
    for a,d in data.items():
        aligned(d,data['selected']);rule=s['rules'][a];rows.append(dict(role=a,candidate=s['choices'][a][0],policy=s['choices'][a][1],eval_month=7,parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    new=[r for r in records if r['candidate'].startswith('bagged_')];refs=[r for r in records if r['candidate'] in REFERENCE_BASES]
    counts=dict(base_results=len(records),new_bagged_fits=len(new),reference_reuses=sum(bool(r.get('reused')) for r in refs),
                reproduced_additive_fits=sum(not r.get('reused',False) and r['spec'].get('representation')=='spline' for r in refs),
                reproduced_tree_fits=sum(not r.get('reused',False) and r['spec'].get('representation')!='spline' for r in refs),
                role_count=len(rows),total_scores=len(list((out/'predictions').glob('exploratory-*.npz'))))
    bounds=s['final_fit_bounds']
    for key,limit in [('new_bagged_fits','max_new_bagged_fits'),('reproduced_additive_fits','max_reproduced_additive_fits'),('reproduced_tree_fits','max_reproduced_tree_fits')]:
        if counts[key]>bounds[limit]:raise ValueError('Frozen fit bound exceeded')
    if counts['base_results']!=bounds['unique_base_results'] or counts['total_scores']!=bounds['physical_scores']:raise ValueError('Transfer DAG work differs')
    write_json(out/'transfer_complete.json',dict(fingerprint=s['fingerprint'],selection_sha256=sha,table_sha256=digest(out/'exploratory.csv'),**counts))
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['record_checks','preflight','miniature','prepare_development','validate','freeze','exploratory'])
    parser.add_argument('--config',default=CONFIG);args=parser.parse_args();globals()[args.command](args.config)
