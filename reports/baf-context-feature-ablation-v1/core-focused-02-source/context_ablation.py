"""Fixed feature removals, native LightGBM margins and immutable model caches."""
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import resource
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from . import shallow_catboost as previous
from .common import CATEGORICAL, digest, make_model, read_config, write_json
from .focal_study import now
from .improvements import paths, prediction
from .recall_weighted import ids_digest
from .recall_weighted_metrics import POLICIES, aligned, check_data

RAW_COLUMNS = ['income','name_email_similarity','prev_address_months_count','current_address_months_count',
    'customer_age','days_since_request','intended_balcon_amount','payment_type','zip_count_4w',
    'velocity_6h','velocity_24h','velocity_4w','bank_branch_count_8w','date_of_birth_distinct_emails_4w',
    'employment_status','credit_risk_score','email_is_free','housing_status','phone_home_valid',
    'phone_mobile_valid','bank_months_count','has_other_cards','proposed_credit_limit','foreign_request',
    'source','session_length_in_minutes','device_os','keep_alive_session','device_distinct_emails_8w']
REMOVED = dict(full=[],drop_velocity=['velocity_6h','velocity_24h','velocity_4w'],
    drop_counts=['zip_count_4w','bank_branch_count_8w','date_of_birth_distinct_emails_4w'])
REMOVED['drop_both'] = REMOVED['drop_velocity'] + REMOVED['drop_counts']
WEIGHTINGS, ALPHAS = ['balanced','unweighted'], [10,25,50]
REFERENCE_ROLES = previous.REFERENCE_ROLES
ANCHOR = REFERENCE_ROLES['bagged_anchor'][0]
FULL_SOURCES = dict(balanced='baseline_balanced',unweighted='baseline_lgbm')
BASELINE_PARAMS = dict(n_estimators=200,learning_rate=.05,num_leaves=15,min_child_samples=100,
    verbosity=-1,deterministic=True,force_col_wise=True)


def detached(v): return json.loads(json.dumps(v,allow_nan=False))


def pure(representation,weighting):
    if representation not in REMOVED or weighting not in WEIGHTINGS:raise ValueError('Undeclared removal/weighting')
    return f'context_{representation}_{weighting}'


def candidate(representation,weighting,alpha):
    n=pure(representation,weighting)
    if type(alpha) is not int or alpha not in [0,*ALPHAS,100]:raise ValueError('Undeclared alpha')
    return ANCHOR if alpha==0 else n if alpha==100 else f'context_anchor_{representation}_{weighting}_a{alpha:02d}'


def recipes():
    return [dict(id=pure(r,w),representation=r,weighting=w,removed_columns=list(REMOVED[r]),
        raw_columns=29-len(REMOVED[r]),new_fit=r!='full',original_full_source=FULL_SOURCES[w])
        for r,w in product(REMOVED,WEIGHTINGS)]


def strategies():
    return ([dict(id=n,kind='reference') for n,k in REFERENCE_ROLES.values()] +
        [dict(id=pure(r,w),kind='pure',representation=r,weighting=w) for r,w in product(REMOVED,WEIGHTINGS)] +
        [dict(id=candidate(r,w,a),kind='blend',representation=r,weighting=w,alpha_numerator=a)
            for r,w,a in product(REMOVED,WEIGHTINGS,ALPHAS)])


def lookup(name):
    s=next((s for s in strategies() if s['id']==name),None)
    if s is None:raise ValueError('Undeclared strategy')
    return s


def choices_for(name,policy):
    s=lookup(name)
    if policy not in POLICIES:raise ValueError('Undeclared policy')
    roles=dict(selected=[name,policy],**detached(REFERENCE_ROLES),same_strategy=[name,'pooled'])
    if s['kind']=='reference':return roles
    r,w=s['representation'],s['weighting'];a=s.get('alpha_numerator',100)
    roles.update({f'representation_{q}':[candidate(q,w,a),policy] for q in REMOVED})
    roles.update({f'weight_{q}':[candidate(r,q,a),policy] for q in WEIGHTINGS})
    roles.update({f'alpha_a{q:03d}':[candidate(r,w,q),policy] for q in [0,*ALPHAS,100]})
    return roles


def dependencies(names):
    old=previous.dependencies([]);raw=set();blends=set()
    for n in names:
        s=lookup(n)
        if s['kind']=='reference':continue
        raw.add(pure(s['representation'],s['weighting']))
        if s['kind']=='blend':blends.add(n)
    full=sorted(n for n in raw if lookup(n)['representation']=='full')
    return dict(old_bases=old['old_bases'],old_derived=old['old_derived'],full_margin_controls=full,
        new_ablation_bases=sorted(raw-set(full)),new_blends=sorted(blends),
        full_scores=sorted(set(old['old_bases']+old['old_derived'])|raw|blends))


def contrast_factors():
    rows=[]
    def add(kind,n,k,b,bk,factors,terms=None):
        row=dict(kind=kind,candidate=n,policy=k,comparator=b,comparator_policy=bk,factors=factors)
        if terms is not None:row['terms']=terms
        rows.append(row)
    for s,k in product(strategies(),POLICIES):
        n=s['id']
        for role,(b,bk) in REFERENCE_ROLES.items():add('fixed_reference',n,k,b,bk,dict(reference_role=role))
        if k!='pooled':add('own_policy',n,k,n,'pooled',{})
    for r,w,a,k in product(REMOVED,WEIGHTINGS,[*ALPHAS,100],POLICIES):
        n=candidate(r,w,a)
        if r in ['full','drop_counts']:
            b='drop_velocity' if r=='full' else 'drop_both'
            add('remove_velocity',candidate(b,w,a),k,n,k,dict(counts_removed=r=='drop_counts',weighting=w,alpha=a))
        if r in ['full','drop_velocity']:
            b='drop_counts' if r=='full' else 'drop_both'
            add('remove_counts',candidate(b,w,a),k,n,k,dict(velocity_removed=r=='drop_velocity',weighting=w,alpha=a))
        if r=='full':
            add('remove_both',candidate('drop_both',w,a),k,n,k,dict(weighting=w,alpha=a))
            terms=[dict(candidate=candidate(q,w,a),policy=k,coefficient=z) for q,z in [('drop_both',1),('drop_velocity',-1),('drop_counts',-1),('full',1)]]
            add('removal_interaction',candidate('drop_both',w,a),k,n,k,dict(weighting=w,alpha=a),terms)
        if w=='unweighted':add('balanced_minus_unweighted',candidate(r,'balanced',a),k,n,k,dict(representation=r,alpha=a))
        if a==100:add('pure_vs_anchor',n,k,ANCHOR,k,dict(representation=r,weighting=w))
        else:
            add('blend_vs_pure',n,k,pure(r,w),k,dict(representation=r,weighting=w,alpha=a))
            add('blend_vs_anchor',n,k,ANCHOR,k,dict(representation=r,weighting=w,alpha=a))
    for r,w,k,(lo,hi) in product(REMOVED,WEIGHTINGS,POLICIES,combinations(ALPHAS,2)):
        add('higher_alpha',candidate(r,w,hi),k,candidate(r,w,lo),k,dict(representation=r,weighting=w,alpha_first=hi,alpha_second=lo))
    return rows


def blend(anchor,data,alpha):return previous.blend(anchor,data,alpha)


def select(base,spec):
    if spec not in recipes():raise ValueError('Undeclared feature-removal recipe')
    if list(base.columns)!=RAW_COLUMNS:raise ValueError('Original ordered 29-column decoded schema required')
    return base[[k for k in RAW_COLUMNS if k not in spec['removed_columns']]].copy()


def constructor(spec):
    if spec not in recipes():raise ValueError('Undeclared recipe or duplicate cost mechanism')
    c=read_config('configs/study.json')
    if c['model_parameters']['lightgbm']!=BASELINE_PARAMS or c['model_seed']!=2026 or c['threads']!=6:
        raise ValueError('Frozen original baseline settings changed')
    cols=[k for k in RAW_COLUMNS if k not in spec['removed_columns']]
    return make_model('lightgbm',spec['weighting']=='balanced',c,cols)


def _token(v):
    if v is None:return dict(kind='none')
    if pd.isna(v):return dict(kind='nan')
    return dict(kind='value',value=v.item() if hasattr(v,'item') else v)


def preprocessing(pipeline,x):
    pre=pipeline.named_steps['preprocess'];numeric=[k for k in x if k not in CATEGORICAL]
    if list(pre.feature_names_in_)!=list(x) or list(pre.transformers_[0][2])!=numeric or list(pre.transformers_[1][2])!=CATEGORICAL:
        raise ValueError('Retained fitted preprocessing schema differs')
    num=pre.named_transformers_['numeric']
    if list(num.named_steps)!=['impute']:raise ValueError('Undeclared numeric scaling/transform')
    imp=num.named_steps['impute'];enc=pre.named_transformers_['categorical']
    expected=np.asarray([float(x[k].median()) if x[k].notna().any() else 0. for k in numeric])
    indicators=np.flatnonzero(x[numeric].isna().any().to_numpy())
    if imp.strategy!='median' or not imp.add_indicator or not imp.keep_empty_features or not np.array_equal(imp.statistics_,expected) or not np.array_equal(imp.indicator_.features_,indicators):
        raise ValueError('Training-only numerical imputation/indicator values differ')
    if enc.handle_unknown!='ignore' or not enc.sparse_output:raise ValueError('Original one-hot handling differs')
    categories=[]
    for k,levels in zip(CATEGORICAL,enc.categories_):
        actual=[_token(v) for v in levels];values=sorted(x[k].dropna().unique().tolist())
        expected=[_token(v) for v in values]
        if any(v is None for v in x[k]):expected.append(dict(kind='none'))
        if any(pd.isna(v) and v is not None for v in x[k]):expected.append(dict(kind='nan'))
        if actual!=expected:raise ValueError('Training-only categorical levels differ: '+k)
        categories.append(dict(column=k,levels=actual))
    return dict(columns=list(x),numeric=numeric,categoricals=list(CATEGORICAL),medians=expected_medians(imp,numeric),
        missing_indicator_columns=[numeric[i] for i in indicators],categories=categories,
        transformed_names=list(pre.get_feature_names_out()),encoded_width=len(pre.get_feature_names_out()))


def expected_medians(imp,numeric):return {k:float(v) for k,v in zip(numeric,imp.statistics_)}


def parameters(pipeline,spec,*,miniature=False):
    estimator=pipeline.named_steps['model'];requested=detached(constructor(spec).named_steps['model'].get_params(deep=False))
    actual=detached(estimator.get_params(deep=False))
    if actual!=requested:raise ValueError('Original constructor or class costs differ')
    booster=estimator.booster_;effective=detached(booster.params);rounds=int(booster.current_iteration())
    if rounds!=int(estimator.n_estimators_) or rounds!=int(booster.num_trees()) or list(estimator.classes_)!=[0,1]:raise ValueError('Fitted rounds/classes differ')
    if not (1<=rounds<=200) or (not miniature and rounds!=200):raise ValueError('Undeclared fitted round count')
    if effective.get('scale_pos_weight',1)!=1 or effective.get('is_unbalance',False) or effective.get('num_threads')!=6:
        raise ValueError('Duplicate class costs or runtime setting differs')
    for key,value in BASELINE_PARAMS.items():
        if key in ['n_estimators','verbosity']:continue
        if effective.get(key)!=value:raise ValueError('Effective original model settings differ: '+key)
    return dict(requested=requested,effective=effective,actual_rounds=rounds,feature_names=booster.feature_name())


def margins(pipeline,x):
    z=pipeline.named_steps['preprocess'].transform(x);model=pipeline.named_steps['model']
    result=np.asarray(model.predict(z,raw_score=True,num_threads=6),dtype=np.float64)
    prob=np.asarray(model.predict_proba(z,num_threads=6),dtype=np.float64)[:,1]
    if result.shape!=(len(x),) or not np.isfinite(result).all() or not np.allclose(expit(result),prob,atol=1e-12,rtol=0):
        raise ValueError('Finite native margin/probability consistency failed')
    return result


def model_paths(c,phase,month,name):
    rp,pp=paths(c,phase,month,name)
    return [rp,pp,Path(c['output_dir'])/'models'/f'{phase}-m{month}-{name}.joblib']


def expected_model(fp,frame,base,spec,month,phase,miniature):
    retained=select(base,spec);tr=frame.index[frame.month<month];ev=frame.index[frame.month==month]
    y=frame.loc[tr,'fraud_bool'].to_numpy(dtype=np.int8)
    if not len(tr) or not len(ev) or set(np.unique(y))!={0,1}:raise ValueError('Both earlier classes/future rows required')
    costs={'0':len(y)/(2*np.count_nonzero(y==0)),'1':len(y)/(2*np.count_nonzero(y==1))} if spec['weighting']=='balanced' else {'0':1.,'1':1.}
    weights=np.asarray([costs[str(v)] for v in y],dtype=np.float64)
    return dict(candidate=spec['id'],phase=phase,eval_month=month,fingerprint=fp,spec=spec,miniature=bool(miniature),
        train_months=sorted(frame.loc[tr,'month'].unique().tolist()),train_rows=len(tr),train_fraud=int(y.sum()),
        train_ids_sha256=ids_digest(tr.to_numpy()),train_labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
        class_costs=costs,row_cost_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        evaluation_rows=len(ev),evaluation_ids_sha256=ids_digest(ev.to_numpy()),raw_predictor_columns=list(base),retained_columns=list(retained),
        requested_model_params=detached(constructor(spec).named_steps['model'].get_params(deep=False)),score_domain='finite native margin')


def audit_model(c,fp,frame,base,spec,month,phase='validation',*,miniature=False):
    from .conditional_expert_study import cache_state,verify_hashes
    ps=model_paths(c,phase,month,spec['id'])
    if not cache_state(ps):raise ValueError('Required reduced-feature model absent')
    r=read_config(ps[0]);expected=expected_model(fp,frame,base,spec,month,phase,miniature)
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Reduced-model recipe/population/cost identity differs')
    verify_hashes({str(ps[1]):r['prediction_sha256'],str(ps[2]):r['model_sha256']})
    artifact=joblib.load(ps[2]);pipe=artifact['model'];x=select(base,spec);tr=frame.index[frame.month<month];ev=frame.index[frame.month==month]
    pre=preprocessing(pipe,x.loc[tr]);params=parameters(pipe,spec,miniature=miniature)
    if artifact['training_identity']!=expected or r['preprocessing']!=pre or r['parameters']!=params or artifact['parameters']!=params:
        raise ValueError('Persisted feature/preprocessing/parameter identity differs')
    if r['warnings'] or r['train_row_rounds']!=len(tr)*params['actual_rounds'] or r['model_bytes']!=ps[2].stat().st_size or r['fit_seconds']<0 or r['peak_process_rss_bytes']<=0:
        raise ValueError('Reduced-model measured work differs')
    data=prediction(c,phase,month,spec['id']);actual=margins(pipe,x.loc[ev])
    if not np.array_equal(data['row_ids'],ev.to_numpy()) or not np.array_equal(data['labels'],frame.loc[ev,'fraud_bool'].to_numpy()) or not np.allclose(data['scores'],actual,atol=1e-12,rtol=0):
        raise ValueError('Persisted native margin population/inference differs')
    return r


def fit_model(c,fp,frame,base,spec,month,phase='validation',allow_missing=True,*,miniature=False):
    from .conditional_expert_study import cache_state
    if spec not in recipes() or not spec['new_fit']:raise ValueError('Only declared reduced recipes can fit')
    ps=model_paths(c,phase,month,spec['id']);failure=Path(c['output_dir'])/'failures'/(ps[0].stem+'.json')
    if cache_state(ps):return audit_model(c,fp,frame,base,spec,month,phase,miniature=miniature)
    if not allow_missing:raise ValueError('Reduced-feature cache absent')
    if failure.exists():raise ValueError('Prior failure retained; no automatic retry')
    try:
        expected=expected_model(fp,frame,base,spec,month,phase,miniature);x=select(base,spec)
        tr=frame.index[frame.month<month];ev=frame.index[frame.month==month];pipe=constructor(spec);start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');pipe.fit(x.loc[tr],frame.loc[tr,'fraud_bool'].to_numpy())
        seconds=time.perf_counter()-start;messages=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if messages:raise ValueError('Reduced-model fit warnings: '+repr(messages))
        pre=preprocessing(pipe,x.loc[tr]);params=parameters(pipe,spec,miniature=miniature);values=margins(pipe,x.loc[ev])
        for p in ps:p.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(dict(model=pipe,training_identity=expected,parameters=params),ps[2])
        np.savez_compressed(ps[1],row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=values)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        r=dict(**expected,preprocessing=pre,parameters=params,warnings=messages,fit_seconds=seconds,peak_process_rss_bytes=rss,
            fitted_tree_count=params['actual_rounds'],train_row_rounds=len(tr)*params['actual_rounds'],model_bytes=ps[2].stat().st_size,
            prediction_sha256=digest(ps[1]),model_sha256=digest(ps[2]),created_at_utc=now())
        write_json(ps[0],r);return audit_model(c,fp,frame,base,spec,month,phase,miniature=miniature)
    except Exception as exc:
        if not failure.exists():write_json(failure,dict(candidate=spec['id'],phase=phase,eval_month=month,fingerprint=fp,spec=spec,error=repr(exc),created_at_utc=now()))
        raise
