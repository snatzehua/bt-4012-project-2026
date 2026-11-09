"""Declared native CatBoost recipes, margin compositions and immutable caches."""
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
from catboost import CatBoostClassifier
from scipy.special import expit

from . import conditional_experts as ancestor
from .common import CATEGORICAL, digest, read_config, write_json
from .engineering import NativeInputs
from .focal_study import now
from .improvements import paths, prediction
from .recall_weighted_metrics import POLICIES, aligned, check_data
from .recall_weighted import ids_digest

DEPTHS, ROUNDS, COSTS, ALPHAS = [3,4,6], [200,400], [1,100], [10,25,50]
REFERENCE_ROLES = ancestor.REFERENCE_ROLES
ANCHOR = REFERENCE_ROLES['bagged_anchor'][0]
COMMON = dict(learning_rate=.05,l2_leaf_reg=5,border_count=128,max_ctr_complexity=1,
    loss_function='Logloss',task_type='CPU',boosting_type='Plain',grow_policy='SymmetricTree',
    random_seed=2026,thread_count=6,use_best_model=False,allow_writing_files=False,verbose=False)


def detached(value):
    return json.loads(json.dumps(value,allow_nan=False))


def native(d,r,w):
    if any(type(v) is not int for v in [d,r,w]) or d not in DEPTHS or r not in ROUNDS or w not in COSTS:
        raise ValueError('Undeclared native coordinates')
    return f'cb_d{d}_r{r}_w{w}'


def candidate(d,r,w,a):
    name=native(d,r,w)
    if type(a) is not int or a not in [0,*ALPHAS,100]:raise ValueError('Undeclared alpha')
    return ANCHOR if a==0 else name if a==100 else f'cb_anchor_d{d}_r{r}_w{w}_a{a:02d}'


def recipes():
    return [dict(id=native(d,r,w),family='catboost',engineered=False,depth=d,iterations=r,
        positive_class_cost=w,params=dict(COMMON,depth=d,iterations=r,class_weights=[1,w],cat_features=list(CATEGORICAL)))
        for d,r,w in product(DEPTHS,ROUNDS,COSTS)]


def strategies():
    return ([dict(id=n,kind='reference') for n,k in REFERENCE_ROLES.values()]+
        [dict(id=s['id'],kind='native',depth=s['depth'],iterations=s['iterations'],
            positive_class_cost=s['positive_class_cost'],alpha_numerator=100,alpha_denominator=100) for s in recipes()]+
        [dict(id=candidate(d,r,w,a),kind='blend',depth=d,iterations=r,positive_class_cost=w,
            alpha_numerator=a,alpha_denominator=100,members=[ANCHOR,native(d,r,w)])
            for d,r,w,a in product(DEPTHS,ROUNDS,COSTS,ALPHAS)])


def lookup(name):
    found=next((s for s in strategies() if s['id']==name),None)
    if found is None:raise ValueError('Undeclared strategy')
    return found


def choices_for(name,policy):
    s=lookup(name)
    if policy not in POLICIES:raise ValueError('Undeclared policy')
    result=dict(selected=[name,policy],**detached(REFERENCE_ROLES),same_strategy=[name,'pooled'])
    if s['kind']=='reference':return result
    d,r,w,a=[s[k] for k in ['depth','iterations','positive_class_cost','alpha_numerator']]
    result.update(anchor_at_policy=[ANCHOR,policy],pure_catboost=[native(d,r,w),policy])
    result.update({f'depth_d{v}':[candidate(v,r,w,a),policy] for v in DEPTHS})
    result.update({f'rounds_r{v}':[candidate(d,v,w,a),policy] for v in ROUNDS})
    result.update({f'cost_w{v}':[candidate(d,r,v,a),policy] for v in COSTS})
    result.update({f'alpha_a{v:03d}':[candidate(d,r,w,v),policy] for v in [0,*ALPHAS,100]})
    return result


def dependencies(names):
    old=ancestor.old_scores.dependencies([n for n,k in REFERENCE_ROLES.values()])
    bases,blends=set(),set()
    for n in names:
        s=lookup(n)
        if s['kind']=='reference':continue
        bases.add(native(s['depth'],s['iterations'],s['positive_class_cost']))
        if s['kind']=='blend':blends.add(n)
    return dict(old_bases=old['bases'],old_scores=old['scores'],old_derived=old['derived'],
        catboost_bases=sorted(bases),new_blends=sorted(blends),
        full_scores=sorted(set(old['scores'])|bases|blends))


def contrast_factors():
    rows=[]
    def add(kind,n,k,b,bk,**factors):rows.append(dict(kind=kind,candidate=n,policy=k,comparator=b,comparator_policy=bk,factors=factors))
    for s,k in product(strategies(),POLICIES):
        for role,(b,bk) in REFERENCE_ROLES.items():add('fixed_reference',s['id'],k,b,bk,reference=role)
        if k!='pooled':add('own_policy',s['id'],k,s['id'],'pooled')
    for r,w,a,k in product(ROUNDS,COSTS,[*ALPHAS,100],POLICIES):
        for lo,hi in combinations(DEPTHS,2):add('matched_depth',candidate(hi,r,w,a),k,candidate(lo,r,w,a),k,lower_depth=lo,higher_depth=hi,iterations=r,positive_class_cost=w,alpha=a)
    for d,w,a,k in product(DEPTHS,COSTS,[*ALPHAS,100],POLICIES):
        add('matched_rounds',candidate(d,400,w,a),k,candidate(d,200,w,a),k,depth=d,positive_class_cost=w,alpha=a)
    for d,r,a,k in product(DEPTHS,ROUNDS,[*ALPHAS,100],POLICIES):
        add('matched_cost',candidate(d,r,100,a),k,candidate(d,r,1,a),k,depth=d,iterations=r,alpha=a)
    for d,r,w,k in product(DEPTHS,ROUNDS,COSTS,POLICIES):
        add('native_vs_anchor',native(d,r,w),k,ANCHOR,k)
        for a in ALPHAS:
            for kind,b in [('blend_vs_native',native(d,r,w)),('blend_vs_anchor',ANCHOR)]:add(kind,candidate(d,r,w,a),k,b,k)
        for lo,hi in combinations(ALPHAS,2):add('higher_alpha',candidate(d,r,w,hi),k,candidate(d,r,w,lo),k,lower_alpha=lo,higher_alpha=hi)
    return rows


def blend(anchor,native_data,alpha):
    if isinstance(alpha,bool) or alpha not in [.1,.25,.5]:raise ValueError('Undeclared blend')
    aligned(anchor,native_data)
    result=dict(row_ids=anchor['row_ids'],labels=anchor['labels'],
        scores=(1-alpha)*np.asarray(anchor['scores'],dtype=np.float64)+alpha*np.asarray(native_data['scores'],dtype=np.float64))
    check_data(result);return result


def model_paths(c,phase,month,name):
    rp,pp=paths(c,phase,month,name);stem=f'{phase}-m{month}-{name}'
    return [rp,pp,Path(c['output_dir'])/'models'/(stem+'.cbm'),Path(c['output_dir'])/'models'/(stem+'.joblib')]


def pre_metadata(adapter,x):
    fresh=NativeInputs(catboost=True).fit(x)
    fields=['columns','categoricals','numeric','levels']
    if adapter.catboost is not True or any(getattr(adapter,k)!=getattr(fresh,k) for k in fields) or not adapter.medians.equals(fresh.medians):
        raise ValueError('Native training-only preprocessing differs')
    return dict(columns=fresh.columns,categoricals=fresh.categoricals,numeric=fresh.numeric,
        levels=fresh.levels,medians={k:float(v) for k,v in fresh.medians.items()})


def validate_recipe(spec):
    if any(type(spec.get(k)) is not int for k in ['depth','iterations','positive_class_cost']) or spec not in recipes():
        raise ValueError('Undeclared recipe or duplicate cost mechanism')
    return detached(spec['params'])


def effective_parameters(model,spec,columns):
    requested=validate_recipe(spec);effective=detached(model.get_all_params())
    for key in ['depth','iterations','l2_leaf_reg','border_count','max_ctr_complexity','loss_function',
                'task_type','boosting_type','grow_policy','random_seed','class_weights','use_best_model']:
        if effective.get(key)!=requested[key]:raise ValueError('Effective native recipe differs: '+key)
    if effective.get('learning_rate')!=float(np.float32(requested['learning_rate'])):
        raise ValueError('Effective native learning rate differs')
    if effective.get('auto_class_weights','None')!='None' or effective.get('scale_pos_weight',1)!=1:
        raise ValueError('Duplicate effective class costs')
    if model.tree_count_!=spec['iterations'] or list(model.classes_)!=[0,1]:raise ValueError('Native fitted rounds/classes differ')
    if model.feature_names_!=list(columns) or model.get_cat_feature_indices()!=[list(columns).index(k) for k in CATEGORICAL]:
        raise ValueError('Native schema/categorical indices differ')
    return effective


def margin(model,adapter,x):
    z=adapter.transform(x)
    if not np.isfinite(z[adapter.numeric].to_numpy(dtype=float)).all():raise ValueError('Invalid native numeric inputs')
    s=np.asarray(model.predict(z,prediction_type='RawFormulaVal',thread_count=6),dtype=np.float64)
    if s.ndim!=1 or len(s)!=len(x) or not np.isfinite(s).all():raise ValueError('Invalid native margins')
    return s


def expected_model(fp,frame,base,spec,month,phase):
    validate_recipe(spec)
    if list(base.columns)!=list(frame.drop(columns=['fraud_bool','month','device_fraud_count'],errors='ignore').columns):
        # Synthetic fixtures may omit the excluded constant but retain all raw columns.
        raise ValueError('Raw predictor schema differs')
    if len(base.columns)!=29 or set(CATEGORICAL)-set(base):raise ValueError('Exactly 29 raw predictors required')
    train=frame.index[frame.month<month];ev=frame.index[frame.month==month]
    y=frame.loc[train,'fraud_bool'].to_numpy()
    if not len(train) or not len(ev) or set(np.unique(y))!={0,1}:raise ValueError('Both earlier binary classes and future rows required')
    costs=np.where(y==1,spec['positive_class_cost'],1).astype(np.float64)
    return dict(candidate=spec['id'],phase=phase,eval_month=month,fingerprint=fp,spec=spec,
        train_months=sorted(frame.loc[train,'month'].unique().tolist()),train_rows=len(train),train_fraud=int(y.sum()),
        train_ids_sha256=ids_digest(train.to_numpy()),train_labels_sha256=hashlib.sha256(np.asarray(y,dtype=np.int8).tobytes()).hexdigest(),
        class_costs={'0':1,'1':spec['positive_class_cost']},row_cost_sha256=hashlib.sha256(costs.tobytes()).hexdigest(),
        evaluation_rows=len(ev),evaluation_ids_sha256=ids_digest(ev.to_numpy()),predictor_columns=list(base),
        requested_model_params=detached(spec['params']),score_domain='finite raw margin')


def audit_model(c,fp,frame,base,spec,month,phase='validation'):
    from .conditional_expert_study import cache_state,verify_hashes
    ps=model_paths(c,phase,month,spec['id'])
    if not cache_state(ps):raise ValueError('Required native cache absent')
    record=read_config(ps[0]);expected=expected_model(fp,frame,base,spec,month,phase)
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Native model population/recipe differs')
    verify_hashes({str(p):record[k] for p,k in zip(ps[1:],['prediction_sha256','native_model_sha256','adapter_sha256'])})
    artifact=joblib.load(ps[3]);adapter=artifact['adapter'];train=frame.index[frame.month<month]
    metadata=pre_metadata(adapter,base.loc[train])
    if record['preprocessing']!=metadata or artifact['requested_model_params']!=expected['requested_model_params']:
        raise ValueError('Native persisted preprocessing/recipe differs')
    model=CatBoostClassifier();model.load_model(str(ps[2]));effective=effective_parameters(model,spec,base.columns)
    if effective!=record['effective_model_params'] or effective!=artifact['effective_model_params'] or artifact['training_identity']!=expected:
        raise ValueError('Native effective/persisted identity differs')
    if record['fitted_tree_count']!=model.tree_count_ or record['warnings'] or record['train_row_rounds']!=len(train)*model.tree_count_:
        raise ValueError('Native rounds/warnings/work differs')
    saved=prediction(c,phase,month,spec['id']);ev=frame.index[frame.month==month]
    if not np.array_equal(saved['row_ids'],ev.to_numpy()) or not np.array_equal(saved['labels'],frame.loc[ev,'fraud_bool'].to_numpy()):
        raise ValueError('Native saved population differs')
    actual=margin(model,adapter,base.loc[ev])
    if not np.allclose(actual,saved['scores'],atol=1e-12,rtol=0):raise ValueError('Native saved inference differs')
    proba=model.predict_proba(adapter.transform(base.loc[ev]),thread_count=6)[:,1]
    if not np.allclose(expit(actual),proba,atol=1e-12,rtol=0):raise ValueError('Native sigmoid consistency differs')
    return record


def fit_model(c,fp,frame,base,spec,month,phase='validation',allow_missing=True):
    from .conditional_expert_study import cache_state
    name=spec['id'];ps=model_paths(c,phase,month,name)
    failure=Path(c['output_dir'])/'failures'/(ps[0].stem+'.json')
    if cache_state(ps):return audit_model(c,fp,frame,base,spec,month,phase)
    if not allow_missing:raise ValueError('Required native cache absent')
    if failure.exists():raise ValueError('Prior native failure retained; no automatic retry')
    start=time.perf_counter()
    try:
        expected=expected_model(fp,frame,base,spec,month,phase)
        train=frame.index[frame.month<month];ev=frame.index[frame.month==month]
        adapter=NativeInputs(catboost=True).fit(base.loc[train]);pre=pre_metadata(adapter,base.loc[train])
        model=CatBoostClassifier(**detached(spec['params']))
        if detached(model.get_params())!=expected['requested_model_params']:raise ValueError('Native constructor differs')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(adapter.transform(base.loc[train]),frame.loc[train,'fraud_bool'].to_numpy())
        messages=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if messages:raise ValueError('Native fit warnings: '+repr(messages))
        fit_seconds=time.perf_counter()-start;effective=effective_parameters(model,spec,base.columns)
        values=margin(model,adapter,base.loc[ev])
        for p in ps:p.parent.mkdir(parents=True,exist_ok=True)
        model.save_model(str(ps[2]),format='cbm')
        joblib.dump(dict(adapter=adapter,requested_model_params=expected['requested_model_params'],
            effective_model_params=effective,training_identity=expected),ps[3])
        np.savez_compressed(ps[1],row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=values)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        record=dict(**expected,preprocessing=pre,effective_model_params=effective,fitted_tree_count=model.tree_count_,
            warnings=messages,fit_seconds=fit_seconds,peak_process_rss_bytes=rss,train_row_rounds=len(train)*model.tree_count_,
            prediction_sha256=digest(ps[1]),native_model_sha256=digest(ps[2]),adapter_sha256=digest(ps[3]),
            model_bytes=sum(p.stat().st_size for p in ps[2:]),created_at_utc=now())
        write_json(ps[0],record)
        return audit_model(c,fp,frame,base,spec,month,phase)
    except Exception as exc:
        if not failure.exists():write_json(failure,dict(candidate=name,phase=phase,eval_month=month,
            fingerprint=fp,spec=spec,error=repr(exc),created_at_utc=now()))
        raise
