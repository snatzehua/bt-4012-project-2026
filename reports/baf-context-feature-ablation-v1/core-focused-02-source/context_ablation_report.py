"""Independent reconstruction for the fixed context-removal study."""
import hashlib
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from lightgbm import LGBMClassifier

from .common import read_config,digest
from .improvements import prediction

CATS=['payment_type','employment_status','housing_status','source','device_os']
VELOCITY=['velocity_6h','velocity_24h','velocity_4w']
COUNTS=['zip_count_4w','bank_branch_count_8w','date_of_birth_distinct_emails_4w']
REMOVALS=dict(full=[],drop_velocity=VELOCITY,drop_counts=COUNTS,drop_both=VELOCITY+COUNTS)


def category_key(v):
    if v is None:return ('none',)
    if pd.isna(v):return ('nan',)
    return ('value',str(v))


def independent_inputs(pipe,train,future):
    """Rebuild imputed numerical/missing-indicator/one-hot CSR from training rows."""
    pre=pipe.named_steps['preprocess'];nums=[k for k in train if k not in CATS]
    if list(pre.feature_names_in_)!=list(train) or list(pre.transformers_[0][2])!=nums or list(pre.transformers_[1][2])!=CATS:
        raise ValueError('Independent retained preprocessing fields differ')
    num_pipeline=pre.named_transformers_['numeric'];imp=num_pipeline.named_steps['impute'];enc=pre.named_transformers_['categorical']
    if list(num_pipeline.named_steps)!=['impute'] or imp.strategy!='median' or not imp.keep_empty_features or not imp.add_indicator:
        raise ValueError('Independent numeric representation differs')
    a=train[nums].to_numpy(dtype=float);z=future[nums].to_numpy(dtype=float)
    medians=np.asarray([float(np.nanmedian(a[:,i])) if np.isfinite(a[:,i]).any() else 0. for i in range(len(nums))])
    indicators=np.flatnonzero(np.isnan(a).any(axis=0));filled=np.where(np.isnan(z),medians,z)
    if not np.array_equal(imp.statistics_,medians) or not np.array_equal(imp.indicator_.features_,indicators):raise ValueError('Independent fitted numerical statistics differ')
    numeric=np.column_stack([filled,np.isnan(z[:,indicators]).astype(float)])
    row=[];col=[];offset=0;categories=[]
    if enc.handle_unknown!='ignore' or not enc.sparse_output:raise ValueError('Independent one-hot policy differs')
    names=['numeric__'+k for k in nums]+['numeric__missingindicator_'+nums[i] for i in indicators]
    for k,levels in zip(CATS,enc.categories_):
        values=sorted(set(train[k].dropna().astype(str)))
        wanted=[('value',v) for v in values]
        if any(v is None for v in train[k]):wanted.append(('none',))
        if any(pd.isna(v) and v is not None for v in train[k]):wanted.append(('nan',))
        actual=[category_key(v) for v in levels]
        if actual!=wanted:raise ValueError('Independent categorical training vocabulary differs')
        index={v:i for i,v in enumerate(wanted)}
        for i,v in enumerate(future[k]):
            code=index.get(category_key(v))
            if code is not None:row.append(i);col.append(offset+code)
        offset+=len(wanted);categories.append(dict(column=k,levels=[dict(kind='value',value=v[1]) if v[0]=='value' else dict(kind=v[0]) for v in wanted]))
        names.extend('categorical__'+k+'_'+str(v) for v in levels)
    direct=sparse.hstack([sparse.csr_matrix(numeric),sparse.csr_matrix((np.ones(len(row)),(row,col)),shape=(len(future),offset))],format='csr')
    actual=sparse.csr_matrix(pre.transform(future))
    if direct.shape!=actual.shape or np.any((direct-actual).data!=0):raise ValueError('Independent encoded input matrix differs')
    if list(pre.get_feature_names_out())!=names:raise ValueError('Independent encoded feature names differ')
    metadata=dict(columns=list(train),numeric=nums,categoricals=CATS,medians={k:float(v) for k,v in zip(nums,medians)},
        missing_indicator_columns=[nums[i] for i in indicators],categories=categories,transformed_names=names,encoded_width=len(names))
    return direct,metadata


def audit_reduced(c,fp,frame,base,record):
    """Inspect underlying saved model/data without using producer audit helpers."""
    n,m,phase=record['candidate'],record['eval_month'],record['phase'];spec=record['spec'];rep=spec['representation'];weight=spec['weighting']
    if rep not in ['drop_velocity','drop_counts','drop_both'] or weight not in ['balanced','unweighted'] or n!=f'context_{rep}_{weight}':raise ValueError('Independent removal recipe differs')
    if spec!=dict(id=n,representation=rep,weighting=weight,removed_columns=REMOVALS[rep],raw_columns=29-len(REMOVALS[rep]),new_fit=True,original_full_source='baseline_balanced' if weight=='balanced' else 'baseline_lgbm'):
        raise ValueError('Independent complete removal declaration differs')
    cols=[k for k in base if k not in REMOVALS[rep]];x=base[cols];tr=frame.index[frame.month<m];ev=frame.index[frame.month==m];y=frame.loc[tr,'fraud_bool'].to_numpy(dtype=np.int8)
    if len(base.columns)!=29 or len(cols)!=spec['raw_columns'] or record['raw_predictor_columns']!=list(base) or record['retained_columns']!=cols:
        raise ValueError('Independent original/retained raw schema differs')
    if record['train_rows']!=len(tr) or record['train_fraud']!=int(y.sum()) or record['train_ids_sha256']!=hashlib.sha256(np.asarray(tr,dtype='<i8').tobytes()).hexdigest() or record['train_labels_sha256']!=hashlib.sha256(y.tobytes()).hexdigest():
        raise ValueError('Independent ordered training identity differs')
    costs={'0':len(y)/(2*np.count_nonzero(y==0)),'1':len(y)/(2*np.count_nonzero(y==1))} if weight=='balanced' else {'0':1.,'1':1.}
    weights=np.where(y==1,costs['1'],costs['0']).astype(np.float64)
    if record['class_costs']!=costs or record['row_cost_sha256']!=hashlib.sha256(weights.tobytes()).hexdigest():raise ValueError('Independent whole-training costs differ')
    stem=f'{phase}-m{m}-{n}';out=Path(c['output_dir']);modelpath=out/'models'/(stem+'.joblib');scorepath=out/'predictions'/(stem+'.npz')
    if digest(modelpath)!=record['model_sha256'] or digest(scorepath)!=record['prediction_sha256']:raise ValueError('Independent persisted hash differs')
    a=joblib.load(modelpath);pipe=a['model'];z,pre=independent_inputs(pipe,x.loc[tr],x.loc[ev]);model=pipe.named_steps['model']
    original=read_config('configs/study.json');params=dict(original['model_parameters']['lightgbm'],n_jobs=6,random_state=2026,class_weight='balanced' if weight=='balanced' else None)
    requested=LGBMClassifier(**params).get_params(deep=False)
    if model.get_params(deep=False)!=requested or record['requested_model_params']!=requested:raise ValueError('Independent original constructor settings differ')
    effective=model.booster_.params;rounds=int(model.booster_.current_iteration());p=record['parameters']
    if p['requested']!=requested or p['effective']!=effective or p['actual_rounds']!=rounds or p['feature_names']!=model.booster_.feature_name() or list(model.classes_)!=[0,1]:raise ValueError('Independent persisted parameters/classes differ')
    if effective.get('scale_pos_weight',1)!=1 or effective.get('is_unbalance',False) or effective.get('num_threads')!=6:raise ValueError('Independent duplicate costs/runtime differs')
    if rounds!=int(model.n_estimators_) or rounds!=int(model.booster_.num_trees()) or not 1<=rounds<=200 or (not record['miniature'] and rounds!=200):raise ValueError('Independent actual fitted rounds differ')
    if record['preprocessing']!=pre or a['parameters']!=p or any(a['training_identity'][k]!=record[k] for k in a['training_identity']):raise ValueError('Independent persisted preprocessing/identity differs')
    actual=np.asarray(model.booster_.predict(z,raw_score=True,num_threads=6));proba=np.asarray(model.booster_.predict(z,num_threads=6));d=prediction(c,phase,m,n)
    if not np.allclose(expit(actual),proba,atol=1e-12,rtol=0) or not np.allclose(actual,d['scores'],atol=1e-12,rtol=0):raise ValueError('Independent saved raw/probability inference differs')
    if not np.array_equal(d['row_ids'],ev.to_numpy()) or not np.array_equal(d['labels'],frame.loc[ev,'fraud_bool'].to_numpy()):raise ValueError('Independent future population differs')
    if record['fingerprint']!=fp or record['warnings'] or record['train_row_rounds']!=len(tr)*rounds or record['fitted_tree_count']!=rounds or record['model_bytes']!=modelpath.stat().st_size or record['fit_seconds']<0 or record['peak_process_rss_bytes']<=0:
        raise ValueError('Independent actual model work differs')
    return dict(candidate=n,eval_month=m,encoded_width=z.shape[1],saved_native_inference_verified=True)
