"""Native categorical XGBoost contract and miniature persistence gate."""
from __future__ import annotations
from pathlib import Path
import tempfile
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from threadpoolctl import threadpool_limits
from .engineering import NativeInputs


def tree_parameters():
    return dict(n_estimators=800,learning_rate=.03,min_child_weight=10,reg_lambda=5,reg_alpha=0,gamma=0,
        subsample=.8,colsample_bytree=.9,tree_method='hist',grow_policy='depthwise',max_bin=256,
        max_cat_to_onehot=4,max_cat_threshold=64,enable_categorical=True,device='cpu',
        objective='binary:logistic',base_score=.5,eval_metric='logloss')


def presets():
    return [dict(id=f'xgb_d{d}_w{w}',max_depth=d,scale_pos_weight=w) for d in [4,6] for w in [1,5]]


def parameters(spec,c):
    return {**c['tree_params'],**{k:spec[k] for k in ['max_depth','scale_pos_weight']},
            'random_state':c['seed'],'n_jobs':c['threads']}


def fit_checked(x,y,spec,c):
    y=np.asarray(y)
    if y.ndim!=1 or len(y)!=len(x) or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ValueError('Invalid binary training population')
    if {'fraud_bool','month','device_fraud_count'}&set(x.columns):raise ValueError('Excluded predictors')
    model=xgb.XGBClassifier(**parameters(spec,c))
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=c['threads']):
        warnings.simplefilter('always');model.fit(x,y)
    messages=sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    if messages:raise ValueError('Inspect XGBoost fit warnings: '+str(messages))
    if not np.array_equal(model.classes_,[0,1]):raise ValueError('Class orientation changed')
    return model,messages


def scores(booster,x):
    result=np.asarray(booster.predict(xgb.DMatrix(x,enable_categorical=True)),dtype=float)
    if result.ndim!=1 or len(result)!=len(x) or not np.isfinite(result).all() or np.any((result<0)|(result>1)):
        raise ValueError('Invalid positive-class logistic scores')
    return result


def load_booster(path,threads=6):
    b=xgb.Booster(params={'nthread':threads});b.load_model(path);return b


def integration_gate(c):
    rng=np.random.default_rng(2026);n=512
    base=pd.DataFrame({'payment_type':np.array(['AA','AB','AC','AD'])[np.arange(n)%4],
                       'signal':rng.normal(size=n),'missing':rng.normal(size=n)})
    base.loc[::17,'missing']=np.nan
    y=(base.signal.to_numpy()+rng.normal(scale=.4,size=n)>.5).astype(int)
    adapter=NativeInputs().fit(base);x=adapter.transform(base)
    ev=base.iloc[:20].copy();ev.loc[ev.index[0],'payment_type']='UNSEEN';ev.loc[ev.index[1],'payment_type']=None
    xe=adapter.transform(ev)
    if not xe.payment_type.isna().iloc[:2].all() or 'UNSEEN' in adapter.levels['payment_type']:
        raise ValueError('Unknown categorical handling differs')
    short={**c,'tree_params':{**c['tree_params'],'n_estimators':12}};rows=[]
    with tempfile.TemporaryDirectory() as folder:
        for spec in presets():
            repeats=[]
            for j in range(2):
                model,messages=fit_checked(x,y,spec,short);b=model.get_booster();p=scores(b,xe)
                if not np.allclose(p,model.predict_proba(xe)[:,1],atol=1e-12,rtol=1e-12):raise ValueError('Positive-class orientation differs')
                path=Path(folder)/f'{spec["id"]}-{j}.ubj';model.save_model(path);reloaded=load_booster(path,c['threads'])
                if not np.allclose(p,scores(reloaded,xe),atol=1e-12,rtol=1e-12):raise ValueError('Categorical reload differs')
                expected=['c','float','float']
                if b.feature_names!=adapter.columns or b.feature_types!=expected:raise ValueError('Native feature types differ')
                if b.num_boosted_rounds()!=12:raise ValueError('Miniature round count differs')
                repeats.append(p)
            difference=float(np.max(np.abs(repeats[0]-repeats[1])))
            if difference>1e-12:raise ValueError('Miniature repeat differs')
            rows.append(dict(candidate=spec['id'],fits=2,rounds_per_fit=12,repeat_max_difference=difference,
                unknown_as_missing=True,reload_checked=True,class_orientation_checked=True,warnings=[]))
    return rows
