"""Stable focal derivatives and explicitly approximate LightGBM curvature."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits


def terms(labels, margins, gamma=2., positive_weight=1.):
    y=np.asarray(labels,dtype=float);z=np.asarray(margins,dtype=float)
    if y.ndim!=1 or not len(y) or z.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(z).all():
        raise ValueError('Invalid binary raw-margin inputs')
    if not np.isfinite(gamma) or gamma<0 or not np.isfinite(positive_weight) or positive_weight<=0:
        raise ValueError('Invalid focal parameters')
    t=2*y-1;q=expit(t*z);r=expit(-t*z);logq=-np.logaddexp(0.,-t*z)
    a=np.where(y==1,positive_weight,1.);power=r**gamma
    loss=-a*power*logq
    grad=a*t*power*(gamma*q*logq-r)
    hess=a*power*q*(gamma*logq*(r-gamma*q)+(2*gamma+1)*r)
    if not all(np.isfinite(v).all() for v in [loss,grad,hess]):raise ValueError('Nonfinite loss derivatives')
    return loss,grad,hess


def objective(gamma,positive_weight,floor,trace):
    if not np.isfinite(floor) or floor<=0:raise ValueError('Invalid curvature floor')
    def evaluate(y_true,y_pred):
        loss,g,h=terms(y_true,y_pred,gamma,positive_weight)
        replaced=(h<floor) if gamma>0 else np.zeros(len(h),dtype=bool)
        supplied=np.maximum(h,floor) if gamma>0 else h
        trace.append(dict(iteration=len(trace)+1,n=len(h),mean_loss=float(loss.mean()),replaced_count=int(replaced.sum()),
                          replaced_fraction=float(replaced.mean()),true_hessian_min=float(h.min()),true_hessian_max=float(h.max()),
                          supplied_hessian_min=float(supplied.min()),gradient_abs_max=float(np.max(np.abs(g))),
                          approximation='positive diagonal curvature floor' if gamma>0 else 'exact logistic curvature'))
        return g,supplied
    return evaluate


def estimator(spec,config,trace):
    params=dict(config['tree_params']);params.update(random_state=config['seed'],n_jobs=config['threads'],verbosity=-1)
    if spec['kind']=='builtin':params.update(objective='binary',scale_pos_weight=spec['positive_weight'])
    elif spec['kind']=='custom':params['objective']=objective(spec['gamma'],spec['positive_weight'],config['hessian_floor'],trace)
    else:raise ValueError('Unknown loss kind')
    return LGBMClassifier(**params)


def scores(booster,inputs):
    raw=np.asarray(booster.predict(inputs,raw_score=True),dtype=float)
    if raw.ndim!=1 or not len(raw) or not np.isfinite(raw).all():raise ValueError('Invalid raw prediction')
    return raw,expit(raw)


def integration_gate(config):
    """Four tiny fits verify only integration, not model performance."""
    x=pd.DataFrame(dict(a=np.tile(np.arange(10),40),b=np.repeat(np.arange(40)%7,10)))
    y=((x.a>=5)^(x.b==3)).to_numpy(dtype=int)
    cfg={**config,'threads':1,'tree_params':{**config['tree_params'],'n_estimators':40,'min_child_samples':5,'colsample_bytree':1.}}
    rows=[]
    for weight in [1,3]:
        outputs={};traces={}
        for kind in ['custom','builtin']:
            trace=[];model=estimator(dict(kind=kind,gamma=0,positive_weight=weight),cfg,trace)
            with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                warnings.simplefilter('always');model.fit(x,y);outputs[kind]=scores(model.booster_,x)
            if caught:raise ValueError('Integration gate warnings: '+str([str(w.message) for w in caught]))
            traces[kind]=len(trace)
        difference=float(np.max(np.abs(outputs['custom'][1]-outputs['builtin'][1])))
        if difference>1e-6 or traces['custom']!=40 or traces['builtin']!=0:
            raise ValueError('Custom logistic integration gate failed: '+str(difference))
        rows.append(dict(positive_weight=weight,max_probability_difference=difference,miniature_fits=2,passed=True))
    return rows
