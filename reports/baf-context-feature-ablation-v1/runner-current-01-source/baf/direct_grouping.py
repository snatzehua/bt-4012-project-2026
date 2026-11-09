"""Target-independent artificial queries and explicit raw ranking scores."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from .recall_weighted import ids_digest


def grouping(ids,months,size,seed=2026):
    ids=np.asarray(ids);months=np.asarray(months)
    if ids.ndim!=1 or not len(ids) or months.shape!=ids.shape or not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(months.dtype,np.integer):
        raise ValueError('Invalid grouping population')
    if len(np.unique(ids))!=len(ids) or np.any(months<0) or not isinstance(size,(int,np.integer)) or size<2 or not isinstance(seed,int) or seed<0:
        raise ValueError('Invalid grouping recipe')
    ordered=[];groups=[];group_months=[]
    for month in np.unique(months):
        selected=np.sort(ids[months==month]);n=len(selected)
        if n<2:raise ValueError('One-row month cannot form a query')
        selected=np.random.default_rng(np.random.SeedSequence([seed,int(month)])).permutation(selected)
        sizes=[size]*(n//size)
        if n%size:
            if n%size==1 and sizes:sizes[-1]+=1
            else:sizes.append(n%size)
        ordered.extend(selected.tolist());groups.extend(sizes);group_months.extend([int(month)]*len(sizes))
    order=np.asarray(ordered,dtype=np.int64);g=np.asarray(groups,dtype=np.int64)
    if g.sum()!=len(ids) or np.any(g<2) or not np.array_equal(np.sort(order),np.sort(ids)):
        raise ValueError('Grouping lost or duplicated rows')
    return order,g,np.asarray(group_months,dtype=np.int64)


def describe(order,groups,group_months,labels):
    y=np.asarray(labels)
    if y.shape!=order.shape or not np.isin(y,[0,1]).all() or groups.sum()!=len(y) or len(groups)!=len(group_months):
        raise ValueError('Grouping label audit differs')
    starts=np.r_[0,np.cumsum(groups)[:-1]];ends=np.cumsum(groups)
    fraud=np.add.reduceat(y.astype(np.int64),starts)
    rows=[dict(query_id=i,month=int(m),start=int(a),end=int(b),n=int(n),fraud=int(f))
          for i,(m,a,b,n,f) in enumerate(zip(group_months,starts,ends,groups,fraud))]
    return dict(ordered_ids_sha256=ids_digest(order),groups_sha256=ids_digest(groups),
                group_months_sha256=ids_digest(group_months),rows=len(order),queries=len(groups),
                zero_positive_fraction=float(np.mean(fraud==0)),all_positive_fraction=float(np.mean(fraud==groups)),
                groups=groups.tolist(),query_rows=rows)


def parameters(spec,config):
    params={**config['tree_params'],'reg_lambda':spec['reg_lambda'],'random_state':config['seed'],
            'n_jobs':config['threads'],'verbosity':-1}
    if spec['kind']=='rank':params.update(objective='rank_xendcg',objective_seed=config['seed'])
    elif spec['kind']=='binary':params.update(objective='binary',boost_from_average=False)
    else:raise ValueError('Unknown direct model kind')
    return params


def estimator(spec,config):
    cls=lgb.LGBMRanker if spec['kind']=='rank' else lgb.LGBMClassifier
    return cls(**parameters(spec,config))


def raw_scores(booster,inputs):
    result=np.asarray(booster.predict(inputs,raw_score=True),dtype=float)
    if result.ndim!=1 or not len(result) or not np.isfinite(result).all():raise ValueError('Invalid raw ranking score')
    return result


def fit_checked(model,x,y,groups=None):
    y=np.asarray(y)
    if y.ndim!=1 or len(y)!=len(x) or not np.isin(y,[0,1]).all():raise ValueError('Invalid binary relevance labels')
    if isinstance(model,lgb.LGBMRanker):
        g=np.asarray(groups)
        if g.ndim!=1 or not len(g) or not np.issubdtype(g.dtype,np.integer) or np.any(g<2) or g.sum()!=len(y):
            raise ValueError('Invalid contiguous query coverage')
        model.fit(x,y.astype(np.int32),group=g)
    else:model.fit(x,y.astype(np.int32))
    return model


def integration_gate(config):
    x=pd.DataFrame(dict(a=np.tile(np.arange(20),20),b=np.repeat(np.arange(20)%5,20)))
    y=((x.a>14)^(x.b==3)).to_numpy(dtype=np.int32);g=np.array([20]*20)
    cfg={**config,'threads':1,'tree_params':{**config['tree_params'],'n_estimators':20,'min_child_samples':5,'colsample_bytree':1.}}
    result=[]
    for kind in ['rank','binary']:
        spec=dict(kind=kind,reg_lambda=.1);outputs=[]
        # A zero-tree native booster directly audits the installed initial margin.
        params=parameters(spec,cfg);dataset=lgb.Dataset(x,label=y,group=g if kind=='rank' else None,
            params={'min_data_in_leaf':5,'verbosity':-1},free_raw_data=False)
        b=lgb.Booster(params=params,train_set=dataset);initial=raw_scores(b,x)
        if b.current_iteration()!=0 or not np.array_equal(initial,np.zeros(len(x))):raise ValueError('Nonzero initial margin')
        for _ in range(2):
            model=estimator(spec,cfg)
            with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                warnings.simplefilter('always');fit_checked(model,x,y,g);outputs.append(raw_scores(model.booster_,x))
            if caught:raise ValueError('Integration warnings: '+str([str(w.message) for w in caught]))
        discrepancy=float(np.max(np.abs(outputs[0]-outputs[1])))
        if discrepancy!=0:raise ValueError('Nondeterministic miniature ranking fit')
        result.append(dict(kind=kind,miniature_fits=2,initial_margin_max=float(np.max(np.abs(initial))),
                           repeat_max_difference=discrepancy,score_min=float(outputs[0].min()),score_max=float(outputs[0].max()),passed=True))
    return result
