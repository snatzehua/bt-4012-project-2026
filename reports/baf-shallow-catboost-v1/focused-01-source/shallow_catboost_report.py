"""Independent saved-model, chronological-policy and result reconstruction."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import shallow_catboost_study as run
from .common import read_config, write_json, digest, load_data, features
from .improvements import paths, prediction
from .conditional_expert_report import keys, close, direct_group_metrics
from .spline_fusion_report import independent_rule, independent_mask
from .bagged_anchor_report import audit_compositions, uncertainty as paired_uncertainty

POLICIES=['pooled','latest','rank_fraction']
CATS=['payment_type','employment_status','housing_status','source','device_os']
REFS={'bagged_anchor':['bagged_anchor_balanced_logit_a25','pooled'],
    'latest_fusion':['interaction_blend_spline_c1p0_balanced_a10','latest'],
    'recent_fusion':['margin_blend_spline_c0p1_unweighted_a25','latest'],
    'fusion':['logit_mean_xgb_d4_w1_baseline_balanced_a50','pooled'],
    'balanced':['baseline_balanced','pooled'],'unweighted':['baseline_lgbm','pooled']}
ANCHOR=REFS['bagged_anchor'][0]


def independent_names():
    return [n for n,k in REFS.values()]+[f'cb_d{d}_r{r}_w{w}' for d,r,w in product([3,4,6],[200,400],[1,100])]+[
        f'cb_anchor_d{d}_r{r}_w{w}_a{a:02d}' for d,r,w,a in product([3,4,6],[200,400],[1,100],[10,25,50])]


def independent_physical_names():
    return sorted(set(independent_names())|{'bagged_balanced_s17','bagged_balanced_s29','bagged_balanced_s43',
        'bagged_balanced_logit_mean','spline_c0p1_unweighted','spline_c1p0_balanced','xgb_d4_w1'})


def metrics(d,mask):
    r=direct_group_metrics(d['labels'],mask);tp,fp,fn=r['tp'],r['fp'],r['fn']
    r['f3']=10*tp/(10*tp+fp+9*fn) if 10*tp+fp+9*fn else 0.
    r['average_precision']=float(average_precision_score(d['labels'],d['scores'])) if np.any(d['labels']) else 0.
    r['roc_auc']=float(roc_auc_score(d['labels'],d['scores'])) if len(np.unique(d['labels']))==2 else None
    return r


def audit_tables(c,frame,h=None,generated=None):
    h=run.history(c,physical=True) if h is None else h
    generated={p[:-4]:pd.read_csv(Path(c['output_dir'])/p) for p in run.TABLES} if generated is None else generated
    names=independent_names();physical=independent_physical_names()
    if set(h)!=set(physical) or any(set(v)!={4,5,6} for v in h.values()):raise ValueError('Independent physical grid differs')
    old_bases=set(physical)-set(names)|{'baseline_balanced','baseline_lgbm'}
    old_bases.discard('bagged_balanced_logit_mean')
    for n,m in product(physical,[4,5,6]):
        d=h[n][m];ids=frame.index[frame.month==m].to_numpy()
        if not np.array_equal(d['row_ids'],ids) or not np.array_equal(d['labels'],frame.loc[ids,'fraud_bool'].to_numpy()):raise ValueError('Independent population differs')
        if np.asarray(d['scores']).shape!=ids.shape or not np.isfinite(d['scores']).all():raise ValueError('Independent finite score domain differs')
        if n in old_bases and ((d['scores']<0)|(d['scores']>1)).any():raise ValueError('Original probability domain differs')
    cf,summary,pooled=[generated[n] for n in ['crossfit','summary','pooled']]
    keys(cf,['candidate','policy','eval_month'],product(names,POLICIES,[5,6]))
    keys(summary,['candidate','policy'],product(names,POLICIES));keys(pooled,['candidate','policy'],product(names,POLICIES))
    rows=[];masks={}
    for r in cf.to_dict('records'):
        n,k,m=[r[v] for v in ['candidate','policy','eval_month']];rule=independent_rule(h[n],k,m)
        if json.loads(r['parameters'])!=rule or str(r['source_months'])!=','.join(map(str,rule['source_months'])):raise ValueError('Independent chronology/rule differs')
        masks[n,k,m]=independent_mask(h[n][m],rule);direct=metrics(h[n][m],masks[n,k,m]);close(r,direct)
        rows.append(dict(candidate=n,policy=k,eval_month=m,**direct))
    columns=['f2','f1','f3','precision','recall','flagged_fraction','average_precision','tp','fp','fn']
    ds=pd.DataFrame(rows).groupby(['candidate','policy'],as_index=False)[columns].mean().sort_values(
        ['f2','recall','precision','candidate','policy'],ascending=[False,False,False,True,True]).reset_index(drop=True)
    if list(summary[['candidate','policy']].itertuples(index=False,name=None))!=list(ds[['candidate','policy']].itertuples(index=False,name=None)):raise ValueError('Actual macro F2 order differs')
    for i,r in summary.iterrows():close(r,ds.iloc[i][columns].to_dict())
    for r in pooled.to_dict('records'):
        n,k=r['candidate'],r['policy'];y=np.concatenate([h[n][m]['labels'] for m in [5,6]]);p=np.concatenate([masks[n,k,m] for m in [5,6]])
        close(r,{k:v for k,v in direct_group_metrics(y,p).items() if k not in ['n','fraud']})
    factors=read_config(run.planning_evidence_path())['contrasts']
    contrast=generated['contrasts'];keycols=['contrast','candidate','policy','reference','reference_policy','factor_details']
    keys(contrast,keycols,[(r['kind'],r['candidate'],r['policy'],r['comparator'],r['comparator_policy'],json.dumps(r['factors'],separators=(',',':'))) for r in factors])
    index=ds.set_index(['candidate','policy'])
    for r in contrast.to_dict('records'):
        a,b=index.loc[(r['candidate'],r['policy'])],index.loc[(r['reference'],r['reference_policy'])]
        close(r,{v+'_difference':float(a[v]-b[v]) for v in ['f2','recall','precision','flagged_fraction','average_precision']})
    for table,nameset in [('physical_score_diagnostics',physical),('score_diagnostics',names)]:
        keys(generated[table],['candidate','eval_month'],product(nameset,[4,5,6]))
        for r in generated[table].to_dict('records'):
            n,m=r['candidate'],r['eval_month'];v=h[n][m]['scores'];unique=len(np.unique(v))
            close(r,dict(n=len(v),domain='probability-like original source' if n in old_bases else 'finite margin',
                score_min=float(v.min()),score_max=float(v.max()),score_mean=float(v.mean()),unique_scores=unique,
                tied_fraction=(len(v)-unique)/len(v),outside_unit=int(((v<0)|(v>1)).sum()),
                **{'q'+str(q):float(np.quantile(v,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]}))
    native=[n for n in names if n.startswith('cb_d')]
    keys(generated['diversity'],['candidate','reference','eval_month'],product(native,[ANCHOR],[4,5,6]))
    for r in generated['diversity'].to_dict('records'):
        x,y=h[r['candidate']][r['eval_month']]['scores'],h[ANCHOR][r['eval_month']]['scores'];constant=bool(np.ptp(x)==0 or np.ptp(y)==0)
        close(r,dict(correlation_undefined=constant,reason='constant member' if constant else 'defined',
            pearson=None if constant else float(pearsonr(x,y).statistic),spearman=None if constant else float(spearmanr(x,y).statistic)))
    overlapkeys=[]
    for n,k,m in product([n for n in names if n.startswith('cb_')],POLICIES,[5,6]):
        refs=[ANCHOR]
        if n.startswith('cb_anchor_'):refs.append('cb_'+n.split('cb_anchor_')[1].rsplit('_a',1)[0])
        overlapkeys.extend((n,k,m,b) for b in refs)
    keys(generated['decision_overlap'],['candidate','policy','eval_month','reference'],overlapkeys)
    for r in generated['decision_overlap'].to_dict('records'):
        n,k,m,b=[r[v] for v in ['candidate','policy','eval_month','reference']];a,other=masks[n,k,m],masks[b,k,m];y=h[n][m]['labels'].astype(bool)
        pairs=a.astype(int)*2+other.astype(int);expected={}
        for tag,take in [('fraud',y),('legitimate',~y)]:
            counts=np.bincount(pairs[take],minlength=4)
            expected.update({tag+'_'+v:int(counts[i]) for i,v in enumerate(['neither','second_only','first_only','both_flagged'])})
        close(r,expected)
    counts=dict(crossfit=324,summary=162,pooled=162,contrasts=1728,physical_score_diagnostics=183,score_diagnostics=162,diversity=36,decision_overlap=504)
    if any(len(generated[k])!=v for k,v in counts.items()):raise ValueError('Independent table cardinality differs')
    return dict(counts=counts,winner=str(ds.iloc[0].candidate),policy=str(ds.iloc[0].policy),development_f2=float(ds.iloc[0].f2))


def audit_native(c,fp,frame,base,r):
    n,m,phase=r['candidate'],r['eval_month'],r['phase'];match=re.fullmatch(r'cb_d(3|4|6)_r(200|400)_w(1|100)',n)
    if match is None:raise ValueError('Independent native identity differs')
    depth,rounds,cost=map(int,match.groups());tr=frame.index[frame.month<m];ev=frame.index[frame.month==m];x=base.loc[tr];y=frame.loc[tr,'fraud_bool'].to_numpy()
    numeric=[k for k in x if k not in CATS];cats=[k for k in x if k in CATS];medians=x[numeric].median().fillna(0)
    pre=dict(columns=list(x),categoricals=cats,numeric=numeric,levels={k:sorted(x[k].dropna().astype(str).unique()) for k in cats},medians={k:float(v) for k,v in medians.items()})
    identity=dict(train_rows=len(tr),train_fraud=int(y.sum()),train_months=sorted(frame.loc[tr,'month'].unique().tolist()),
        train_ids_sha256=hashlib.sha256(np.asarray(tr,dtype='<i8').tobytes()).hexdigest(),train_labels_sha256=hashlib.sha256(np.asarray(y,dtype=np.int8).tobytes()).hexdigest(),
        row_cost_sha256=hashlib.sha256(np.where(y==1,cost,1).astype(np.float64).tobytes()).hexdigest(),class_costs={'0':1,'1':cost},preprocessing=pre)
    if any(r.get(k)!=v for k,v in identity.items()) or r['fingerprint']!=fp:raise ValueError('Independent native training/preprocessing/cost identity differs')
    ps=run.scores.model_paths(c,phase,m,n);artifact=joblib.load(ps[3]);adapter=artifact['adapter'];model=CatBoostClassifier();model.load_model(str(ps[2]))
    expected_params=dict(iterations=rounds,depth=depth,class_weights=[1,cost],cat_features=CATS,learning_rate=.05,l2_leaf_reg=5,border_count=128,max_ctr_complexity=1,
        loss_function='Logloss',task_type='CPU',boosting_type='Plain',grow_policy='SymmetricTree',random_seed=2026,thread_count=6,use_best_model=False,allow_writing_files=False,verbose=False)
    if r['requested_model_params']!=expected_params or artifact['requested_model_params']!=expected_params:raise ValueError('Independent requested parameters differ')
    effective=json.loads(json.dumps(model.get_all_params(),allow_nan=False))
    if effective!=r['effective_model_params'] or effective!=artifact['effective_model_params']:raise ValueError('Independent effective parameters differ')
    for k in ['depth','iterations','class_weights','l2_leaf_reg','border_count','max_ctr_complexity','loss_function','task_type','boosting_type','grow_policy','random_seed','use_best_model']:
        if effective.get(k)!=expected_params[k]:raise ValueError('Independent effective recipe differs: '+k)
    if effective['learning_rate']!=float(np.float32(.05)) or effective.get('auto_class_weights','None')!='None' or model.tree_count_!=rounds or list(model.classes_)!=[0,1]:raise ValueError('Independent costs/rounds/classes differ')
    if model.feature_names_!=list(x) or model.get_cat_feature_indices()!=[list(x).index(k) for k in CATS]:raise ValueError('Independent native schema differs')
    if adapter.catboost is not True or any(getattr(adapter,k)!=pre[k] for k in ['columns','numeric','categoricals','levels']) or not adapter.medians.equals(medians):raise ValueError('Independent adapter differs')
    z=base.loc[ev].copy()
    for k in cats:z[k]=z[k].fillna('__MISSING__').astype(str)
    z[numeric]=z[numeric].fillna(medians)
    pd.testing.assert_frame_equal(z,adapter.transform(base.loc[ev]))
    actual=np.asarray(model.predict(z,prediction_type='RawFormulaVal',thread_count=6));saved=prediction(c,phase,m,n)
    if not np.isfinite(actual).all() or not np.allclose(actual,saved['scores'],atol=1e-12,rtol=0):raise ValueError('Independent saved native inference differs')
    if not np.allclose(expit(actual),model.predict_proba(z,thread_count=6)[:,1],atol=1e-12,rtol=0):raise ValueError('Independent sigmoid consistency differs')
    if r['warnings'] or r['train_row_rounds']!=len(tr)*rounds or r['fitted_tree_count']!=rounds or r['model_bytes']!=sum(p.stat().st_size for p in ps[2:]):raise ValueError('Independent measured native work differs')
    if r['fit_seconds']<0 or r['peak_process_rss_bytes']<=0:raise ValueError('Invalid actual native resource measurements')


def audit_actual_models(c,fp,frame,base,phase='validation',names=None):
    records,ds,work=run.prepare(c,fp,frame,base,phase,names,allow_missing=False)
    for r in records:
        if r['candidate'].startswith('cb_d'):audit_native(c,fp,frame,base,r)
    months=[4,5,6] if phase=='validation' else [7]
    audit_compositions(c,fp,months,phase,[n for n,k in REFS.values()])
    names=independent_names() if names is None else names
    for n,m in product(names,months):
        match=re.fullmatch(r'cb_anchor_(d(?:3|4|6)_r(?:200|400)_w(?:1|100))_a(10|25|50)',n)
        if match is None:continue
        suffix,a=match.groups();a=int(a)/100;anchor=prediction(c,phase,m,ANCHOR);native=prediction(c,phase,m,'cb_'+suffix);saved=prediction(c,phase,m,n)
        for d in [native,saved]:
            if not np.array_equal(d['row_ids'],anchor['row_ids']) or not np.array_equal(d['labels'],anchor['labels']):raise ValueError('Independent blend population differs')
        actual=(1-a)*np.asarray(anchor['scores'],dtype=np.float64)+a*np.asarray(native['scores'],dtype=np.float64)
        if not np.array_equal(saved['scores'],actual):raise ValueError('Independent associated margin arithmetic differs')
    run.exact_artifact_grid(c,phase,names)
    return records,ds,work


def static_work(counts):return {k:v for k,v in counts.items() if not k.startswith('invocation_')}


def audit_work(c,records):
    out=Path(c['output_dir']);table=pd.read_csv(out/'model_work.csv');keys(table,['candidate','eval_month'],[(r['candidate'],r['eval_month']) for r in records]);lookup={(r['candidate'],r['eval_month']):r for r in records}
    for row in table.to_dict('records'):
        r=lookup[row['candidate'],row['eval_month']]
        if json.loads(row['original_record'])!=r:raise ValueError('Original model work record differs')
        close(row,dict(kind='native' if r['candidate'].startswith('cb_d') else 'reference',reused=bool(r.get('reused')),
            **{k:r.get(k) for k in ['train_rows','train_fraud','fitted_tree_count','train_row_rounds','fit_seconds','peak_process_rss_bytes']}))
    native=[r for r in records if r['candidate'].startswith('cb_d')]
    for filename,fields in [('native_parameters.json',['requested_model_params','effective_model_params']),('native_preprocessing.json',['preprocessing'])]:
        actual=read_config(out/filename);expected=[dict(candidate=r['candidate'],eval_month=r['eval_month'],**(
            dict(requested=r[fields[0]],effective=r[fields[1]]) if len(fields)==2 else dict(preprocessing=r['preprocessing']))) for r in native]
        if actual!=expected:raise ValueError('Complete native parameter/preprocessing evidence differs')
    return dict(model_work_rows=len(table),native_parameter_rows=len(native),native_preprocessing_rows=len(native))


def audit_development(config_path=run.CONFIG,*,write=True):
    c,fp=run.initialize(config_path);run.execution_gate(c,fp);out=Path(c['output_dir'])
    if write and (out/'selection.json').exists():raise ValueError('Frozen independent audit cannot be overwritten')
    done=read_config(out/'validation_complete.json')
    if done['fingerprint']!=fp:raise ValueError('Development identity differs')
    run.verify_hashes({str(out/p):h for p,h in done['files'].items()});frame=load_data(c);frame=frame[frame.month<7];start=time.perf_counter()
    records,ds,work=audit_actual_models(c,fp,frame,features(frame));counts=static_work(run.work_accounting(c,records,ds,work));prep=read_config(out/'preparation.json')
    if len(records)!=60 or len(ds)!=123 or any(prep.get(k)!=v for k,v in counts.items()):raise ValueError('Complete measured development work differs')
    result=audit_tables(c,frame);result.update(audit_work(c,records));index=pd.read_csv(out/'summary.csv').set_index(['candidate','policy'])
    previous=pd.read_csv(Path(read_config(c['parent_config'])['output_dir'])/'summary.csv').set_index(['candidate','policy'])
    for role,nk in REFS.items():close(index.loc[tuple(nk)],previous.loc[tuple(nk)][['f2','precision','recall','flagged_fraction']].to_dict())
    run.guard(c);result.update(fingerprint=fp,validation_sha256=digest(out/'validation_complete.json'),saved_model_checks=60,actual_source_models_audited=True,
        independent_preprocessing_inference_costs=True,independent_policies_metrics_factors=True,elapsed_seconds=time.perf_counter()-start,created_at_utc=run.now())
    if write:write_json(out/'independent_development_audit.json',result)
    return result


def audit_final(c,s,frame,base):
    out=Path(c['output_dir']);names=sorted({n for n,k in s['choices'].values()});sha=digest(out/'selection.json')
    records,ds,work=audit_actual_models(c,s['fingerprint'],frame,base,'exploratory',names)
    access=read_config(out/'exploratory_access.json');done=read_config(out/'transfer_complete.json')
    if access['selection_sha256']!=sha or done['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc'] or done['table_sha256']!=digest(out/'exploratory.csv'):raise ValueError('Transfer access/evidence differs')
    actual=static_work(run.work_accounting(c,records,ds,work));bounds=s['final_fit_bounds'];grid=run.exact_artifact_grid(c,'exploratory',names)
    if any(done.get(k)!=v for k,v in actual.items()) or len(records)!=bounds['unique_base_results'] or grid['prediction_artifacts']!=bounds['physical_scores'] or grid['model_files']!=bounds['model_files']:raise ValueError('Frozen transfer work differs')
    data={a:prediction(c,'exploratory',7,n) for a,(n,k) in s['choices'].items()};h=run.history(c);final=pd.read_csv(out/'exploratory.csv');masks={};keys(final,['role'],[(a,) for a in s['choices']])
    for a,(n,k) in s['choices'].items():
        rule=independent_rule(h[n],k,7);row=final[final.role==a].iloc[0]
        if s['rules'][a]!=rule or json.loads(row.parameters)!=rule or row.candidate!=n or row.policy!=k:raise ValueError('Independent frozen role/rule differs')
        masks[a]=independent_mask(data[a],rule);close(row,metrics(data[a],masks[a]))
    return records,data,masks,final


def uncertainty(c,s,cf,h,data,masks):
    result=paired_uncertainty(c,s,cf,h,data,masks);di,dd,fi,fd=result;roles=len(s['choices'])
    if roles not in [8,22] or (len(di),len(fi),len(dd),len(fd))!=(3*(roles-1),roles-1,1000,500) or c['bootstrap_replicates']!=500:raise ValueError('All 1500 role-specific paired draws required')
    return result


def curve_rows(summary):
    index=summary.set_index(['candidate','policy']);rows=[]
    for d,r,w,k,a in product([3,4,6],[200,400],[1,100],POLICIES,[0,10,25,50,100]):
        n=ANCHOR if a==0 else f'cb_d{d}_r{r}_w{w}' if a==100 else f'cb_anchor_d{d}_r{r}_w{w}_a{a:02d}'
        rows.append(dict(depth=d,iterations=r,positive_class_cost=w,policy=k,alpha=a/100,candidate=n,f2=float(index.loc[(n,k),'f2'])))
    return pd.DataFrame(rows)


def audit_figure(summary,contrasts,di,fi,dest):
    points=pd.read_csv(dest/'plotted_operating_points.csv');curves=pd.read_csv(dest/'plotted_alpha_curves.csv');effects=pd.read_csv(dest/'plotted_matched_effects.csv');refs=pd.read_csv(dest/'plotted_reference_intervals.csv')
    keys(points,['candidate','policy'],product(independent_names(),POLICIES));index=summary.set_index(['candidate','policy'])
    for r in points.to_dict('records'):close(r,index.loc[(r['candidate'],r['policy'])][['f2','precision','recall']].to_dict())
    wanted=curve_rows(summary);pd.testing.assert_frame_equal(curves,wanted,check_exact=False,atol=1e-12,rtol=0)
    matched=contrasts[contrasts.contrast.isin(['matched_depth','matched_rounds','matched_cost'])].reset_index(drop=True)
    pd.testing.assert_frame_equal(effects,matched,check_exact=False,atol=1e-12,rtol=0)
    selected=pd.concat([di[di.scope=='macro'],fi]);selected=selected[selected.reference.isin(REFS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(refs,selected,check_exact=False,atol=1e-12,rtol=0)
    return dict(operating_points=len(points),curves=36,endpoint_rows=len(curves),matched_effects=len(effects),reference_intervals=len(refs))


def operating_figure(summary,contrasts,di,fi,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);curves=curve_rows(summary)
    matched=contrasts[contrasts.contrast.isin(['matched_depth','matched_rounds','matched_cost'])].reset_index(drop=True)
    refints=pd.concat([di[di.scope=='macro'],fi]);refints=refints[refints.reference.isin(REFS)].reset_index(drop=True)
    summary[['candidate','policy','f2','precision','recall']].to_csv(dest/'plotted_operating_points.csv',index=False)
    curves.to_csv(dest/'plotted_alpha_curves.csv',index=False);matched.to_csv(dest/'plotted_matched_effects.csv',index=False);refints.to_csv(dest/'plotted_reference_intervals.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(19,11));axes=axes.ravel()
    for k,marker in zip(POLICIES,['o','^','s']):
        g=summary[summary.policy==k];artist=axes[0].scatter(g.recall,g.precision,marker=marker,s=16,alpha=.65,label=k)
        if not np.allclose(artist.get_offsets(),g[['recall','precision']].to_numpy(),atol=1e-12,rtol=0):raise ValueError('Rendered operating coordinates differ')
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 162 chronological operating points');axes[0].legend(fontsize=8)
    for ax,k in zip(axes[1:4],POLICIES):
        for (d,r,w),g in curves[curves.policy==k].groupby(['depth','iterations','positive_class_cost']):
            line,=ax.plot(g.alpha,g.f2,marker='.',linewidth=1,label=f'd{d} / {r} rounds / cost {w}')
            if not np.array_equal(line.get_xdata(),g.alpha) or not np.array_equal(line.get_ydata(),g.f2):raise ValueError('Rendered alpha curve differs')
        ax.set(title=k+' policy: 12 matched curves',xlabel='Native margin weight',ylabel='Macro development F2');ax.set_xticks([0,.1,.25,.5,1]);ax.legend(fontsize=6,ncol=2)
    for i,(kind,g) in enumerate(matched.groupby('contrast',sort=True)):
        axes[4].scatter(np.full(len(g),i),g.f2_difference,s=9,alpha=.4,label=kind.replace('matched_',''))
    axes[4].axhline(0,color='black',linewidth=.7);axes[4].set(xticks=[0,1,2],xticklabels=['cost 100−1','depth higher−lower','rounds 400−200'],ylabel='Matched macro F2 difference',title='All matched effects, including losses and ties')
    for offset,scope in [(-.12,'macro'),(.12,'7_consumed')]:
        g=refints[refints.scope==scope].set_index('reference').loc[list(REFS)];y=g.difference.to_numpy();lo=g.ci_low.to_numpy();hi=g.ci_high.to_numpy();x=np.arange(6)+offset
        axes[5].vlines(x,lo,hi,colors='tab:blue' if scope=='macro' else 'tab:orange');axes[5].scatter(x,y,s=18,label=scope)
    axes[5].axhline(0,color='black',linewidth=.7);axes[5].set(xticks=np.arange(6),xticklabels=list(REFS),ylabel='Selected minus reference F2',title='Conditional paired 95% intervals');axes[5].tick_params(axis='x',rotation=30);axes[5].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'shallow_catboost_operating.png',dpi=160);plt.close(fig)
    return audit_figure(summary,contrasts,di,fi,dest)


def report(config_path=run.CONFIG):
    c,s=run.load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    dev=audit_development(config_path,write=False);frame=load_data(c);records,data,masks,final=audit_final(c,s,frame,features(frame))
    cf=pd.read_csv(out/'crossfit.csv');summary=pd.read_csv(out/'summary.csv');contrasts=pd.read_csv(out/'contrasts.csv');h=run.history(c)
    di,dd,fi,fd=uncertainty(c,s,cf,h,data,masks)
    for n,t in [('development_intervals',di),('development_draws',dd),('consumed_intervals',fi),('consumed_draws',fd)]:t.to_csv(dest/(n+'.csv'),index=False)
    for name in run.TABLES+run.WORK_TABLES+['selection.json','validation_complete.json','independent_development_audit.json','preparation.json','exploratory.csv','exploratory_access.json','transfer_complete.json','checks.json','declaration.json','preservation.json','contract_verification.json','resource_preflight.json','source_miniature_gate.json','older_miniature_audit.json','miniature_gate.json']:
        shutil.copyfile(out/name,dest/name)
    shutil.copyfile(config_path,dest/'config.json');write_json(dest/'transfer_provenance.json',records)
    figure=operating_figure(summary,contrasts,di,fi,dest);w=summary.iloc[0];lookup=summary.set_index(['candidate','policy']);gains={a:float(w.f2-lookup.loc[tuple(nk),'f2']) for a,nk in REFS.items()};fg={r.reference:float(r.difference) for r in fi.itertuples()}
    achieved=all(gains[a]>0 and fg[a]>0 for a in REFS);audit=dict(fingerprint=s['fingerprint'],selection_sha256=digest(out/'selection.json'),development=dev,
        final_models=len(records),visible_roles=len(s['choices']),development_gain=gains,consumed_gain=fg,measured_gain_against_all_six_in_both_scopes=achieved,figure=figure,
        uncertainty='1500 paired fixed-decision draws; excludes adaptive search, retraining, temporal/generative variation and multiplicity; month7 already consumed')
    write_json(dest/'audit.json',audit)
    lines=['# Shallow CatBoost study findings','',f'Selected **{w.candidate} / {w.policy}**, macro chronological F2 **{w.f2:.9f}**. Measured improvement against all six fixed references in development and consumed transfer: **{achieved}**. Completing this finite search does not itself achieve the broader performance goal.','',
        'Twelve native recipes cross depth 3/4/6, 200/400 rounds and fixed positive cost 1/100. Thirty-six fixed raw-margin blends and six references yield 54 strategies and 162 policy cases. Positive-class F2 uses beta=2 without a review quota or precision floor. All settings are adaptive analyst selections, open to group review.','',
        'All 29 raw predictors use training-only numerical medians and native categorical inputs. RawFormulaVal scores and fixed blends are finite margins, not calibrated fraud probabilities. Original references retain their frozen representations. Depth/round comparisons do not equalize runtime, storage or categorical-statistic work.','',
        'Models for months 4/5/6 use strictly earlier training rows. Month-5 rules use month 4; month-6 rules use months 4/5 or latest 5. Selection is actual macro F2 across months 5/6. Transfer roles and exact dependencies were frozen before further month-7 access. Month 7 is an already-consumed exploratory evaluation, not independent confirmation.','',
        '| Fixed reference | Macro F2 difference | Consumed F2 difference |','| --- | ---: | ---: |']
    lines += [f'| {a} | {gains[a]:+.9f} | {fg[a]:+.9f} |' for a in REFS]
    lines += ['', 'All operating results, matched effects including negative values and ties, score/diversity/overlap diagnostics, requested/effective parameters, preprocessing and actual model work are exported beside this report. Current/historical methodological test calls, fourteen production miniature fits, full native fits, exact reference reproductions and verified reuses are separately recorded in their ledgers.', '',
        'The 1500 paired row-bootstrap draws keep policies and decisions fixed. Intervals are conditional on the observed synthetic cases and exclude retraining, adaptive recipe/source/policy/selection search, temporal/generative variation and multiple-comparison uncertainty. Clear-gain claims require positive lower bounds in the named scope. Repeated consumed-month access further limits generalization claims.', '',
        'Codex assisted with design, implementation, tests and interpretation. Primary documentation: [CatBoost raw predictions](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict), [training parameters](https://catboost.ai/docs/en/references/training-parameters/common), [margin/probability output](https://catboost.ai/docs/en/concepts/output-data_model-value-output), and [native serialization](https://catboost.ai/docs/en/concepts/python-reference_catboostranker_save_model). No public notebook was copied.']
    (dest/'findings.md').write_text('\n'.join(lines)+'\n');run.guard(c)
    write_json(dest/'report_hashes.json',{str(p):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='report_hashes.json' and p.suffix!='.log'})
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('step',choices=['audit_development','report']);parser.add_argument('--config',default=run.CONFIG);args=parser.parse_args();print(json.dumps(globals()[args.step](args.config),indent=2))
