"""Reconstruct additive study evidence, fixed-decision uncertainty and effects."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score

from .common import digest, features, load_data, read_config, write_json
from .recall_weighted_metrics import decisions, learn, measure
from .direct_ranking_report import development, intervals
from .score_fusion_report import audit_intervals
from .improvement_metrics import confusion_metrics
from .spline_inputs import SMOOTH
from . import score_fusion_study as old
from .spline_study import (CONFIG,FUSION,REFERENCE_IDS,TABLES,load_selection,audit_grid,
    history,crossfit,verify_table,order_summary,contrast_factors,reference_bases,audit_new,
    final_data,model_path,paths,guard)


def markdown(table):
    def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(table.columns)+' |\n| '+' | '.join(['---']*len(table.columns))+' |\n'+\
        '\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in table.itertuples(index=False,name=None))


def independent_metrics(d,mask,row):
    y=d['labels'];tn,fp,fn,tp=confusion_matrix(y,mask,labels=[0,1]).ravel()
    values=dict(tn=tn,fp=fp,fn=fn,tp=tp,precision=precision_score(y,mask,zero_division=0),
        recall=recall_score(y,mask,zero_division=0),f1=fbeta_score(y,mask,beta=1,zero_division=0),
        f2=fbeta_score(y,mask,beta=2,zero_division=0),f3=fbeta_score(y,mask,beta=3,zero_division=0),
        flagged_fraction=float(mask.mean()))
    if any(not np.isclose(row[k],v,atol=1e-12,rtol=0) for k,v in values.items()):
        raise ValueError('Independent metrics differ')


def audit_tables(c,cf,h):
    out=Path(c['output_dir']);summary=order_summary(cf);lookup=summary.set_index(['candidate','policy'])
    for r in cf.to_dict('records'):
        d=h[r['candidate']][r['eval_month']];rule=json.loads(r['parameters'])
        if rule!=learn({m:v for m,v in h[r['candidate']].items() if m<r['eval_month']},r['policy'],r['eval_month']):
            raise ValueError('Earlier-only rule differs')
        independent_metrics(d,decisions(d,rule),r)
    contrasts=pd.read_csv(out/'contrasts.csv')
    if list(contrasts[['contrast','candidate','policy','reference','reference_policy']].itertuples(index=False,name=None))!=contrast_factors(c):
        raise ValueError('Contrast family/factors differ')
    for r in contrasts.itertuples():
        a,b=lookup.loc[(r.candidate,r.policy)],lookup.loc[(r.reference,r.reference_policy)]
        for m in ['f2','precision','recall','flagged_fraction','average_precision']:
            if not np.isclose(getattr(r,m+'_difference'),a[m]-b[m],atol=1e-12,rtol=0):raise ValueError('Contrast arithmetic differs')
    for r in pd.read_csv(out/'pooled.csv').itertuples():
        g=cf[(cf.candidate==r.candidate)&(cf.policy==r.policy)];t=g[['tn','fp','fn','tp']].sum()
        expected=confusion_metrics(*[t[k] for k in ['tn','fp','fn','tp']])
        if any(not np.isclose(getattr(r,k),expected[k],atol=1e-12,rtol=0) for k in expected):raise ValueError('Pooled arithmetic differs')
    for r in pd.read_csv(out/'score_diagnostics.csv').itertuples():
        x=h[r.candidate][r.eval_month]['scores']
        if not np.allclose([len(x),x.min(),x.max(),x.mean(),len(np.unique(x)),1-len(np.unique(x))/len(x)],
            [r.n,r.score_min,r.score_max,r.score_mean,r.unique_scores,r.tied_fraction],atol=1e-12,rtol=0):raise ValueError('Score diagnostics differ')
    for name,count in [('crossfit',90),('summary',45),('pooled',45),('contrasts',237),('score_diagnostics',45),('solver',36),('feature_diagnostics',2088)]:
        if len(pd.read_csv(out/(name+'.csv')))!=count:raise ValueError('Output count differs: '+name)


def direct_draw_audit(data,masks,rng,draws):
    """Independent sampled-row implementation, separate from weighted count draws."""
    y=data['selected']['labels'].astype(bool);refs=[n for n in data if n!='selected']
    for i in range(len(draws)):
        idx=rng.integers(0,len(y),len(y));actual={}
        for n,mask in masks.items():
            p=mask[idx];labels=y[idx];tp=int((p&labels).sum());fp=int((p&~labels).sum());fn=int((~p&labels).sum())
            den=5*tp+4*fn+fp;actual[n]=5*tp/den if den else 0.
        if any(not np.isclose(draws.iloc[i][n],actual['selected']-actual[n],atol=1e-12,rtol=0) for n in refs):
            raise ValueError('Independent direct-row bootstrap differs')


def additive_curves(c,s,summary,dest):
    name=s['selected_candidate']
    if name in REFERENCE_IDS:
        name=str(summary[summary.candidate.str.startswith('spline_')].iloc[0].candidate)
        reason='best new spline; development diagnostic only; not transferred'
    else:reason='development winner; month-6 fitted representation'
    mp=model_path(c,'validation',6,name);a=joblib.load(mp);inp=a['inputs'];model=a['model']
    fixed=pd.DataFrame([inp.reference_values],columns=inp.columns);baseline=inp.transform(fixed)
    margin=float((baseline@model.coef_.T).item()+model.intercept_[0]);rows=[]
    for col in SMOOTH:
        lo,hi=inp.training_quantiles[col];x=np.linspace(lo,hi,101)
        frame=pd.concat([fixed]*len(x),ignore_index=True);frame[col]=x;matrix=inp.transform(frame)
        delta=np.asarray((matrix-baseline[np.zeros(len(x),dtype=int)])@model.coef_.T).ravel()
        if not np.allclose(delta,model.decision_function(matrix)-margin,atol=1e-12,rtol=1e-12):
            raise ValueError('Plotted additive contributions differ')
        rows += [dict(candidate=name,eval_month=6,feature=col,value=float(v),log_odds_difference=float(d),
                      reference_value=inp.reference_values[col],diagnostic=reason) for v,d in zip(x,delta)]
    table=pd.DataFrame(rows);table.to_csv(dest/'additive_curves.csv',index=False)
    cols=['name_email_similarity','income','customer_age','credit_risk_score','current_address_months_count',
          'prev_address_months_count','proposed_credit_limit','days_since_request']
    fig,axes=plt.subplots(2,4,figsize=(14,6))
    for ax,col in zip(axes.ravel(),cols):
        g=table[table.feature==col];ax.plot(g.value,g.log_odds_difference)
        ax.axhline(0,color='grey',linewidth=.6);ax.set(title=col.replace('_',' '),xlabel='Original feature value',ylabel='Log-odds difference')
    fig.suptitle(name+' — '+reason+'\nTraining 1–99% feature ranges; associations, not causal effects',fontsize=10)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(dest/'additive_effects.png',dpi=160);plt.close(fig)
    return dict(candidate=name,model_sha256=digest(mp),curve_rows=len(table),feature_count=len(SMOOTH),diagnostic=reason)


def report(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    frame=load_data(c);base=features(frame);dev=frame[frame.month<7]
    records=audit_grid(c,s['fingerprint'],dev,base.loc[dev.index]);h=history(c);cf=crossfit(c);verify_table(cf,out/'crossfit.csv')
    summary=order_summary(cf).reset_index(drop=True);verify_table(summary,out/'summary.csv');audit_tables(c,cf,h)
    access=read_config(out/'exploratory_access.json');transfer=read_config(out/'transfer_complete.json');sha=digest(out/'selection.json')
    if access['selection_sha256']!=sha or transfer['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:
        raise ValueError('Freeze/access identity differs')
    if transfer['table_sha256']!=digest(out/'exploratory.csv'):raise ValueError('Transfer table changed')
    final_records=reference_bases(c,s['fingerprint'],frame,base,'exploratory',False)
    needed={n for n,k in s['choices'].values()}
    for spec in c['candidates']:
        if spec['id'] in needed:final_records.append(audit_new(c,s['fingerprint'],frame,base,spec,7,'exploratory'))
    data=final_data(c,s);rows=[];masks={a:decisions(d,s['rules'][a]) for a,d in data.items()}
    for a,d in data.items():
        rule=s['rules'][a];metrics=measure(d,rule);independent_metrics(d,masks[a],metrics)
        rows.append(dict(role=a,candidate=s['choices'][a][0],policy=s['choices'][a][1],eval_month=7,
            parameters=json.dumps(rule,sort_keys=True),**metrics))
    final=pd.DataFrame(rows);verify_table(final,out/'exploratory.csv')
    dint,ddraws=development(c,s,cf,h);fint,fdraws=intervals(data,masks,np.random.default_rng(c['bootstrap_seed']),500,'7_consumed')
    audit_intervals(dint,ddraws,fint,fdraws,s['choices'],500)
    rng=np.random.default_rng(c['bootstrap_seed'])
    for m in [5,6]:
        d={a:h[n][m] for a,(n,k) in s['choices'].items()};masks_m={}
        for a,(n,k) in s['choices'].items():
            row=cf[(cf.candidate==n)&(cf.policy==k)&(cf.eval_month==m)].iloc[0]
            masks_m[a]=decisions(d[a],json.loads(row.parameters))
        direct_draw_audit(d,masks_m,rng,ddraws[ddraws.scope==str(m)].reset_index(drop=True))
    direct_draw_audit(data,masks,np.random.default_rng(c['bootstrap_seed']),fdraws)
    for name,table in [('development_intervals',dint),('development_draws',ddraws),('consumed_intervals',fint),
                       ('consumed_draws',fdraws.assign(replicate=np.arange(len(fdraws))))]:table.to_csv(dest/(name+'.csv'),index=False)
    for name in TABLES+['miniature.json','selection.json','validation_complete.json','exploratory_access.json','transfer_complete.json','exploratory.csv']:
        shutil.copyfile(out/name,dest/name)
    shutil.copyfile(config_path,dest/'config.json');write_json(dest/'run_provenance.json',records);write_json(dest/'transfer_provenance.json',final_records)
    for stage,months in [('validation',[4,5,6]),('exploratory',[7])]:
        for m in months:old_record=read_config(paths(c,stage,m,FUSION)[0]);old.verify_origin(old_record)
    curve=additive_curves(c,s,summary,dest)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));types={x['id']:x.get('representation','reference') for x in c['candidates']}
    for rep in ['linear','spline','reference']:
        g=summary[summary.candidate.map(lambda n:types.get(n,'reference'))==rep]
        axes[0].scatter(g.recall,g.precision,label=rep,s=30,alpha=.8)
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 45 strategy/policy comparisons');axes[0].legend()
    contrasts=pd.read_csv(out/'contrasts.csv');g=contrasts[contrasts.contrast=='spline minus linear']
    specs={x['id']:x for x in c['candidates']}
    for k in c['policies']:
        for weight,marker in [(None,'o'),('balanced','x')]:
            sub=g[(g.policy==k)&g.candidate.map(lambda n:specs[n]['class_weight']==weight)]
            axes[1].scatter(sub.candidate.map(lambda n:specs[n]['C']),sub.f2_difference,marker=marker,label=k+'/'+str(weight))
    axes[1].set(xlabel='Regularization C',ylabel='Macro F2: spline minus linear',title='18 matched representation comparisons',xscale='log')
    axes[1].axhline(0,color='grey',linewidth=.7);axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(dest/'spline_operating.png',dpi=160);plt.close(fig)
    winner=summary.iloc[0];lookup=summary.set_index(['candidate','policy'])
    dg={r:float(winner.f2-lookup.loc[(n,'pooled'),'f2']) for r,n in [('fusion',FUSION),('balanced','baseline_balanced'),('unweighted','baseline_lgbm')]}
    fg={r.reference:float(r.difference) for r in fint.itertuples()};measured=all(dg[n]>0 and fg[n]>0 for n in dg)
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,measured_gain_on_both_scopes=measured,
        development_gain=dg,consumed_gain=fg,independent_metric_checks=True,direct_draws_checked=1500,
        conditional_clear_development_gain=all(float(dint[(dint.scope=='macro')&(dint.reference==n)].iloc[0].ci_low)>0 for n in dg),
        conditional_clear_consumed_gain=all(float(fint[fint.reference==n].iloc[0].ci_low)>0 for n in dg),
        additive_curve=curve,report_source_sha256=digest('baf/spline_report.py'),preservation_checked=True)
    write_json(dest/'audit.json',audit);complete=read_config(out/'validation_complete.json')
    text=f'''# Additive spline-logistic findings

Selected **{winner.candidate} / {winner.policy}**, macro chronological F2 **{winner.f2:.6f}**. Positive measured F2 differences against fusion, balanced and unweighted references on both development and consumed transfer: **{measured}**. This is exploratory evidence after adaptive development reuse; month 7 was already consumed and supplies no independent confirmation. Search completion alone does not achieve the performance goal.

## Design and verified inference

Twelve paired logistic recipes cross linear/spline representation, C 0.01/0.1/1 and unweighted/balanced training. They use the same 29 original decoded predictors and earlier training populations. Seventeen numeric effects use observed-training quantile knots, exact duplicate removal, cubic bases and constant extrapolation; seven discrete numeric fields stay linear. Both representations retain training medians, all 24 numeric missing indicators, variance scaling without centering and the same five categorical one-hot vocabularies. Degenerate smooth columns have explicit constant/all-missing fallbacks. No target/month inputs, interactions or target encoding are used. Missing/new-category/range handling is saved with the model.

There were **{complete['new_logistic_fits']} new full development fits**, **{complete['reference_reuses']} verified base reuses** and **{complete['reference_fits']} reference fits**, plus three fixed reference fusions. The separate miniature gate made 24 fits. L-BFGS/L2 settings, actual iterations, warnings, dimensions, sparse sizes, time and process memory are exported. Every saved model reproduces positive-class scores and coefficient/intercept margins. Weighted outputs are not asserted calibrated fraud probabilities. All ten older source/config/software/selection identities remain preserved.

## Forward actual performance

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

All 90 forward rows, 45 macro summaries, 45 pooled-confusion rows, 237 matched contrasts and 45 selectable score summaries remain visible. There are 45 selectable prediction artifacts plus three nonselectable XGBoost dependencies. Month-5 policies use month 4; month-6 policies use months 4/5 or latest month 5. Rank fractions maximize earlier F2, with no 1% quota or precision floor. Actual macro F2 selects the winner; pooled F2 and AP are descriptive. Equal C does not imply equal effective model complexity across representations.

## Conditional development differences

{markdown(dint)}

## Already-consumed month 7

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

{markdown(fint)}

Only the frozen winner, same-strategy pooled rule, three existing references and winner-type-matched controls were transferred. Fit/reuse counts are in transfer_complete.json; aliases retain comparisons while models are deduplicated. Selection, dependencies and earlier-only final rules were frozen before recorded further access. No losing-setting sweep or hindsight threshold was computed. Failed comparisons remain visible. Clear/confirmed gain language requires a positive conditional interval on the named scope; neither such an interval nor a positive estimate restores an independent test.

## Features, uncertainty and reproducibility

feature_blocks.json records knots, fallbacks, medians, vocabularies, scaler statistics and transformed names. feature_diagnostics.csv records training/prediction missingness, unseen categories and boundary exceedances for every new development fit. coefficients.csv and additive_curves.csv support the checked effects figure; curves describe log-odds associations over training-derived ranges, not causal effects. The curve model and diagnostic scope are recorded in audit.json.

Intervals use 500 paired fixed-decision row draws per month with seed 1027; development macro differences average separately resampled months. All 1500 draws are independently reconstructed using direct sampled Boolean rows. Models, selection, policies and realized decisions remain fixed. Intervals exclude adaptive multiple-study search/selection, retraining, temporal/generative uncertainty and multiplicity adjustment. Results are from one small synthetic benchmark with eight months.

Reproduction uses configs/spline-logistic.json, docs/spline-logistic-protocol.md and the sequential miniature/validate/freeze/exploratory/report commands in README. No package or old protocol was changed. Codex assisted with planning, implementation, analysis and checks; official scikit-learn SplineTransformer and LogisticRegression documentation supplied API definitions. No public notebook was copied. Settings remain analyst-selected, not group-approved operating policy.

![Operating results](spline_operating.png)

![Additive effects](additive_effects.png)
'''
    (dest/'findings.md').write_text(text);guard(c);print(text[:900],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=CONFIG)
    report(parser.parse_args().config)
