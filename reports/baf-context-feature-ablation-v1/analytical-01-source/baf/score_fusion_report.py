"""Reconstruct fixed fusion evidence and conditional fixed-decision uncertainty."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import digest,features,load_data,read_config,write_json
from .recall_weighted_metrics import decisions,measure,learn
from .calibrated_policy_report import independent_check,markdown
from .direct_ranking_report import intervals,development
from .improvement_metrics import confusion_metrics
from .score_fusion import choices_for
from .score_fusion_study import (CONFIG,load_selection,audit_grid,history,crossfit,verify_table,order_summary,
    prepare_references,audit_model,ensure_score,final_data,export_comparisons)


def audit_tables(c,cf,h):
    out=Path(c['output_dir']);summary=order_summary(cf);lookup=summary.set_index(['candidate','policy'])
    for row in cf.itertuples():
        d=h[row.candidate][row.eval_month];rule=json.loads(row.parameters)
        expected=learn({m:v for m,v in h[row.candidate].items() if m<row.eval_month},row.policy,row.eval_month)
        if rule!=expected:raise ValueError('Earlier-only policy reconstruction differs')
        independent_check(d,decisions(d,rule),row._asdict())
    for r in pd.read_csv(out/'contrasts.csv').itertuples():
        a=lookup.loc[(r.candidate,r.policy)];b=lookup.loc[(r.reference,r.reference_policy)]
        for k in ['f2','precision','recall','flagged_fraction','average_precision']:
            if not np.isclose(getattr(r,k+'_difference'),a[k]-b[k],atol=1e-12,rtol=0):raise ValueError('Contrast arithmetic differs')
    for r in pd.read_csv(out/'pooled.csv').itertuples():
        g=cf[(cf.candidate==r.candidate)&(cf.policy==r.policy)];t=g[['tn','fp','fn','tp']].sum()
        expected=confusion_metrics(*[t[k] for k in ['tn','fp','fn','tp']])
        for k in ['tn','fp','fn','tp','f2','precision','recall']:
            if not np.isclose(getattr(r,k),expected[k],atol=1e-12,rtol=0):raise ValueError('Pooled arithmetic differs')
    byid={x['id']:x for x in c['strategies']}
    expected_factors=[]
    for spec in c['strategies']:
        n=spec['id']
        for policy in c['policies']:
            expected_factors.extend([('old pooled reference',n,policy,ref,'pooled') for ref in ['baseline_balanced','baseline_lgbm']])
            if policy!='pooled':expected_factors.append(('policy minus own pooled',n,policy,n,'pooled'))
            if spec['kind']=='logit':expected_factors.append(('logit minus raw',n,policy,spec['members'][0],policy))
            if spec['kind'].endswith('_mean'):
                for member in spec['members']:
                    ref=('logit_' if spec['kind']=='logit_mean' else '')+member
                    expected_factors.append(('fusion minus matched member',n,policy,ref,policy))
                if spec['kind']=='logit_mean':expected_factors.append(('logit fusion minus raw fusion',n,policy,n.replace('logit_mean_','raw_mean_',1),policy))
    contrasts=pd.read_csv(out/'contrasts.csv')
    actual_factors=list(contrasts[['contrast','candidate','policy','reference','reference_policy']].itertuples(index=False,name=None))
    if actual_factors!=expected_factors:raise ValueError('Matched contrast factors differ')
    clips=pd.read_csv(out/'clipping.csv')
    for r in clips.itertuples():
        x=h[r.candidate][r.eval_month]['scores'];e=c['epsilon']
        expected=dict(n=len(x),epsilon=e,input_min=x.min(),input_max=x.max(),below_epsilon=int((x<e).sum()),
                      above_one_minus_epsilon=int((x>1-e).sum()),changed_count=int(((x<e)|(x>1-e)).sum()))
        if any(not np.isclose(getattr(r,k),v,atol=1e-12,rtol=0) for k,v in expected.items()):raise ValueError('Clipping diagnostic differs')
    for spec in c['strategies']:
        for month in c['validation_months']:
            arrays=[h[n][month]['scores'] for n in spec['members']]
            if spec['kind'] in ['logit','logit_mean']:
                arrays=[np.log(np.clip(x,c['epsilon'],1-c['epsilon'])/(1-np.clip(x,c['epsilon'],1-c['epsilon']))) for x in arrays]
            actual=arrays[0] if len(arrays)==1 else spec['alpha']*arrays[0]+(1-spec['alpha'])*arrays[1]
            if not np.allclose(actual,h[spec['id']][month]['scores'],atol=1e-12,rtol=1e-12):raise ValueError('Independent score arithmetic differs')
    for r in pd.read_csv(out/'score_diagnostics.csv').itertuples():
        x=h[r.candidate][r.eval_month]['scores']
        if len(x)!=r.n or len(np.unique(x))!=r.unique_scores:raise ValueError('Score counts differ')
        if not np.allclose([x.min(),x.max(),x.mean(),1-len(np.unique(x))/len(x)],
            [r.score_min,r.score_max,r.score_mean,r.tied_fraction],atol=1e-12,rtol=0):raise ValueError('Score diagnostic differs')
    for name,count in [('crossfit',366),('summary',183),('contrasts',866),('pooled',183),('score_diagnostics',183),('clipping',18)]:
        if len(pd.read_csv(out/(name+'.csv')))!=count:raise ValueError('Table count differs: '+name)


def audit_intervals(dint,ddraws,fint,fdraws,roles,replicates):
    refs=set(roles)-{'selected'}
    if len(dint)!=3*len(refs) or len(fint)!=len(refs) or len(ddraws)!=2*replicates or len(fdraws)!=replicates:
        raise ValueError('Uncertainty grid incomplete')
    for table in [dint,fint]:
        if set(table.reference)!=refs:raise ValueError('Uncertainty roles differ')
    d5=ddraws[ddraws.scope=='5'].reset_index(drop=True)
    d6=ddraws[ddraws.scope=='6'].reset_index(drop=True)
    for r in dint.itertuples():
        vals=(d5[r.reference]+d6[r.reference])/2 if r.scope=='macro' else (d5 if r.scope=='5' else d6)[r.reference]
        if not np.allclose([r.ci_low,r.ci_high],np.quantile(vals,[.025,.975]),atol=1e-12,rtol=0):raise ValueError('Development interval quantiles differ')
    for r in fint.itertuples():
        if not np.allclose([r.ci_low,r.ci_high],np.quantile(fdraws[r.reference],[.025,.975]),atol=1e-12,rtol=0):raise ValueError('Consumed interval quantiles differ')


def report(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    frame=load_data(c);base=features(frame);dev=frame[frame.month<7]
    records=audit_grid(c,s['fingerprint'],dev,base.loc[dev.index]);h=history(c);cf=crossfit(c);verify_table(cf,out/'crossfit.csv')
    summary=order_summary(cf).reset_index(drop=True);verify_table(summary,out/'summary.csv');audit_tables(c,cf,h);winner=summary.iloc[0]
    if [str(winner.candidate),str(winner.policy)]!=s['choices']['selected']:raise ValueError('Selection order differs')
    selected_spec=next(x for x in c['strategies'] if x['id']==winner.candidate)
    if s['choices']!=choices_for(selected_spec,str(winner.policy)):raise ValueError('Control inclusion differs')
    for alias,(n,k) in s['choices'].items():
        if s['rules'][alias]!=learn(h[n],k,7):raise ValueError('Frozen policy reconstruction differs')
    access=read_config(out/'exploratory_access.json');transfer=read_config(out/'transfer_complete.json');sha=digest(out/'selection.json')
    if access['selection_sha256']!=sha or transfer['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Freeze/access mismatch')
    if transfer['table_sha256']!=digest(out/'exploratory.csv'):raise ValueError('Transfer table changed')
    refids={x['id'] for x in c['references']};refs=[x for x in s['fit_specs'] if x['id'] in refids];final_records=[]
    if refs:final_records+=prepare_references(c,s['fingerprint'],frame,base,'exploratory',refs,False)
    for spec in s['fit_specs']:
        if spec['id'] not in refids:final_records.append(audit_model(c,s['fingerprint'],frame,base,spec,7,'exploratory'))
    for spec in s['score_specs']:ensure_score(c,s['fingerprint'],spec,7,'exploratory',False)
    data=final_data(c,s);rows=[];masks={}
    for alias,d in data.items():
        rule=s['rules'][alias];mask=decisions(d,rule);masks[alias]=mask;metrics=measure(d,rule);independent_check(d,mask,metrics)
        rows.append(dict(role=alias,candidate=s['choices'][alias][0],policy=s['choices'][alias][1],eval_month=7,parameters=json.dumps(rule,sort_keys=True),**metrics))
    final=pd.DataFrame(rows);verify_table(final,out/'exploratory.csv');dint,ddraws=development(c,s,cf,h)
    fint,fdraws=intervals(data,masks,np.random.default_rng(c['bootstrap_seed']),c['bootstrap_replicates'],'7_consumed')
    audit_intervals(dint,ddraws,fint,fdraws,s['choices'],c['bootstrap_replicates'])
    for name,table in [('crossfit',cf),('summary',summary),('exploratory',final),('development_intervals',dint),('development_draws',ddraws),
        ('consumed_intervals',fint),('consumed_draws',fdraws.assign(replicate=np.arange(len(fdraws))))]:table.to_csv(dest/(name+'.csv'),index=False)
    for name in ['selection.json','validation_complete.json','exploratory_access.json','transfer_complete.json',
        'contrasts.csv','pooled.csv','score_diagnostics.csv','clipping.csv']:shutil.copyfile(out/name,dest/name)
    shutil.copyfile(config_path,dest/'config.json');write_json(dest/'run_provenance.json',records);write_json(dest/'transfer_provenance.json',final_records)
    derived=[read_config(p) for p in (out/'runs').glob('*.json') if read_config(p).get('transform')]
    write_json(dest/'score_provenance.json',derived)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));types={x['id']:x['kind'] for x in c['strategies']}
    for kind in ['raw','logit','raw_mean','logit_mean']:
        g=summary[summary.candidate.map(types)==kind];axes[0].scatter(g.recall,g.precision,s=25,alpha=.75,label=kind)
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 183 strategy/policy comparisons');axes[0].legend(fontsize=8)
    matched=pd.read_csv(out/'contrasts.csv');g=matched[matched.contrast=='logit fusion minus raw fusion'].copy()
    definitions={x['id']:x for x in c['strategies']}
    for policy,color in zip(c['policies'],['tab:blue','tab:orange','tab:green']):
        for anchor,marker in [('baseline_balanced','o'),('baseline_lgbm','x')]:
            sub=g[(g.policy==policy)&g.candidate.map(lambda n:definitions[n]['members'][1]==anchor)]
            axes[1].scatter(sub.candidate.map(lambda n:definitions[n]['alpha']),sub.f2_difference,color=color,marker=marker,s=30,label=policy+'/'+anchor.replace('baseline_',''))
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].set(xlabel='Fixed XGBoost weight',ylabel='Macro F2: logit fusion minus raw fusion',title='72 matched composition comparisons');axes[1].set_xticks([.25,.5,.75]);axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(dest/'fusion_operating.png',dpi=160);plt.close(fig)
    lookup=summary.set_index(['candidate','policy']);dg={n:float(winner.f2-lookup.loc[(name,'pooled'),'f2']) for n,name in [('balanced','baseline_balanced'),('unweighted','baseline_lgbm')]}
    fg={r.reference:float(r.difference) for r in fint.itertuples()};measured=all(dg[n]>0 and fg[n]>0 for n in ['balanced','unweighted'])
    clear_dev=all(float(dint[(dint.scope=='macro')&(dint.reference==n)].iloc[0].ci_low)>0 for n in ['balanced','unweighted'])
    clear_final=all(float(fint[fint.reference==n].iloc[0].ci_low)>0 for n in ['balanced','unweighted'])
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,measured_gain_on_both_scopes=measured,development_gain=dg,consumed_gain=fg,
        conditional_clear_development_gain=clear_dev,conditional_clear_consumed_gain=clear_final,independent_metrics_checked=True,
        prior_preservation_checked=True,base_results=len(records),score_results=183,forward_rows=len(cf),macro_rows=len(summary),contrasts=866,
        raw_score_semantics='weighted base scores and raw/logit fusions are not assumed calibrated fraud probabilities',
        report_source_sha256=digest('baf/score_fusion_report.py'),uncertainty='500 paired fixed-decision draws; excludes selection/retraining/time/generative/multiplicity')
    write_json(dest/'audit.json',audit)
    best_raw=summary[summary.candidate.str.startswith('xgb_')].iloc[0]
    best_raw_fusion=summary[summary.candidate.str.startswith('raw_mean_')].iloc[0]
    best_logit_fusion=summary[summary.candidate.str.startswith('logit_mean_')].iloc[0]
    complete=read_config(out/'validation_complete.json')
    text=f'''# Fixed score-fusion study findings

Selected **{winner.candidate} / {winner.policy}** scores actual macro chronological F2 **{winner.f2:.6f}**. Best raw fusion: **{best_raw_fusion.candidate} / {best_raw_fusion.policy}**, F2 **{best_raw_fusion.f2:.6f}**. Best logit fusion: **{best_logit_fusion.candidate} / {best_logit_fusion.policy}**, F2 **{best_logit_fusion.f2:.6f}**. Best raw XGBoost control F2: **{best_raw.f2:.6f}**. Positive measured gain against both original references on development and consumed transfer: **{measured}**. Completing a search alone does not establish the performance objective.

## Design and verified inference

Seven raw controls, six single-logit controls and 48 fixed fusions produce 183 strategy/month scores and 366 forward operating rows. Members, weights 0.25/0.5/0.75 and epsilon 1e-6 were fixed before evaluation. Raw fusion averages the two original scores; logit fusion averages their clipped log-odds, retained as finite real-valued scores without a sigmoid. Single-logit controls isolate clipping/transformation from fusion. Inputs align by stable ID and labels; arithmetic uses no labels or batch ranks. Neither class-weighted base scores nor their fusions are asserted to be calibrated fraud probabilities.

Development used **{complete['xgboost_reuses']} verified XGBoost model reuses**, **{complete['new_xgboost_fits']} new XGBoost fits**, **{complete['reference_prediction_reuses']} reference prediction reuses** and **{complete['reference_prediction_refits']} reference prediction refits**. Older references had no saved development models; **{complete['reference_model_verification_fits']} separately counted exact-recipe verification fits** reconstructed their frozen predictions and supplied saved models. No new learning recipe or package was introduced. All required models reload and reproduce scores. Test fixtures are separate from these counts.

All models and preprocessing use strictly earlier training months. The 29 original predictors exclude fraud_bool, month and constant device_fraud_count. XGBoost and native31 retain training-only category vocabularies and numeric missing values; baseline LightGBM retains its own frozen pipeline. All source/config/software/selection identities from nine prior studies remain preserved. Cross-family preset differences are not a universal algorithm effect. F2 retains beta=2 without a fixed review quota or precision floor; settings are analyst-selected.

## Forward actual performance

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

All 366 monthly rows, 183 macro summaries, 183 pooled-confusion summaries and 866 contrasts remain exported, including failures and self comparisons. Month-5 policies learn from month 4; month-6 policies learn from months 4/5 or latest month 5. Actual macro F2 selects the winner, not AP or pooled F2. Nine earlier studies have adaptively reused development data.

Matched contrasts separate fusion versus each corresponding member, logit versus raw fusion, logit versus raw single, and each policy versus its own pooled rule; both old pooled references remain visible. Independent reconstruction checks every score, decision, confusion/F2 row and contrast. clipping.csv has 18 source/month rows, with epsilon, ranges and exact changed-score counts; score_diagnostics.csv has 183 range/tie summaries. Clipping can introduce ties and affect threshold transfer.

## Paired conditional development differences

{markdown(dint)}

## Already-consumed month 7: exploratory transfer

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

{markdown(fint)}

Only the actual development winner, same-strategy pooled rule, both original pooled references and type-determined member/representation controls were evaluated. Base fits are deduplicated while every role remains visible. Transfer counts and saved-model provenance are exported in transfer_complete.json and transfer_provenance.json. Freeze/control/dependency hashes and access chronology agree. No losing-setting sweep, hindsight threshold or retuning occurred. A development gain that fails transfer is not out-of-time improvement; this consumed period cannot provide fresh independent confirmation.

## Uncertainty, preservation and reproduction

Intervals use 500 paired fixed-decision row draws with seed 1027. Development resamples months separately and averages monthly F2 differences. Models, selection and realized decisions stay fixed, including rank-policy decisions; intervals exclude adaptive search/selection, retraining, temporal/generative uncertainty and multiplicity correction. Overlapping-zero intervals preclude clear/confirmed gain claims; significance does not replace positive measured differences. Exact model/score origin provenance, config/freeze/access identities and every draw/interval are exported. README gives pinned installation, verified Base access, nine selection-restoration commands and new-destination reproduction.

Codex assisted with planning, implementation, execution, methodological checks, figures and interpretation. The [official SciPy logit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logit.html) supplies the mathematical definition and endpoint behavior; epsilon, weights and study settings are our choices. Models reuse the frozen package interfaces and recipes. No public notebook was copied. These synthetic-data results are supporting coursework evidence, not a bank deployment claim.

![Operating and matched-composition diagnostics](fusion_operating.png)
'''
    (dest/'findings.md').write_text(text);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=CONFIG);a=p.parse_args();report(a.config)
