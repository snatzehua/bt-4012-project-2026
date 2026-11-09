"""Direct score/artifact audits and fixed-decision uncertainty for calibrated F2."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score,recall_score,fbeta_score,confusion_matrix
from .common import digest,load_data,read_config,write_json
from .improvements import paths
from .recall_extension import audit_bases
from .recall_weighted_metrics import aligned
from .improvement_metrics import confusion_metrics
from .calibrated_policy_metrics import paired_draws,evaluate,fbeta
from .calibrated_policy import (CONFIG,load_selection,prepare_sources,histories,crossfit,verify_table,
    apply_rule,final_data,guard,diagnostics)

REFS=['balanced','unweighted','same_model']


def markdown(table):
    from .recall_extension_report import markdown as render
    return render(table,list(table.columns))


def independent_check(data,mask,metrics):
    y=data['labels'];tn,fp,fn,tp=confusion_matrix(y,mask,labels=[0,1]).ravel()
    for key,value in dict(tn=tn,fp=fp,fn=fn,tp=tp,precision=precision_score(y,mask,zero_division=0),
                          recall=recall_score(y,mask,zero_division=0),f2=fbeta_score(y,mask,beta=2,zero_division=0)).items():
        if not np.isclose(metrics[key],value,atol=1e-12): raise ValueError('Independent metric mismatch: '+key)


def intervals(data,masks,scope,rng,reps):
    anchor=data['selected']
    for d in data.values():aligned(anchor,d)
    draws=pd.DataFrame(paired_draws(anchor['labels'],masks,rng,reps))
    values={n:evaluate(data[n],masks[n])['f2'] for n in data};rows=[]
    diffs=pd.DataFrame({n:draws.selected-draws[n] for n in REFS})
    for ref in REFS:
        rows.append(dict(scope=scope,reference=ref,difference=values['selected']-values[ref],
                         ci_low=float(diffs[ref].quantile(.025)),ci_high=float(diffs[ref].quantile(.975)),replicates=reps))
    return pd.DataFrame(rows),diffs


def development(config,selected,cf,h):
    choices=selected['choices'];rng=np.random.default_rng(config['bootstrap_seed']);tables=[];draws=[]
    for month in [5,6]:
        data={alias:h[name][month] for alias,(name,kind) in choices.items()};masks={}
        for alias,(name,kind) in choices.items():
            row=cf[(cf.candidate==name)&(cf.policy==kind)&(cf.eval_month==month)].iloc[0]
            masks[alias]=apply_rule(config,data[alias],json.loads(row.parameters))[0]
        table,d=intervals(data,masks,str(month),rng,config['bootstrap_replicates']);tables.append(table);draws.append(d)
    macro=(draws[0]+draws[1])/2
    rows=[dict(scope='macro',reference=ref,difference=float(np.mean([t[t.reference==ref].iloc[0].difference for t in tables])),
               ci_low=float(macro[ref].quantile(.025)),ci_high=float(macro[ref].quantile(.975)),replicates=config['bootstrap_replicates']) for ref in REFS]
    return pd.concat(tables+[pd.DataFrame(rows)],ignore_index=True),pd.concat([d.assign(scope=str(m),replicate=np.arange(len(d))) for m,d in zip([5,6],draws)])


def report(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    identity=s['fingerprint'];frame=load_data(c);dev=frame[frame.month<7];bases,metas=prepare_sources(c,identity,dev,False)
    cf,bins=crossfit(c,identity,False);verify_table(cf,out/'crossfit.csv');verify_table(bins,out/'reliability.csv')
    h,_=histories(c)
    for row in cf.itertuples():
        d=h[row.candidate][row.eval_month];mask,_,_=apply_rule(c,d,json.loads(row.parameters));independent_check(d,mask,row._asdict())
    summary=pd.read_csv(out/'summary.csv');winner=summary.iloc[0]
    if (str(winner.candidate),str(winner.policy))!=(s['selected_candidate'],s['selected_policy']) or not np.isclose(winner.f2,s['selected_development_f2']):
        raise ValueError('Frozen selection disagrees with development')
    diagnostics(config_path);access=read_config(out/'exploratory_access.json');transfer=read_config(out/'transfer_complete.json')
    sha=digest(out/'selection.json')
    if access['selection_sha256']!=sha or transfer['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:
        raise ValueError('Freeze/access ordering changed')
    if transfer['table_sha256']!=digest(out/'exploratory.csv'): raise ValueError('Final table changed')
    audit_bases(c,identity,frame,'exploratory',s['fit_specs'])
    data=final_data(c,s);rows=[];masks={}
    for alias,d in data.items():
        mask,p,info=apply_rule(c,d,s['rules'][alias]);masks[alias]=mask
        metrics=evaluate(d,mask,p);independent_check(d,mask,metrics)
        rows.append(dict(role=alias,candidate=s['choices'][alias][0],policy=s['choices'][alias][1],eval_month=7,
                         parameters=json.dumps(s['rules'][alias],sort_keys=True),**{**info,'k':int(mask.sum())},**metrics))
    final=pd.DataFrame(rows);verify_table(final,out/'exploratory.csv')
    dint,ddraws=development(c,s,cf,h)
    fint,fdraws=intervals(data,masks,'7_consumed',np.random.default_rng(c['bootstrap_seed']),c['bootstrap_replicates'])
    contrasts=[]
    lookup=summary.set_index(['candidate','policy'])
    for row in summary.itertuples():
        for label,pair in [('balanced',('baseline_balanced','pooled')),('unweighted',('baseline_lgbm','pooled')),('same_model',(row.candidate,'pooled'))]:
            ref=lookup.loc[pair]
            contrasts.append(dict(candidate=row.candidate,policy=row.policy,reference=label,
                **{metric+'_difference':float(getattr(row,metric)-ref[metric]) for metric in ['f2','precision','recall','flagged_fraction']}))
    pooled=[]
    for (name,kind),group in cf.groupby(['candidate','policy']):
        counts=group[['tn','fp','fn','tp']].sum();m=confusion_metrics(*[counts[k] for k in ['tn','fp','fn','tp']])
        pooled.append(dict(candidate=name,policy=kind,**m,flagged_fraction=float((counts.tp+counts.fp)/group.n.sum())))
    for name,table in [('crossfit',cf),('summary',summary),('reliability',bins),('contrasts',pd.DataFrame(contrasts)),('pooled',pd.DataFrame(pooled)),
                       ('development_intervals',dint),('development_draws',ddraws),('consumed_intervals',fint),
                       ('consumed_draws',fdraws.assign(replicate=np.arange(len(fdraws)))),('exploratory',final)]:
        table.to_csv(dest/(name+'.csv'),index=False)
    for name in ['selection.json','validation_complete.json','diagnostic_provenance.json','hindsight.csv','exploratory_access.json','transfer_complete.json']:
        shutil.copyfile(out/name,dest/name)
    shutil.copyfile(config_path,dest/'config.json')
    cal_records=[read_config(p) for p in sorted((out/'calibration').glob('*.json'))]
    write_json(dest/'calibration_provenance.json',cal_records)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['fingerprint','source_hashes']} for r in cal_records]).to_csv(dest/'calibration_records.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name,group in summary.groupby('candidate'):
        axes[0].scatter(group.recall,group.precision,s=35,label=name.replace('stack_logit_percentile_c0p01_w1','fixed stack'))
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 21 model/policy combinations');axes[0].legend(fontsize=8)
    selected_bins=bins[(bins.candidate==s['selected_candidate'])&(bins.policy==s['selected_policy'])&(bins.eval_month==6)]
    for flavor,g in selected_bins.groupby('score_type'):axes[1].plot(g.mean_probability,g.observed_rate,'o-',label=flavor)
    axes[1].plot([0,1],[0,1],'--',color='gray',linewidth=1);axes[1].set(xlabel='Mean estimated probability',ylabel='Observed fraud rate',title='Selected policy: forward month 6 reliability')
    axes[1].legend();fig.tight_layout();fig.savefig(dest/'calibration_operating.png',dpi=160);plt.close(fig)
    prior={n:float(lookup.loc[(name,'pooled'),'f2']) for n,name in [('balanced','baseline_balanced'),('unweighted','baseline_lgbm')]}
    dev_gain={n:float(winner.f2)-v for n,v in prior.items()}
    consumed={r.reference:float(r.difference) for r in fint.itertuples()}
    # Preserve the goal unless actual F2 beats both references and the paired conditional evidence is positive.
    achieved=all(v>0 for v in dev_gain.values()) and all(v>0 for v in consumed.values() if abs(v)>1e-15) and all(
        float(fint[fint.reference==n].iloc[0].ci_low)>0 for n in ['balanced','unweighted'])
    audit=dict(fingerprint=identity,selection_sha256=sha,development_gain=dev_gain,consumed_gain=consumed,
        goal_achieved=achieved,forward_rows=len(cf),calibration_fits=sum(r['eval_month'] in [5,6] for r in cal_records),
        diagnostic_rows=len(pd.read_csv(out/'hindsight.csv')),independent_metrics_checked=True,prior_preservation_checked=True,
        uncertainty='500 paired row draws; fixed realized decisions; no recalibration/retraining/selection/time/generative/multiplicity correction',
        report_source_sha256=digest('baf/calibrated_policy_report.py'))
    write_json(dest/'audit.json',audit);guard(c)
    best_cal=summary[summary.policy.isin(c['calibration_policies'])].iloc[0]
    text=f'''# Calibrated F2 policy findings

The finite policy study is complete. Selected **{s['selected_candidate']} / {s['selected_policy']}** has actual macro development F2 **{winner.f2:.6f}**, versus balanced **{prior['balanced']:.6f}** and unweighted **{prior['unweighted']:.6f}**. The best calibrated policy is **{best_cal.candidate} / {best_cal.policy}**, F2 **{best_cal.f2:.6f}**. Performance goal achieved by this study: **{achieved}**. No 1% quota or precision floor is imposed; beta=2 remains analyst-selected.

## Forward operating comparisons

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

All 42 rows on identical month-5/6 cases are in crossfit.csv. Four calibration policies use unweighted sigmoid-logit or isotonic calibration fitted on pooled/latest earlier predictions, alongside three empirical policies. Month 5 calibration uses month 4; month 6 uses 4/5 or 5. Current labels determine neither calibration nor batch flagged count. Scores and all five prior studies retain their frozen recipes. Source reuse is adaptive reuse of development evidence.

The batch surrogate S(k)=5*sum(top-k p)/(4*sum(p)+k) is a ratio of estimated counts, not exact expected F2 or achieved F2. Batch inference sorts calibrated probability, then raw score, then stable ID; the selected k may vary by unlabeled batch. Calibration does not guarantee transfer under temporal drift. reliability.csv and crossfit.csv include ten equal-frequency bins, raw/calibrated Brier/log-loss, estimated positives/TP/surrogate and actual confusion counts. Calibrated policy ranking and actual performance are separate evidence.

## Paired development differences

{markdown(dint)}

## Consumed month 7: exploratory transfer

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

{markdown(fint)}

Selection and dependencies were frozen before further access. This period was previously consumed and is not independent confirmation. Only the selected policy, both old raw pooled baselines and its same-model pooled comparator are evaluated; no losing calibrated policies or month-7 hindsight optimum are swept. A development gain that fails this transfer is not out-of-time improvement.

## Diagnostic limits and reproduction

Six hindsight.csv rows were generated only after selection freeze. They are label-aware optimistic optima for the fixed deterministic rankings on development months, excluded from selection and calibration. They are not deployable gains or general model ceilings.

Intervals use 500 paired fixed-decision row-bootstrap draws, seed 1027, with within-month draws averaged for development macro F2. Batch choices remain fixed in each draw. Intervals exclude recalibration, retraining, selection, between-month/generative uncertainty and multiplicity correction. An interval overlapping zero does not support a clear-gain claim. All failed policies and contrasts remain visible; AP, Brier or surrogate improvement alone does not satisfy the user's objective.

See config.json, calibration_records.csv, pooled.csv, contrasts.csv, validation_complete.json, transfer_complete.json and audit.json for identities, counts and checked evidence. README documents dependencies, Base access, five-record restoration and missing-cache refits in new destinations. Codex assisted with study design, code, execution, tests, figures and interpretation; no external notebook was copied. These outputs support the eventual course report.

![Calibration and operating comparisons](calibration_operating.png)
'''
    (dest/'findings.md').write_text(text)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=CONFIG);a=p.parse_args();report(a.config)
