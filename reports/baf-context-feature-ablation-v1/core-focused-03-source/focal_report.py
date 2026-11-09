"""Reconstruct focal-study predictions, actual metrics and conditional uncertainty."""
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
from .common import digest,features,load_data,read_config,write_json
from .recall_weighted_metrics import aligned,decisions
from .calibrated_policy_metrics import evaluate,paired_draws
from .calibrated_policy_report import independent_check,markdown
from .focal_study import (CONFIG,load_selection,guard,history,crossfit,audit_grid,audit_loss,prepare_references,
    final_data,verify_table,model_paths,all_specs,export_comparisons)


def intervals(data,masks,rng,replicates,scope):
    anchor=data['selected'];refs=[n for n in data if n!='selected']
    for d in data.values():aligned(d,anchor)
    values={n:evaluate(data[n],masks[n])['f2'] for n in data}
    full=pd.DataFrame(paired_draws(anchor['labels'],masks,rng,replicates))
    diffs=pd.DataFrame({n:full.selected-full[n] for n in refs})
    table=pd.DataFrame([dict(scope=scope,reference=n,difference=values['selected']-values[n],
        ci_low=float(diffs[n].quantile(.025)),ci_high=float(diffs[n].quantile(.975)),replicates=replicates) for n in refs])
    return table,diffs


def development(config,selected,cf,h):
    rng=np.random.default_rng(config['bootstrap_seed']);tables=[];draws=[];refs=[n for n in selected['choices'] if n!='selected']
    for month in [5,6]:
        data={alias:h[name][month] for alias,(name,kind) in selected['choices'].items()};masks={}
        for alias,(name,kind) in selected['choices'].items():
            row=cf[(cf.candidate==name)&(cf.policy==kind)&(cf.eval_month==month)].iloc[0]
            masks[alias]=decisions(data[alias],json.loads(row.parameters))
        table,d=intervals(data,masks,rng,config['bootstrap_replicates'],str(month));tables.append(table);draws.append(d)
    macro=(draws[0]+draws[1])/2
    table=pd.DataFrame([dict(scope='macro',reference=n,difference=float(np.mean([t[t.reference==n].iloc[0].difference for t in tables])),
        ci_low=float(macro[n].quantile(.025)),ci_high=float(macro[n].quantile(.975)),replicates=config['bootstrap_replicates']) for n in refs])
    return pd.concat(tables+[table],ignore_index=True),pd.concat([d.assign(scope=str(m),replicate=np.arange(len(d))) for m,d in zip([5,6],draws)],ignore_index=True)


def report(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    frame=load_data(c);base=features(frame);dev=frame[frame.month<7];devbase=base.loc[dev.index]
    records=audit_grid(c,s['fingerprint'],dev,devbase)
    cf=crossfit(c);verify_table(cf,out/'crossfit.csv');h=history(c)
    for row in cf.itertuples():
        d=h[row.candidate][row.eval_month];mask=decisions(d,json.loads(row.parameters));independent_check(d,mask,row._asdict())
    summary=pd.DataFrame(cf.groupby(['candidate','policy'],as_index=False)[['f2','f1','f3','precision','recall','flagged_fraction','average_precision','tp','fp','fn']].mean())
    summary=summary.sort_values(['f2','recall','precision','candidate','policy'],ascending=[False,False,False,True,True]).reset_index(drop=True)
    verify_table(summary,out/'summary.csv');winner=summary.iloc[0]
    if (str(winner.candidate),str(winner.policy))!=(s['selected_candidate'],s['selected_policy']):raise ValueError('Selection differs from actual F2 order')
    access=read_config(out/'exploratory_access.json');transfer=read_config(out/'transfer_complete.json');sha=digest(out/'selection.json')
    if access['selection_sha256']!=sha or transfer['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:
        raise ValueError('Freeze/access mismatch')
    if transfer['table_sha256']!=digest(out/'exploratory.csv'):raise ValueError('Final table changed')
    reference_ids={r['id'] for r in c['references']};refs=[r for r in s['fit_specs'] if r['id'] in reference_ids]
    if refs:prepare_references(c,s['fingerprint'],frame,base,'exploratory',refs,False)
    for spec in s['fit_specs']:
        if spec['id'] not in reference_ids:audit_loss(c,s['fingerprint'],frame,base,spec,7,'exploratory')
    data=final_data(c,s);rows=[];masks={}
    for alias,d in data.items():
        rule=s['rules'][alias];mask=decisions(d,rule);masks[alias]=mask;metrics=evaluate(d,mask)
        independent_check(d,mask,metrics)
        rows.append(dict(role=alias,candidate=s['choices'][alias][0],policy=s['choices'][alias][1],eval_month=7,
                         parameters=json.dumps(rule,sort_keys=True),**metrics))
    final=pd.DataFrame(rows);verify_table(final,out/'exploratory.csv')
    dint,ddraws=development(c,s,cf,h)
    fint,fdraws=intervals(data,masks,np.random.default_rng(c['bootstrap_seed']),c['bootstrap_replicates'],'7_consumed')
    for label,table in [('crossfit',cf),('summary',summary),('exploratory',final),('development_intervals',dint),
        ('development_draws',ddraws),('consumed_intervals',fint),('consumed_draws',fdraws.assign(replicate=np.arange(len(fdraws))))]:
        table.to_csv(dest/(label+'.csv'),index=False)
    for name in ['selection.json','validation_complete.json','integration_gate.json','exploratory_access.json','transfer_complete.json',
                 'contrasts.csv','pooled.csv','ordinary_score_diagnostics.csv']:
        shutil.copyfile(out/name,dest/name)
    shutil.copyfile(config_path,dest/'config.json')
    write_json(dest/'run_provenance.json',records)
    numeric=[]
    for spec in c['presets']:
        for month in [4,5,6]:
            trace=read_config(model_paths(c,'validation',month,spec['id'])[1])
            numeric.extend(dict(candidate=spec['id'],eval_month=month,**r) for r in trace)
    num=pd.DataFrame(numeric);num.to_csv(dest/'numeric_traces.csv',index=False)
    numeric_summary=num.groupby(['candidate','eval_month'],as_index=False).agg(
        iterations=('iteration','max'),mean_replaced_fraction=('replaced_fraction','mean'),
        max_replaced_fraction=('replaced_fraction','max'),minimum_true_hessian=('true_hessian_min','min'),
        maximum_gradient=('gradient_abs_max','max'),final_training_loss=('mean_loss','last'))
    numeric_summary.to_csv(dest/'numeric_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for label,names in [('focal',[x['id'] for x in c['presets'] if x['gamma']>0]),
        ('ordinary controls',[x['id'] for x in c['presets'] if x['gamma']==0]),('old references',list(reference_ids))]:
        g=summary[summary.candidate.isin(names)];axes[0].scatter(g.recall,g.precision,s=32,label=label)
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 33 model/policy comparisons');axes[0].legend(fontsize=8)
    for name,g in num[num.eval_month==6].groupby('candidate'):
        if (g.replaced_fraction>0).any():axes[1].plot(g.iteration,g.replaced_fraction,label=name)
    axes[1].set(xlabel='Boosting iteration',ylabel='Fraction with replaced curvature',title='Focal numerical safeguard: month 6');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'loss_operating.png',dpi=160);plt.close(fig)
    lookup=summary.set_index(['candidate','policy'])
    prior={alias:float(lookup.loc[(name,'pooled'),'f2']) for alias,name in [('balanced','baseline_balanced'),('unweighted','baseline_lgbm')]}
    dg={n:float(winner.f2)-v for n,v in prior.items()};fg={r.reference:float(r.difference) for r in fint.itertuples()}
    clear_dev=all(float(dint[(dint.scope=='macro')&(dint.reference==n)].iloc[0].ci_low)>0 for n in ['balanced','unweighted'])
    clear_final=all(float(fint[fint.reference==n].iloc[0].ci_low)>0 for n in ['balanced','unweighted'])
    achieved=all(v>0 for v in dg.values()) and all(fg[n]>0 for n in ['balanced','unweighted']) and clear_dev and clear_final
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,goal_achieved=achieved,development_gain=dg,consumed_gain=fg,
        conditional_clear_development_gain=clear_dev,conditional_clear_consumed_gain=clear_final,
        independent_metrics_checked=True,prior_preservation_checked=True,prediction_results=len(records),forward_rows=len(cf),
        numeric_trace_rows=len(num),report_source_sha256=digest('baf/focal_report.py'),
        uncertainty='500 paired fixed-decision row draws; no model/selection/retraining/time/generative/multiplicity correction')
    write_json(dest/'audit.json',audit);guard(c)
    best_focal=summary[summary.candidate.isin([x['id'] for x in c['presets'] if x['gamma']>0])].iloc[0]
    contrasts=pd.read_csv(out/'contrasts.csv');loss_contrasts=contrasts[contrasts.contrast.isin(['focal minus custom logistic','custom minus built-in logistic'])]
    text=f'''# Focal-loss study findings

Selected **{s['selected_candidate']} / {s['selected_policy']}** has actual macro development F2 **{winner.f2:.6f}**, versus balanced **{prior['balanced']:.6f}** and unweighted **{prior['unweighted']:.6f}**. Best focal policy: **{best_focal.candidate} / {best_focal.policy}**, F2 **{best_focal.f2:.6f}**. Goal achieved by this study: **{achieved}**. Completing this search alone is not achieving the performance objective.

## Design and numerical checks

Eight matched new presets make 24 development fits; three frozen references supply nine prediction results. Four miniature integration fits are separate checks. The gamma=0 custom/built-in probability gate passes at the declared 1e-6 tolerance. Native feature handling, tree settings, positive weights and zero initialization are matched across new loss presets. Every model uses earlier training months; policy evaluation on months 5/6 uses earlier out-of-time predictions only. The user objective remains positive-class F2 beta=2, with no review quota or precision floor. All settings are analyst-selected, not approved bank cost ratios.

The custom objective consumes raw margins. Saved boosters predict raw margins and apply expit exactly once; all predictions and actual confusion/F2 metrics reconstruct. Focal Hessians below 1e-6 are replaced while gradients remain unchanged: this is approximate positive-curvature boosting, not exact Newton optimization. Gamma=0 uses its actual logistic curvature. numeric_traces.csv/numeric_summary.csv retain safeguard use and training losses. A changed loss also changes gradient/Hessian scale and regularization interaction; comparisons cannot attribute effects solely to hard-example emphasis. Focal scores are not claimed calibrated; Brier/log-loss are descriptive. Ordinary score discrepancies from full fits are exported rather than assuming bitwise equivalence after hundreds of rounds.

## Forward actual performance

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

All 66 forward rows, 33 pooled summaries and all unsuccessful contrasts remain visible. AP and training loss do not replace F2. The three empirical policies retain their original ties and fraction rounding. Development evidence has been adaptively reused across six previous studies.

## Matched loss comparisons

{markdown(loss_contrasts[['contrast','candidate','policy','reference','f2_difference','recall_difference','flagged_fraction_difference']])}

## Paired conditional development differences

{markdown(dint)}

## Already-consumed month 7: exploratory transfer

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

{markdown(fint)}

Freeze and access hashes agree, with selection before further access. Only selected/required matched controls, same-model pooled policy and both old pooled baselines are evaluated. Losing focal settings are not swept; no month-7 hindsight optimum or threshold retuning is performed. A development gain failing this transfer is not out-of-time improvement. This consumed period cannot supply independent confirmation.

## Uncertainty, preservation and reproduction

Intervals use 500 paired fixed-decision row draws, seed 1027; development resamples within each month and averages differences for macro F2. Models and decisions remain fixed, with no reselection or refitting. Intervals exclude model-selection, retraining, temporal/generative uncertainty and multiplicity correction. An interval overlapping zero does not establish a clear gain. Six earlier frozen studies remain unchanged. The exact config, integration gate, run provenance, selection/access records and audit accompany these findings; README describes pinned dependencies, Base access, six-record restoration and new-destination missing-cache reproduction. Codex assisted with design, code, execution, numerical/methodological verification, plots and interpretation; no public notebook was copied.

![Operating and numerical diagnostics](loss_operating.png)
'''
    (dest/'findings.md').write_text(text);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=CONFIG);a=p.parse_args();report(a.config)
