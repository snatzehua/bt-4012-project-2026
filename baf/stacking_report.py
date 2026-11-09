"""Direct prediction audits and exploratory F2 evidence for chronological stacking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import digest,load_data,read_config,write_json
from .improvements import paths,prediction
from .recall_extension_report import markdown
from .recall_weighted_metrics import aligned,decisions,fbeta,measure
from .stacking import (audit_basegrid,audit_metas,base_history,crossfit,final_predictions,
                      fit_meta,guard,histories,load_selection,order_summary,verify_tables)
from .stacking_features import matched_control


def paired_draws(data,rules,refs,rng,replicates):
    anchor=data['selected'];y=anchor['labels'].astype(np.int64)
    masks={};estimates={}
    for name,d in data.items():
        aligned(anchor,d);masks[name]=decisions(d,rules[name]).astype(np.int64)
        estimates[name]=measure(d,rules[name])['f2']
    draws=[]
    for _ in range(replicates):
        weights=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y))
        positives=np.dot(weights,y);values={}
        for name,pred in masks.items():
            tp=np.dot(weights,y*pred);fp=np.dot(weights,(1-y)*pred)
            values[name]=float(fbeta(tp,fp,positives-tp))
        draws.append([values['selected']-values[ref] for ref in refs])
    draws=np.array(draws)
    table=pd.DataFrame([dict(reference=ref,difference=estimates['selected']-estimates[ref],
                             ci_low=float(np.quantile(draws[:,i],.025)),ci_high=float(np.quantile(draws[:,i],.975)))
                        for i,ref in enumerate(refs)])
    return table,draws


def references(config,selected):
    mapping=dict(balanced=('baseline_balanced','pooled'),unweighted=('baseline_lgbm','pooled'))
    spec=next((s for s in config['combiners'] if s['id']==selected['selected_candidate']),None)
    if spec:
        ctrl=matched_control(spec,config['combiners'])
        if ctrl:mapping['matched_control']=(ctrl['id'],selected['selected_policy'])
    return mapping


def development_intervals(config,selected,cf,history):
    mapping=references(config,selected);rng=np.random.default_rng(config['bootstrap_seed']);tables=[];draws=[]
    for month in config['validation_months'][1:]:
        choices={'selected':(selected['selected_candidate'],selected['selected_policy']),**mapping};data={};rules={}
        for key,(name,kind) in choices.items():
            data[key]=history[name][month]
            row=cf[(cf.candidate==name)&(cf.policy==kind)&(cf.eval_month==month)].iloc[0]
            rules[key]=json.loads(row.parameters)
        table,d=paired_draws(data,rules,list(mapping),rng,config['bootstrap_replicates'])
        table['eval_month']=str(month);tables.append(table);draws.append(d)
    macro=np.mean(draws,axis=0);rows=[]
    for i,name in enumerate(mapping):
        rows.append(dict(eval_month='macro',reference=name,difference=float(np.mean([t[t.reference==name].iloc[0].difference for t in tables])),
                         ci_low=float(np.quantile(macro[:,i],.025)),ci_high=float(np.quantile(macro[:,i],.975))))
    return pd.concat(tables+[pd.DataFrame(rows)],ignore_index=True)


def contrasts(config,summary):
    lookup=summary.set_index(['candidate','policy']);rows=[]
    def add(label,a,b):
        for kind in config['policies']:
            x,y=lookup.loc[(a,kind)],lookup.loc[(b,kind)]
            rows.append(dict(contrast=label,candidate=a,reference=b,policy=kind,
                             **{m+'_difference':float(x[m]-y[m]) for m in ['f2','precision','recall','flagged_fraction']}))
    specs=config['combiners']
    for s in specs:
        ctrl=matched_control(s,specs)
        if ctrl:add('stack minus matched single-member control',s['id'],ctrl['id'])
        for field,base_value,label in [('representation','logit','percentile feature'),('pos_weight',1,'positive weight'),('C',.001,'regularization')]:
            if s[field]==base_value:continue
            match=[t for t in specs if t[field]==base_value and all(t[k]==s[k] for k in ['group','representation','pos_weight','C'] if k!=field)]
            if len(match)!=1:raise ValueError('Missing matched factor control')
            add(label,s['id'],match[0]['id'])
    policy=[]
    for name in summary.candidate.unique():
        for kind in ['latest','rank_fraction']:
            x,y=lookup.loc[(name,kind)],lookup.loc[(name,'pooled')]
            policy.append(dict(candidate=name,policy=kind,**{m+'_difference':float(x[m]-y[m]) for m in ['f2','precision','recall','flagged_fraction']}))
    return pd.DataFrame(rows),pd.DataFrame(policy)


def report(config_path):
    config,selected=load_selection(config_path);out=Path(config['output_dir']);dest=Path(config['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    frame=load_data(config);identity=selected['fingerprint']
    base_records=audit_basegrid(config,identity,frame)
    if len(base_records)!=config['expected_base_results']:raise ValueError('Base grid incomplete')
    history,records=base_history(config);meta_records=audit_metas(config,identity,history,records)
    for spec in selected['learned_specs']:
        rebuilt=fit_meta(config,spec,7,identity,history,records,'frozen')
        if rebuilt!=selected['final_fits'][spec['id']]:raise ValueError('Final fit changed after freeze')
    final_base=audit_basegrid(config,identity,frame,'exploratory',selected['fit_specs'])
    cf=crossfit(config);verify_tables(config,cf);summary=order_summary(cf);top=summary.iloc[0]
    if (top.candidate,top.policy)!=(selected['selected_candidate'],selected['selected_policy']) or not np.isclose(top.f2,selected['selected_development_f2'],atol=1e-12):
        raise ValueError('Selection is not deterministic development winner')
    final_data=final_predictions(config,selected);final=pd.read_csv(out/'exploratory.csv')
    expected={(n,k) for n in selected['evaluate_names'] for k in config['policies']}
    if len(final)!=len(expected) or {(r.candidate,r.policy) for r in final.itertuples()}!=expected:raise ValueError('Final policy grid differs from freeze')
    for row in final.itertuples():
        rule=selected['rules'][row.candidate][row.policy]
        if json.loads(row.parameters)!=rule:raise ValueError('Final rule differs from freeze')
        d=final_data[row.candidate];saved=prediction(config,'exploratory',7,row.candidate);aligned(d,saved)
        if not np.allclose(d['scores'],saved['scores'],atol=1e-12,rtol=1e-12):raise ValueError('Final scores differ from frozen model')
        for key,value in measure(d,rule).items():
            if not np.isclose(value,getattr(row,key),atol=1e-12,rtol=1e-10):raise ValueError('Reported final metric differs: '+key)
    access=read_config(out/'exploratory_access.json')
    if access['selection_sha256']!=digest(out/'selection.json') or not selected['created_at_utc']<access['access_at_utc']:
        raise ValueError('Freeze/access ordering or selection hash changed')
    h,_=histories(config);dev_ci=development_intervals(config,selected,cf,h)
    mapping=references(config,selected);choices={'selected':(selected['selected_candidate'],selected['selected_policy']),**mapping}
    data={key:final_data[name] for key,(name,kind) in choices.items()}
    rules={key:selected['rules'][name][kind] for key,(name,kind) in choices.items()}
    final_ci,_=paired_draws(data,rules,list(mapping),np.random.default_rng(config['bootstrap_seed']),config['bootstrap_replicates'])
    matched,policy=contrasts(config,summary)
    pooled=cf.groupby(['candidate','policy'])[['tp','fp','fn']].sum().reset_index();pooled['pooled_f2']=fbeta(pooled.tp,pooled.fp,pooled.fn)
    for name,t in [('crossfit',cf),('summary',summary),('exploratory',final),('matched_contrasts',matched),('policy_contrasts',policy),('pooled_development',pooled),('development_intervals',dev_ci),('paired_intervals',final_ci)]:
        t.to_csv(dest/(name+'.csv'),index=False)
    write_json(dest/'config.json',config);write_json(dest/'selection.json',selected)
    name=selected['selected_candidate'];kind=selected['selected_policy'];winner=final[(final.candidate==name)&(final.policy==kind)].iloc[0]
    baselines={key:final[(final.candidate==ref)&(final.policy==p)].iloc[0] for key,(ref,p) in mapping.items()}
    dev_ref=summary[(summary.candidate=='baseline_balanced')&(summary.policy=='pooled')].iloc[0]
    dev_gain=float(top.f2-dev_ref.f2);gains={key:float(winner.f2-r.f2) for key,r in baselines.items()}
    audit=dict(base_development_results=len(base_records),new_base_development_fits=sum(not r['reused'] for r in base_records),
               reused_base_results=sum(r['reused'] for r in base_records),combiner_control_development_fits=len(meta_records),
               strategy_count=int(summary.candidate.nunique()),crossfit_rows=len(cf),selected_candidate=name,selected_policy=kind,
               development_f2_gain=dev_gain,consumed_month7_f2_gains=gains,
               performance_goal_achieved=bool(dev_gain>0 and all(v>0 for k,v in gains.items() if k!='matched_control')),
               final_combiner_fits=len(selected['learned_specs']),new_exploratory_base_fits=sum(not r['reused'] for r in final_base),
               reused_exploratory_base_results=sum(r['reused'] for r in final_base),
               saved_predictions_verified_and_metrics_recomputed=True,earlier_only_base_combiner_and_policy_sources_pass=True,
               training_only_scalers_and_feature_order_pass=True,identical_future_cases=True,prior_four_studies_preserved=True,
               source_config_software_identity_pass=True,selection_unchanged_after_access=True,final_combiners_fitted_before_access=True,
               freeze_precedes_access=True,month7_previously_consumed=True,config_sha256=digest(config_path),selection_sha256=digest(out/'selection.json'),
               report_source_sha256=digest('baf/stacking_report.py'),final_prediction_sha256={n:digest(paths(config,'exploratory',7,n)[1]) for n in selected['evaluate_names']})
    guard(config);write_json(dest/'audit.json',audit)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    plot_choices=[('Selected',name,kind,winner),('Balanced baseline','baseline_balanced','pooled',baselines['balanced']),('Unweighted baseline','baseline_lgbm','pooled',baselines['unweighted'])]
    for label,n,k,end in plot_choices:
        values=list(cf[(cf.candidate==n)&(cf.policy==k)].sort_values('eval_month').f2)+[end.f2]
        axes[0].plot([5,6,7],values,marker='o',label=label)
    axes[0].set(xticks=[5,6,7],xlabel='Evaluation month (7 already consumed)',ylabel='F2',title='Chronological detection');axes[0].legend(fontsize=9)
    pos=np.arange(3)
    for i,(label,n,k,row) in enumerate(plot_choices):
        axes[1].bar(pos+(i-1)*.24,100*np.array([row.precision,row.recall,row.flagged_fraction]),.24,label=label)
    axes[1].set(xticks=pos,xticklabels=['Precision','Recall','Flagged'],ylabel='Percent',title='Consumed month 7: workload and detection');axes[1].legend(fontsize=9)
    fig.tight_layout();fig.savefig(dest/'stacking_f2.png',dpi=170);plt.close(fig)
    stronger_gain=min(gains['balanced'],gains['unweighted'])
    message='The exploratory point gain exceeds both previous baseline policies.' if stronger_gain>0 else 'The frozen selection does not improve on both consumed-period baseline policies.'
    verdict='The measured point gain is limited to this adaptive exploratory study.' if audit['performance_goal_achieved'] else 'The requested performance improvement remains unachieved.'
    findings=f'''# Chronological score stacking for recall-weighted detection

This adaptive study optimizes positive-class **F2=5TP/(5TP+4FN+FP)** (beta=2), with no imposed review quota or precision floor. It is distinct from support-weighted F1. The finite settings are analyst-selected, not group-approved. Four earlier studies and their negative results remain frozen.

## Measured result

Development selected **{name} / {kind}** at macro month-5/6 F2 **{top.f2:.4f}**, versus balanced LightGBM **{dev_ref.f2:.4f}** (difference **{dev_gain:+.4f}**). On consumed month 7, selected F2 is **{winner.f2:.4f}**, versus balanced **{baselines['balanced'].f2:.4f}** and unweighted **{baselines['unweighted'].f2:.4f}**. {message} {verdict}

{markdown(final_ci,['reference','difference','ci_low','ci_high'])}

These are conditional paired intervals at fixed models and decisions. Intervals overlapping zero preclude clear-gain claims. Repeated development tuning and already-consumed month 7 prevent independent confirmation, even when an interval excludes zero.

## Precision, recall and workload

{markdown(final[(final.policy=='pooled')|final.selected|((final.policy==kind)&final.candidate.isin([x[0] for x in mapping.values()]))],['candidate','policy','f2','f1','f3','precision','recall','flagged_fraction','tp','fp','fn'])}

The selected policy catches **{int(winner.tp):,}** frauds, misses **{int(winner.fn):,}**, flags **{int(winner.fp):,}** legitimate applications and reviews **{winner.flagged_fraction:.2%}** of applications. Other policies are diagnostics, not replacements selected after seeing month 7. A better F2 need not make these operating costs acceptable.

![Chronological F2 and operating trade-off](stacking_f2.png)

## Search, chronology and controls

The {len(base_records)} base development results comprise {audit['new_base_development_fits']} new fits and {audit['reused_base_results']} verified reuses. Four fixed members use unchanged recipes: balanced LightGBM, native LightGBM, engineered CatBoost and engineered logistic regression. Their 1/2/3 predictions train only on earlier applicant months; 4/5/6 predictions are verified against their frozen sources. Missing caches refit into new directories.

The 24 learned presets cross four-member/single-balanced-member inputs, logit/logit-plus-percentile representations, C 0.001/0.01/0.1 and positive weights 1/5. They generate {len(meta_records)} combiner/control fits. Month-4 combiners train on out-of-time base predictions 1/2/3, month-5 combiners on 1-4 and month-6 combiners on 1-5. Threshold evaluation on month 5 uses out-of-time combiner predictions from 4; month 6 uses 4/5. No scored month's labels enter base fitting, combiner/scaler fitting or operating-rule selection.

Percentile features use only the unlabeled prediction-month score batch, with average ties. This representation requires batch inference and is not a calibrated probability claim. Scalers use only earlier training rows; fixed member/column order and clip bounds are recorded. Single-member controls isolate transformation/decision effects from combination effects.

{markdown(summary.head(12),['candidate','policy','f2','precision','recall','flagged_fraction'])}

There are {audit['strategy_count']} strategies and {len(cf)} forward operating comparisons. Both old macro F2 baselines reproduce. Macro monthly F2 determines selection; [pooled_development.csv](pooled_development.csv) reports pooled confusion-count F2 separately.

## Contrasts and uncertainty

[Matched contrasts](matched_contrasts.csv) compare stack versus its single-member control, percentile inputs, positive weighting and regularization under the same policy. [Policy contrasts](policy_contrasts.csv) hold predictions fixed. Negative results remain visible; a gain due to a single-model control is not attributed to stacking. The selected strategy is compared to its matched control when it is a four-member stack.

{markdown(dev_ci,['eval_month','reference','difference','ci_low','ci_high'])}

Intervals use 500 paired row-bootstrap draws, seed 1027, with within-month resampling for development macro differences. They condition on realized decisions, including batch/rank decisions, and exclude retraining, selection, between-month/generative uncertainty and multiplicity adjustment. A point gain supports only its named exploratory scope.

## Freeze, verification and reproduction

Final fitting used {len(selected['learned_specs'])} frozen selected/control combiners trained on out-of-time base predictions 1-6 before further month-7 access. Transfer needed {audit['new_exploratory_base_fits']} new base fits and {audit['reused_exploratory_base_results']} verified reuses. Only frozen dependencies and both original baselines were scored. Source/config/software/data, training hashes/class counts, future IDs/labels, score/model hashes, scaler means/column order, earlier-only policy inputs, deterministic selection, freeze-before-access and unchanged selection were audited. Reported metrics were rebuilt from saved predictions and frozen rules.

Use the [README](../../README.md) validate/freeze/exploratory/report sequence, [protocol](../../docs/chronological-stacking-protocol.md), [exact config](config.json), [frozen selection](selection.json) and [audit](audit.json). Small exported freeze records enable fresh-clone prerequisites; ignored caches can be refitted into this study's destination without writing old studies. New tuning requires another study.

One synthetic variant, eight coarse months, only two operating-policy evaluation months, unknown label delays and adaptive reuse limit generalization. Month 7 cannot become a fresh holdout again. Independent confirmation requires new data.

Codex assisted with design, implementation, execution, checks, figures and interpretation. Acknowledge BAF, scikit-learn, LightGBM and CatBoost in the eventual course report. This is a supporting experiment export, not the final 8-12 page submission.
'''
    (dest/'findings.md').write_text(findings);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/chronological-stacking.json');report(p.parse_args().config)
