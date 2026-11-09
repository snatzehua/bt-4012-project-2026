"""Audit raw-score ranking decisions and paired chronological F2 differences."""
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
from .recall_weighted_metrics import aligned,decisions,measure
from .calibrated_policy_metrics import paired_draws
from .calibrated_policy_report import independent_check,markdown
from .improvement_metrics import confusion_metrics
from .direct_ranking import (CONFIG,load_selection,guard,history,crossfit,audit_grid,audit_model,prepare_references,
    final_data,verify_table,model_paths,order_summary)


def actual_f2(data,mask):
    y=data['labels'].astype(bool);p=np.asarray(mask,dtype=bool)
    if p.shape!=y.shape:raise ValueError('Fixed decision population differs')
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());den=5*tp+4*fn+fp
    return 5*tp/den if den else 0.


def intervals(data,masks,rng,replicates,scope):
    anchor=data['selected'];refs=[n for n in data if n!='selected']
    for d in data.values():aligned(d,anchor)
    values={n:actual_f2(data[n],masks[n]) for n in data}
    full=pd.DataFrame(paired_draws(anchor['labels'],masks,rng,replicates));diffs=pd.DataFrame({n:full.selected-full[n] for n in refs})
    table=pd.DataFrame([dict(scope=scope,reference=n,difference=values['selected']-values[n],
        ci_low=float(diffs[n].quantile(.025)),ci_high=float(diffs[n].quantile(.975)),replicates=replicates) for n in refs])
    return table,diffs


def development(c,s,cf,h):
    rng=np.random.default_rng(c['bootstrap_seed']);tables=[];draws=[];refs=[n for n in s['choices'] if n!='selected']
    for month in [5,6]:
        data={alias:h[name][month] for alias,(name,kind) in s['choices'].items()};masks={}
        for alias,(name,kind) in s['choices'].items():
            row=cf[(cf.candidate==name)&(cf.policy==kind)&(cf.eval_month==month)].iloc[0]
            masks[alias]=decisions(data[alias],json.loads(row.parameters))
        table,d=intervals(data,masks,rng,c['bootstrap_replicates'],str(month));tables.append(table);draws.append(d)
    macro=(draws[0]+draws[1])/2
    table=pd.DataFrame([dict(scope='macro',reference=n,difference=float(np.mean([t[t.reference==n].iloc[0].difference for t in tables])),
        ci_low=float(macro[n].quantile(.025)),ci_high=float(macro[n].quantile(.975)),replicates=c['bootstrap_replicates']) for n in refs])
    return pd.concat(tables+[table],ignore_index=True),pd.concat([d.assign(scope=str(m),replicate=np.arange(len(d))) for m,d in zip([5,6],draws)],ignore_index=True)


def report(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    frame=load_data(c);base=features(frame);dev=frame[frame.month<7]
    records=audit_grid(c,s['fingerprint'],dev,base.loc[dev.index]);cf=crossfit(c);verify_table(cf,out/'crossfit.csv');h=history(c)
    for row in cf.itertuples():
        d=h[row.candidate][row.eval_month];independent_check(d,decisions(d,json.loads(row.parameters)),row._asdict())
    summary=order_summary(cf).reset_index(drop=True);verify_table(summary,out/'summary.csv');winner=summary.iloc[0]
    if (str(winner.candidate),str(winner.policy))!=(s['selected_candidate'],s['selected_policy']):raise ValueError('Selection order differs')
    access=read_config(out/'exploratory_access.json');transfer=read_config(out/'transfer_complete.json');sha=digest(out/'selection.json')
    if access['selection_sha256']!=sha or transfer['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Freeze/access mismatch')
    if transfer['table_sha256']!=digest(out/'exploratory.csv'):raise ValueError('Final table changed')
    refids={x['id'] for x in c['references']};refs=[x for x in s['fit_specs'] if x['id'] in refids];final_records=[]
    if refs:final_records+=prepare_references(c,s['fingerprint'],frame,base,'exploratory',refs,False)
    for spec in s['fit_specs']:
        if spec['id'] not in refids:final_records.append(audit_model(c,s['fingerprint'],frame,base,spec,7,'exploratory'))
    data=final_data(c,s);rows=[];masks={}
    for alias,d in data.items():
        rule=s['rules'][alias];mask=decisions(d,rule);masks[alias]=mask;metrics=measure(d,rule);independent_check(d,mask,metrics)
        rows.append(dict(role=alias,candidate=s['choices'][alias][0],policy=s['choices'][alias][1],eval_month=7,parameters=json.dumps(rule,sort_keys=True),**metrics))
    final=pd.DataFrame(rows);verify_table(final,out/'exploratory.csv');dint,ddraws=development(c,s,cf,h)
    fint,fdraws=intervals(data,masks,np.random.default_rng(c['bootstrap_seed']),c['bootstrap_replicates'],'7_consumed')
    for name,table in [('crossfit',cf),('summary',summary),('exploratory',final),('development_intervals',dint),('development_draws',ddraws),
        ('consumed_intervals',fint),('consumed_draws',fdraws.assign(replicate=np.arange(len(fdraws))))]:table.to_csv(dest/(name+'.csv'),index=False)
    for name in ['selection.json','validation_complete.json','integration_gate.json','exploratory_access.json','transfer_complete.json',
        'contrasts.csv','pooled.csv','score_diagnostics.csv']:shutil.copyfile(out/name,dest/name)
    shutil.copyfile(config_path,dest/'config.json');write_json(dest/'run_provenance.json',records);write_json(dest/'transfer_provenance.json',final_records)
    group_rows=[];group_summary=[]
    for record in records+final_records:
        if record['candidate'] in refids:continue
        meta=read_config(model_paths(c,record['phase'],record['eval_month'],record['candidate'])[1])
        shared=dict(candidate=record['candidate'],phase=record['phase'],eval_month=record['eval_month'],used_by_objective=meta['used_by_objective'])
        group_rows.extend({**shared,**r} for r in meta['query_rows'])
        group_summary.append({**shared,**{k:v for k,v in meta.items() if k not in ['groups','query_rows']},
            'fit_seconds':record['fit_seconds'],'fitted_tree_count':record['fitted_tree_count']})
    gs=pd.DataFrame(group_summary);pd.DataFrame(group_rows).to_csv(dest/'queries.csv',index=False);gs.to_csv(dest/'group_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for label,names in [('ranking',[x['id'] for x in c['presets'] if x['kind']=='rank']),('binary controls',[x['id'] for x in c['presets'] if x['kind']=='binary']),('old references',list(refids))]:
        g=summary[summary.candidate.isin(names)];axes[0].scatter(g.recall,g.precision,s=32,label=label)
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 27 model/policy comparisons');axes[0].legend(fontsize=8)
    for name,g in gs[(gs.phase=='validation')&gs.used_by_objective].groupby('candidate'):
        axes[1].plot(g.eval_month,g.zero_positive_fraction,marker='o',label=name)
    axes[1].set(xlabel='Prediction month',ylabel='Fraction of artificial queries with no fraud',title='All queries retained; grouping ignores labels');axes[1].set_xticks([4,5,6]);axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'ranking_operating.png',dpi=160);plt.close(fig)
    lookup=summary.set_index(['candidate','policy']);prior={alias:float(lookup.loc[(name,'pooled'),'f2']) for alias,name in [('balanced','baseline_balanced'),('unweighted','baseline_lgbm')]}
    dg={n:float(winner.f2)-v for n,v in prior.items()};fg={r.reference:float(r.difference) for r in fint.itertuples()}
    clear_dev=all(float(dint[(dint.scope=='macro')&(dint.reference==n)].iloc[0].ci_low)>0 for n in ['balanced','unweighted'])
    clear_final=all(float(fint[fint.reference==n].iloc[0].ci_low)>0 for n in ['balanced','unweighted'])
    measured=all(dg[n]>0 and fg[n]>0 for n in ['balanced','unweighted'])
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,measured_gain_on_both_scopes=measured,
        development_gain=dg,consumed_gain=fg,conditional_clear_development_gain=clear_dev,conditional_clear_consumed_gain=clear_final,
        independent_metrics_checked=True,prior_preservation_checked=True,prediction_results=len(records),forward_rows=len(cf),
        query_rows=len(group_rows),group_summaries=len(gs),score_semantics='raw finite scores, not probabilities; no Brier/log loss',
        report_source_sha256=digest('baf/direct_ranking_report.py'),uncertainty='500 paired fixed-decision row draws; no selection/retraining/time/generative/multiplicity correction')
    write_json(dest/'audit.json',audit);guard(c)
    best_rank=summary[summary.candidate.isin([x['id'] for x in c['presets'] if x['kind']=='rank'])].iloc[0]
    contrasts=pd.read_csv(out/'contrasts.csv');matched=contrasts[contrasts.contrast.isin(['rank minus binary','group 1024 minus 256','lambda 1 minus 0.1'])]
    text=f'''# Direct fraud-ranking study findings

Selected **{s['selected_candidate']} / {s['selected_policy']}** scores actual macro chronological F2 **{winner.f2:.6f}**, against balanced **{prior['balanced']:.6f}** and unweighted **{prior['unweighted']:.6f}**. Best ranker: **{best_rank.candidate} / {best_rank.policy}**, F2 **{best_rank.f2:.6f}**. Positive measured gain against both old references on development and consumed transfer: **{measured}**. Search completion alone does not establish the performance objective.

## Design and integration

Six new presets produce 18 full development fits; three old references provide nine results. Four miniature integration fits are counted separately and verify deterministic repeats and zero initialization. Models fit only earlier months. Native categorical levels are training-fitted; numeric NaNs remain missing. The four rank_xendcg presets cross artificial query size 256/1024 with lambda 0.1/1; two matched binary-logistic controls use the same ordered rows and lambda values. Training weights and effective regularization are not bank costs. Beta remains 2 without a fixed quota or precision floor; settings are analyst-selected.

Queries use month/IDs/seed only, keep every row once and retain all-negative groups. Single-row tails merge with the previous group without using labels; month boundaries remain intact. queries.csv and group_summary.csv reconstruct from training identities, with class counts added after grouping. These artificial queries are optimization devices, not real bank sessions or review batches. group_summary.csv marks controls' descriptive group metadata as unused by their objective. Query size and loss scale interact with regularization, so matched differences do not isolate a universal ranking effect.

Saved models reproduce raw finite predictions. Thresholds and AP/ROC use raw scores directly, including negative values; no sigmoid/clipping or fraud-probability claim is made. Old references retain original score scales, and policies fit within each candidate. No cross-model score average, Brier or log loss is reported. NDCG training motivation and higher AP do not substitute for actual F2.

## Forward actual performance

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

All 54 monthly operating rows, 27 macro summaries, pooled confusion-count summaries and unsuccessful contrasts remain visible. Policy fitting uses only earlier out-of-time predictions; all decisions and confusion/F2 values reconstruct independently. Development evidence has been adaptively reused across seven earlier studies.

## Matched comparisons

{markdown(matched[['contrast','candidate','policy','reference','f2_difference','recall_difference','flagged_fraction_difference']])}

## Paired conditional development differences

{markdown(dint)}

## Already-consumed month 7: exploratory transfer

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

{markdown(fint)}

Selection preceded further access and all hashes agree. Only frozen selected/matched controls, same-model pooled and both old pooled references were evaluated. No losing-preset sweep, hindsight threshold optimum or month-7 retuning occurred. A development gain that fails transfer is not out-of-time improvement. This consumed period cannot provide fresh independent confirmation.

## Uncertainty, preservation and reproduction

Intervals use 500 paired fixed-decision row draws, seed1027. Development resamples each month separately and averages F2 differences. Models, selection and decisions remain fixed; intervals exclude adaptive selection, retraining, temporal/generative uncertainty and multiplicity correction. Zero overlap precludes clear/confirmed gain claims; statistical significance is not silently added as a requirement for a positive measured difference. All seven old source/config/software/selection identities remain preserved. Exact provenance, group/model identities, config, freeze/access records and audit are exported. README gives pinned dependencies, verified Base access, seven-record restoration and missing-cache reproduction in new destinations. Codex assisted with design, code, execution, methodological checks, figures and interpretation. The official LightGBM ranker documentation and Bruch ranking paper inform the method; no public notebook was copied, and retrieval benchmark gains are not BAF evidence.

![Operating and grouping diagnostics](ranking_operating.png)
'''
    (dest/'findings.md').write_text(text);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=CONFIG);a=p.parse_args();report(a.config)
