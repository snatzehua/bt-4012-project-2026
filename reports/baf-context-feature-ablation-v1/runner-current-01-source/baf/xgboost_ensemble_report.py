"""Reconstruct ensemble evidence and conditional fixed-decision uncertainty."""
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
from scipy.stats import spearmanr
from .common import digest,features,load_data,read_config,write_json
from .recall_weighted_metrics import decisions,measure,learn
from .calibrated_policy_report import independent_check,markdown
from .direct_ranking_report import intervals,development
from .improvement_metrics import confusion_metrics
from .rank_ensemble_scores import percentile,choices_for
from .xgboost_ensemble import (CONFIG,load_selection,audit_grid,history,crossfit,verify_table,order_summary,
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
    for r in pd.read_csv(out/'diversity.csv').itertuples():
        # Independent correlation of average rank vectors.
        a=percentile(h[r.candidate][r.eval_month]['scores']);b=percentile(h['baseline_balanced'][r.eval_month]['scores'])
        actual=float(np.corrcoef(a,b)[0,1])
        if not np.isclose(actual,r.spearman_with_balanced,atol=1e-12,rtol=0):raise ValueError('Diversity correlation differs')
    for r in pd.read_csv(out/'score_diagnostics.csv').itertuples():
        x=h[r.candidate][r.eval_month]['scores']
        if len(x)!=r.n or len(np.unique(x))!=r.unique_scores:raise ValueError('Score counts differ')
        if not np.allclose([x.min(),x.max(),x.mean(),1-len(np.unique(x))/len(x)],
            [r.score_min,r.score_max,r.score_mean,r.tied_fraction],atol=1e-12,rtol=0):raise ValueError('Score diagnostic differs')
    for name,count in [('crossfit',150),('summary',75),('contrasts',302),('pooled',75),('score_diagnostics',75),('diversity',12)]:
        if len(pd.read_csv(out/(name+'.csv')))!=count:raise ValueError('Table count differs: '+name)


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
    for name,table in [('crossfit',cf),('summary',summary),('exploratory',final),('development_intervals',dint),('development_draws',ddraws),
        ('consumed_intervals',fint),('consumed_draws',fdraws.assign(replicate=np.arange(len(fdraws))))]:table.to_csv(dest/(name+'.csv'),index=False)
    for name in ['selection.json','validation_complete.json','integration_gate.json','exploratory_access.json','transfer_complete.json',
        'contrasts.csv','pooled.csv','score_diagnostics.csv','diversity.csv']:shutil.copyfile(out/name,dest/name)
    shutil.copyfile(config_path,dest/'config.json');write_json(dest/'run_provenance.json',records);write_json(dest/'transfer_provenance.json',final_records)
    derived=[read_config(p) for p in (out/'runs').glob('*.json') if read_config(p).get('transform')]
    write_json(dest/'score_provenance.json',derived)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));types={x['id']:x['kind'] for x in c['strategies']}
    for kind in ['raw','percentile','blend']:
        g=summary[summary.candidate.map(types)==kind];axes[0].scatter(g.recall,g.precision,s=28,label=kind)
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 75 strategy/policy comparisons');axes[0].legend(fontsize=8)
    diversity=pd.read_csv(out/'diversity.csv')
    for name,g in diversity.groupby('candidate'):axes[1].plot(g.eval_month,g.spearman_with_balanced,marker='o',label=name)
    axes[1].set(xlabel='Prediction month',ylabel='Spearman correlation with balanced LightGBM',title='Model diversity is descriptive');axes[1].set_xticks([4,5,6]);axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'ensemble_operating.png',dpi=160);plt.close(fig)
    lookup=summary.set_index(['candidate','policy']);dg={n:float(winner.f2-lookup.loc[(name,'pooled'),'f2']) for n,name in [('balanced','baseline_balanced'),('unweighted','baseline_lgbm')]}
    fg={r.reference:float(r.difference) for r in fint.itertuples()};measured=all(dg[n]>0 and fg[n]>0 for n in ['balanced','unweighted'])
    clear_dev=all(float(dint[(dint.scope=='macro')&(dint.reference==n)].iloc[0].ci_low)>0 for n in ['balanced','unweighted'])
    clear_final=all(float(fint[fint.reference==n].iloc[0].ci_low)>0 for n in ['balanced','unweighted'])
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,measured_gain_on_both_scopes=measured,development_gain=dg,consumed_gain=fg,
        conditional_clear_development_gain=clear_dev,conditional_clear_consumed_gain=clear_final,independent_metrics_checked=True,
        prior_preservation_checked=True,base_results=len(records),score_results=75,forward_rows=len(cf),macro_rows=len(summary),contrasts=302,
        raw_score_semantics='logistic scores not assumed calibrated; percentiles/blends are scores, not probabilities',
        report_source_sha256=digest('baf/xgboost_ensemble_report.py'),uncertainty='500 paired fixed-decision draws; excludes selection/retraining/batch/time/generative/multiplicity')
    write_json(dest/'audit.json',audit)
    best_raw=summary[summary.candidate.str.startswith('xgb_')].iloc[0];best_blend=summary[summary.candidate.str.startswith('blend_')].iloc[0]
    matched=pd.read_csv(out/'contrasts.csv');matched=matched[matched.contrast!='old pooled reference']
    text=f'''# XGBoost and rank-ensemble study findings

Selected **{winner.candidate} / {winner.policy}** scores actual macro chronological F2 **{winner.f2:.6f}**. Best new raw XGBoost: **{best_raw.candidate} / {best_raw.policy}**, F2 **{best_raw.f2:.6f}**. Best blend: **{best_blend.candidate} / {best_blend.policy}**, F2 **{best_blend.f2:.6f}**. Positive measured gain against both old references on development and consumed transfer: **{measured}**. Search completion alone does not establish the performance objective.

## Design and verified inference

Four fixed native categorical presets cross depth4/6 with positive weight1/5. Twelve full new development fits and nine verified reference results produce21 base predictions; six percentile representations and12 fixed blends produce75 total score results. Eight separate miniature fits verify deterministic repeats, missing/unseen categorical handling, class orientation and native saved-model reconstruction. Native category vocabularies are training-fitted, numeric NaNs remain missing, and all29 original eligible predictors exclude target/month/device_fraud_count. No evaluation-set early stopping or new feature engineering occurs. Old references retain their exact frozen recipes; cross-family preset differences do not isolate a universal algorithm effect.

Percentiles are average ascending rank/n within each current unlabeled application batch. Ties receive their average rank. Blends use predeclared members/weights0.25/0.5/0.75. Saved dependency hashes and composed predictions reconstruct exactly. These batch-dependent scores are not calibrated fraud probabilities. Raw/percentile/member controls distinguish representation from blending. Beta remains2 with no fixed review quota or precision floor; all settings are analyst-selected.

## Forward actual performance

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

All150 monthly rows,75 macro rows,75 pooled-confusion summaries and302 contrasts remain exported, including failures and zero/self comparisons. Models and preprocessing fit strictly earlier months. Month5 policies use month4; month6 policies use earlier4/5 or5. Actual macro F2 selects the winner, not AP, diversity or training loss. Eight prior studies have adaptively reused development data.

The complete matched comparisons are in contrasts.csv; they include percentile-minus-raw, blend-minus-each-percentile-member, policy-minus-own-pooled, and raw depth/weight changes. All arithmetic was independently reconstructed. diversity.csv contains12 label-free rank correlations; score_diagnostics.csv contains75 range/tie summaries. Diversity alone proves no useful fraud gain.

## Paired conditional development differences

{markdown(dint)}

## Already-consumed month7: exploratory transfer

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

{markdown(fint)}

Only the actual winner, same-strategy pooled rule, both old pooled references and the controls determined by its declared strategy type were evaluated. Base fits are deduplicated while all roles remain visible. Freeze/control/dependency hashes and access chronology agree. No losing-setting sweep, hindsight month7 threshold or retuning occurred. A development gain that fails transfer is not out-of-time improvement. This period cannot provide fresh independent confirmation.

## Uncertainty, preservation and reproduction

Intervals use500 paired fixed-decision row draws with seed1027. Development resamples each month separately and averages monthly F2 differences. Models, selection, percentile batches and decisions remain fixed; intervals exclude adaptive search/selection, retraining, batch-composition, temporal/generative uncertainty and multiplicity correction. Intervals overlapping zero preclude clear/confirmed gain claims; statistical significance is not substituted for a positive measured difference. All eight earlier source/config/software/selection identities remain preserved. Exact model/score provenance, config, freeze/access records, draws and audit are exported. README provides pinned install, verified Base access, eight-record restoration and new-destination reference reproduction.

Codex assisted with design, implementation, execution, methodological checks, figures and interpretation. The official XGBoost categorical/parameter documentation and pinned package informed the model interface; stable online docs were newer than the pinned installed version, whose contract was checked locally. No public notebook was copied. These synthetic-data experiments are supporting material, not a bank deployment claim.

![Operating and diversity diagnostics](ensemble_operating.png)
'''
    (dest/'findings.md').write_text(text);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=CONFIG);a=p.parse_args();report(a.config)
