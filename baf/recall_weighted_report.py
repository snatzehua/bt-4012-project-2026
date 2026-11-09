"""Recompute F2 evidence, export comparisons and render inspectable findings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import digest, load_data, read_config, write_json
from .improvements import paths, prediction
from .recall_weighted import (audit_runs, cf_predictions, crossfit, fingerprint, guard,
                             histories, ids_digest, load_selection)
from .recall_weighted_metrics import aligned, blend, decisions, fbeta, measure, paired_f2


def numeric_agreement(actual, expected, keys):
    for key in keys:
        if not np.isclose(actual[key], expected[key], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Recomputed {key} differs: {actual[key]} vs {expected[key]}")


def development_intervals(config, selection, cf):
    history = histories(config)
    rows, monthly_draws = [], []
    rng = np.random.default_rng(config["bootstrap_seed"])
    for month in config["validation_months"][1:]:
        entries = cf_predictions(history, config, month)
        selected_data = entries[selection["selected_candidate"]][0]
        reference_data = entries["baseline_balanced"][0]
        chosen = cf[(cf.candidate == selection["selected_candidate"]) & (cf.policy == selection["selected_policy"]) & (cf.eval_month == month)].iloc[0]
        reference = cf[(cf.candidate == "baseline_balanced") & (cf.policy == "pooled") & (cf.eval_month == month)].iloc[0]
        policies = dict(selected=json.loads(chosen.parameters), baseline=json.loads(reference.parameters))
        aligned(selected_data, reference_data)
        intervals = paired_f2(dict(selected=selected_data,baseline=reference_data),policies,"baseline",config["bootstrap_replicates"],config["bootstrap_seed"]+month)
        rows.append(dict(eval_month=month,**intervals[0]))
        y = reference_data["labels"].astype(np.int64)
        preds = {n:decisions(selected_data if n == "selected" else reference_data,p).astype(np.int64) for n,p in policies.items()}
        draws=[]
        for _ in range(config["bootstrap_replicates"]):
            counts=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y))
            values={}
            for name,pred in preds.items():
                tp=np.dot(counts,y*pred);fp=np.dot(counts,(1-y)*pred)
                values[name]=float(fbeta(tp,fp,np.dot(counts,y)-tp))
            draws.append(values["selected"]-values["baseline"])
        monthly_draws.append(draws)
    macro=np.mean(monthly_draws,axis=0)
    rows.append(dict(eval_month="macro",candidate="selected",baseline="baseline",difference=float(np.mean([r["difference"] for r in rows])),
                     ci_low=float(np.quantile(macro,.025)),ci_high=float(np.quantile(macro,.975)),replicates=config["bootstrap_replicates"],
                     uncertainty="conditional paired within-month row bootstrap of macro F2; fixed decisions; no selection correction"))
    return pd.DataFrame(rows)


def comparisons(summary):
    indexed=summary.set_index(["candidate","policy"])
    rows=[]
    def add(label,a,b):
        for policy in ["pooled","latest","rank_fraction"]:
            left,right=indexed.loc[(a,policy)],indexed.loc[(b,policy)]
            rows.append(dict(contrast=label,candidate=a,reference=b,policy=policy,
                             f2_difference=left.f2-right.f2,recall_difference=left.recall-right.recall,
                             precision_difference=left.precision-right.precision,flagged_fraction_difference=left.flagged_fraction-right.flagged_fraction))
    for leaves in [15,31,63]:
        for weight in [3,10]: add(f"LightGBM weight {weight} vs 1 at {leaves} leaves",f"lgb{leaves}_w{weight}",f"lgb{leaves}_w1")
    for leaves in [31,63]:
        for weight in [1,3,10]: add(f"LightGBM {leaves} vs 15 leaves at weight {weight}",f"lgb{leaves}_w{weight}",f"lgb15_w{weight}")
    for name in ["catboost_weight3","catboost_weight10"]: add("CatBoost weight vs 1",name,"catboost_engineered6")
    policies=[]
    for name in summary.candidate.unique():
        ref=indexed.loc[(name,"pooled")]
        for policy in ["latest","rank_fraction"]:
            item=indexed.loc[(name,policy)]
            policies.append(dict(candidate=name,policy=policy,reference_policy="pooled",f2_difference=item.f2-ref.f2,
                                 precision_difference=item.precision-ref.precision,recall_difference=item.recall-ref.recall))
    return pd.DataFrame(rows),pd.DataFrame(policies)


def report(config_path):
    config,selected=load_selection(config_path)
    out,report_dir=Path(config["output_dir"]),Path(config["report_dir"])
    report_dir.mkdir(parents=True,exist_ok=True)
    frame=load_data(config)
    records=audit_runs(config,selected["fingerprint"])
    if len(records) != len(config["candidates"])*len(config["validation_months"]):
        raise ValueError("Unexpected number of base run records")
    audit_runs(config,selected["fingerprint"],"exploratory")
    for record in records:
        month,name=record["eval_month"],record["candidate"]
        data=prediction(config,"validation",month,name)
        ids=frame.index[frame.month==month].to_numpy()
        expected=dict(row_ids=ids,labels=frame.loc[ids,"fraud_bool"].to_numpy(),scores=np.zeros(len(ids)))
        aligned(data,expected)
        if record["train_ids_sha256"] != ids_digest(frame.index[frame.month<month].to_numpy()):
            raise ValueError("Full-data training audit failed")
        # Also verify scores in reused predictions against the original recorded AP.
        stats=measure(data,dict(kind="pooled",threshold=.5))
        numeric_agreement(stats,record,["average_precision","roc_auc"])
    cf=crossfit(config)
    saved_cf=pd.read_csv(out/"crossfit.csv")
    for col in cf.select_dtypes(include="number").columns:
        if not np.allclose(cf[col],saved_cf[col]): raise ValueError("Crossfit export mismatch")
    summary=pd.read_csv(out/"summary.csv")
    recomputed=cf.groupby(["candidate","policy"])[["f2","precision","recall"]].mean()
    for row in summary.itertuples():
        numeric_agreement(row._asdict(),recomputed.loc[(row.candidate,row.policy)].to_dict(),["f2","precision","recall"])
    final=pd.read_csv(out/"exploratory.csv")
    final_data={s["id"]:prediction(config,"exploratory",config["test_month"],s["id"]) for s in selected["fit_specs"]}
    for name,spec in selected["blends"].items():
        final_data[name]=blend(*[final_data[m] for m in spec["members"]],spec["weight"])
    for row in final.itertuples():
        numeric_agreement(measure(final_data[row.candidate],selected["rules"][row.candidate][row.policy]),row._asdict(),
                          ["f2","f1","f3","precision","recall","tp","fp","fn","tn","flagged_fraction",
                           "flagged","n","fraud","average_precision","roc_auc","mcc","balanced_accuracy","fpr","accuracy"])
    access=read_config(out/"exploratory_access.json")
    if not selected["created_at_utc"] < access["access_at_utc"] or access["selection_sha256"] != digest(out/"selection.json"):
        raise ValueError("Freeze/access ordering or selection hash mismatch")
    dev_ci=development_intervals(config,selected,cf)
    ablations,policy_comparisons=comparisons(summary)
    grouped=cf.groupby(["candidate","policy"])[["tp","fp","fn"]].sum().reset_index()
    grouped["pooled_f2"]=fbeta(grouped.tp,grouped.fp,grouped.fn)
    for name,table in [("crossfit.csv",cf),("summary.csv",summary),("exploratory.csv",final),
                       ("development_intervals.csv",dev_ci),("ablations.csv",ablations),
                       ("policy_comparisons.csv",policy_comparisons),("pooled_development.csv",grouped)]:
        table.to_csv(report_dir/name,index=False)
    intervals=pd.read_csv(out/"paired_intervals.csv")
    intervals.to_csv(report_dir/"paired_intervals.csv",index=False)
    write_json(report_dir/"selection.json",selected)
    write_json(report_dir/"config.json",config)
    chosen=final[(final.candidate==selected["selected_candidate"]) & (final.policy==selected["selected_policy"])].iloc[0]
    ref=final[(final.candidate=="baseline_balanced") & (final.policy=="pooled")].iloc[0]
    old_unweighted=final[(final.candidate=="baseline_lgbm") & (final.policy=="pooled")].iloc[0]
    dev_ref=summary[(summary.candidate=="baseline_balanced") & (summary.policy=="pooled")].iloc[0]
    dev_gain=selected["selected_development_f2"]-dev_ref.f2
    transfer_gain=chosen.f2-ref.f2
    audit=dict(validation_results=len(records),new_development_fits=sum(not r["reused"] for r in records),
               reused_development_results=sum(r["reused"] for r in records),crossfit_rows=len(cf),
               new_exploratory_fits=sum(not read_config(p).get("reused",False) for p in (out/"runs").glob("exploratory-*.json")),
               reused_exploratory_results=sum(read_config(p).get("reused",False) for p in (out/"runs").glob("exploratory-*.json")),
               selected_candidate=selected["selected_candidate"],selected_policy=selected["selected_policy"],
               development_f2_gain=float(dev_gain),consumed_month7_f2_gain=float(transfer_gain),
               consumed_month7_gain_vs_unweighted=float(chosen.f2-old_unweighted.f2),
               saved_predictions_verified_and_reported_metrics_recomputed=True,source_config_software_identity_pass=True,
               training_populations_temporal_boundaries_pass=True,evaluation_cases_identical=True,
               prior_studies_preserved=True,freeze_precedes_exploratory_access=True,selection_unchanged_after_access=True,
               month7_previously_consumed=True,config_sha256=digest(config_path),selection_sha256=digest(out/"selection.json"),
               report_source_sha256=digest("baf/recall_weighted_report.py"))
    guard(config)
    write_json(report_dir/"audit.json",audit)
    # Show month-to-month performance and the actual operating trade-off.
    plt.rcParams.update({"font.size":11})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    a=cf[(cf.candidate==selected["selected_candidate"]) & (cf.policy==selected["selected_policy"])].sort_values("eval_month")
    b=cf[(cf.candidate=="baseline_balanced") & (cf.policy=="pooled")].sort_values("eval_month")
    axes[0].plot([5,6,7],list(b.f2)+[ref.f2],marker="o",label="Previous F2 baseline")
    axes[0].plot([5,6,7],list(a.f2)+[chosen.f2],marker="o",label="F2-selected strategy")
    u=cf[(cf.candidate=="baseline_lgbm") & (cf.policy=="pooled")].sort_values("eval_month")
    axes[0].plot([5,6,7],list(u.f2)+[old_unweighted.f2],marker="o",linestyle="--",label="Unweighted reference")
    axes[0].set(xticks=[5,6,7],xlabel="Evaluation month (7 already consumed)",ylabel="F2",title="Chronological F2")
    axes[0].legend(fontsize=9)
    pos=np.arange(3)
    axes[1].bar(pos-.18,100*np.array([ref.precision,ref.recall,ref.flagged_fraction]),.36,label="Previous F2 baseline")
    axes[1].bar(pos+.18,100*np.array([chosen.precision,chosen.recall,chosen.flagged_fraction]),.36,label="F2-selected strategy")
    axes[1].set(xticks=pos,xticklabels=["Precision","Recall","Flagged"],ylabel="Percent",title="Consumed month 7: workload and detection")
    axes[1].legend(fontsize=9)
    fig.tight_layout();fig.savefig(report_dir/"f2_tradeoff.png",dpi=170);plt.close(fig)
    def md(table,cols):
        # Explicit formatting keeps small exported tables explainable and stable.
        rows=["| " + " | ".join(cols)+" |","| " + " | ".join(["---"]*len(cols))+" |"]
        for _,r in table.iterrows():
            rows.append("| "+" | ".join(f"{r[c]:.4f}" if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+" |")
        return "\n".join(rows)
    winner_interval=intervals[intervals.candidate==selected["selected_candidate"]+"__"+selected["selected_policy"]]
    interval_text=(f"The conditional month-7 paired F2 difference interval is [{winner_interval.iloc[0].ci_low:.4f}, {winner_interval.iloc[0].ci_high:.4f}]. "
                   + ("It includes zero, so the gain is uncertain." if winner_interval.iloc[0].ci_low<=0<=winner_interval.iloc[0].ci_high else "It excludes zero conditional on the fitted models and policies; it is still exploratory.")) if len(winner_interval) else "The selected strategy is the reference policy itself."
    findings=f'''# BAF recall-weighted F2 study

The user requested improved detection with greater emphasis on recall. The objective here is **F2**, not class-support-weighted F1, and there is **no imposed 1% review quota**. F2=5TP/(5TP+4FN+FP), so missed fraud has a fourfold denominator penalty relative to false alarms. Beta=2 remains analyst-selected.

## Measured result

The chronological development selection is **{selected['selected_candidate']} / {selected['selected_policy']}**, macro F2 **{selected['selected_development_f2']:.4f}** versus **{dev_ref.f2:.4f}** for the previous balanced LightGBM F2 policy (difference **{dev_gain:+.4f}**). Selection uses months 5/6 with policy parameters and blend members fitted only from earlier prediction months.

On already-consumed month 7, its F2 is **{chosen.f2:.4f}**, versus **{ref.f2:.4f}** for balanced LightGBM (difference **{transfer_gain:+.4f}**) and **{old_unweighted.f2:.4f}** for unweighted LightGBM. {'The selected gain transfers as a positive exploratory point estimate.' if transfer_gain>0 else 'The development selection does not improve the previous balanced F2 policy on this consumed period.'} {interval_text}

Month 7 is not fresh confirmation. Repeated development selection, prior dataset knowledge and the finite search make all new comparisons exploratory. A completed search does not by itself establish a performance gain.

## Selected operating points

{md(final[(final.policy=='pooled') | final.selected],['candidate','policy','f2','f1','f3','precision','recall','flagged_fraction','tp','fp','fn'])}

Fraud not caught: **{int(chosen.fn):,}**. Legitimate applications flagged: **{int(chosen.fp):,}**. The selected policy reviews **{chosen.flagged_fraction:.2%}** of applications. These costs remain visible even if F2 improves. Other policies below are diagnostics and are not substituted for the frozen selected policy after seeing month 7.

![F2 and operating trade-off](f2_tradeoff.png)

## Development search

{audit['new_development_fits']} new development fits and {audit['reused_development_results']} verified read-only reuses cover 15 base candidates across months 4/5/6. Nine LightGBM presets cross leaves 15/31/63 with positive weights 1/3/10, with all other training parameters fixed. CatBoost compares weights 1/3/10 using the same engineered depth-6 preset. The original unweighted/balanced LightGBM and native31 are retained references. Three family blends and three policies yield {len(cf)} cross-fitted evaluations. See [exact config](config.json) and [protocol](../../docs/recall-weighted-protocol.md).

{md(summary.head(12),['candidate','policy','f2','precision','recall','flagged_fraction'])}

The original balanced/unweighted pooled-policy macro F2 values reproduce exactly. Macro monthly F2 and pooled confusion-count F2 differ; [pooled_development.csv](pooled_development.csv) reports the latter without replacing the selection objective.

## Controlled contrasts and policy effects

The [matched ablations](ablations.csv) retain all weight and leaf-capacity contrasts, including negative effects. The [policy comparisons](policy_comparisons.csv) hold predictions fixed while varying earlier-only policy fitting. New models under the original pooled policy and old models under the new policies separate modelling changes from policy changes; a threshold-only gain is not claimed as improved score ranking.

Blends select each family's member using earlier prediction months only. Their members can differ across forward folds. Weighted member scores and raw-score averages are not calibrated fraud probabilities. Threshold optimization emphasizes F2 rather than enforcing 90% recall, a precision floor, or a review quota.

## Uncertainty, verification and limits

{md(dev_ci,['eval_month','difference','ci_low','ci_high'])}

These are conditional paired row-bootstrap intervals at fixed decisions (500 draws, seed 1027; per-month seeds recorded by implementation). The macro interval resamples within each month and averages month-level differences. They exclude tuning/retraining/time/generative uncertainty and multiplicity correction; selection optimism remains. Rank-fraction uncertainty is conditional on realized decisions, not a bootstrap of reselected review rankings. Month-7 intervals for every declared diagnostic are in [paired_intervals.csv](paired_intervals.csv).

The [audit](audit.json) verifies all saved metrics against predictions, source/config/software identity, eligible training populations, identical evaluation cases, preserved prior runs and an unchanged selection hash frozen before further month-7 access. {audit['new_exploratory_fits']} new selected/dependency fits and {audit['reused_exploratory_results']} verified reuses support the consumed-period comparison. Six threads and fixed iterations were used; no outer-fold early stopping or evaluation-fitted preprocessing is allowed.

One synthetic variant, eight coarse monthly periods, unknown label delays, no applicant/transaction identifiers and only two cross-fitted operating-policy evaluation months limit generalization. A feature bundle's success does not identify causal predictors. The settings are analyst-selected and remain open to group review. Further tuning needs a separate frozen study; new data is needed for independent confirmation.

## Reproduction and acknowledgment

Follow the [README](../../README.md) commands: development validation, selection freeze, exploratory transfer, then report. Bulk data/models/predictions remain ignored. Config and selection are exported here for review; checkpoints reject source/config/software drift and changed reuse artifacts.

Codex assisted with protocol design, modelling implementation, verification, plotting and interpretation. Acknowledge BAF, scikit-learn, LightGBM and CatBoost in the course report. These experiments do not constitute the final 8-12 page course deliverable.
'''
    (report_dir/"findings.md").write_text(findings)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/recall-weighted.json")
    report(parser.parse_args().config)
