"""Audit and report the adaptive F2 extension without modifying frozen runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import digest,load_data,read_config,write_json
from .improvements import paths,prediction
from .recall_extension import (audit_bases,base_history,crossfit,guard,load_selection,
                               load_npz,order_summary,save_derived,selected_data)
from .recall_extension_metrics import construct
from .recall_weighted_metrics import aligned,decisions,fbeta,measure,paired_f2


def agreement(a,b,keys):
    for k in keys:
        if not np.isclose(a[k],b[k],rtol=1e-10,atol=1e-12): raise ValueError(f"Reported {k} mismatch")


def intervals_development(config,selected,cf,history):
    rng=np.random.default_rng(config["bootstrap_seed"]);rows=[];draws=[]
    for month in config["validation_months"][1:]:
        items=construct(history,config,month)
        names=[selected["selected_candidate"],"baseline_balanced","baseline_lgbm"]
        data={};rules={}
        for i,name in enumerate(names):
            key=["selected","balanced","unweighted"][i]
            kind=selected["selected_policy"] if i==0 else "pooled"
            row=cf[(cf.candidate==name)&(cf.policy==kind)&(cf.eval_month==month)].iloc[0]
            data[key]=items[name][0];rules[key]=json.loads(row.parameters)
        preds={n:decisions(data[n],rules[n]).astype(np.int64) for n in data}
        y=data["selected"]["labels"].astype(np.int64)
        for n in data:aligned(data["selected"],data[n])
        samples=[]
        for _ in range(config["bootstrap_replicates"]):
            w=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y));values={}
            for name,pred in preds.items():
                tp=np.dot(w,y*pred);fp=np.dot(w,(1-y)*pred)
                values[name]=float(fbeta(tp,fp,np.dot(w,y)-tp))
            samples.append([values["selected"]-values[n] for n in ["balanced","unweighted"]])
        samples=np.array(samples);draws.append(samples)
        for i,name in enumerate(["balanced","unweighted"]):
            diff=measure(data["selected"],rules["selected"])["f2"]-measure(data[name],rules[name])["f2"]
            rows.append(dict(eval_month=str(month),reference=name,difference=diff,ci_low=float(np.quantile(samples[:,i],.025)),ci_high=float(np.quantile(samples[:,i],.975))))
    macro=np.mean(draws,axis=0)
    for i,name in enumerate(["balanced","unweighted"]):
        diff=np.mean([r["difference"] for r in rows if r["reference"]==name])
        rows.append(dict(eval_month="macro",reference=name,difference=float(diff),ci_low=float(np.quantile(macro[:,i],.025)),ci_high=float(np.quantile(macro[:,i],.975))))
    return pd.DataFrame(rows)


def contrasts(config,summary,cf):
    idx=summary.set_index(["candidate","policy"]);rows=[]
    def add(label,a,b):
        for policy in config["policies"]:
            x,y=idx.loc[(a,policy)],idx.loc[(b,policy)]
            rows.append(dict(contrast=label,candidate=a,reference=b,policy=policy,f2_difference=x.f2-y.f2,
                             precision_difference=x.precision-y.precision,recall_difference=x.recall-y.recall,
                             workload_difference=x.flagged_fraction-y.flagged_fraction))
    for leaves in [15,31]:
        for rounds in [200,600]:
            for weight in [30,100]:add("matched positive weight",f"ext_l{leaves}_w{weight}_r{rounds}",f"ext_l{leaves}_w1_r{rounds}")
    for weight in [1,30,100]:
        for leaves in [15,31]:add("matched boosting duration",f"ext_l{leaves}_w{weight}_r600",f"ext_l{leaves}_w{weight}_r200")
        for rounds in [200,600]:add("matched leaf capacity",f"ext_l31_w{weight}_r{rounds}",f"ext_l15_w{weight}_r{rounds}")
    for k in [100000,250000,500000]:
        for seed in [17,29,43]:
            add("recent minus size/class matched history",f"window_recent_k{k}_s17",f"window_historical_matched_k{k}_s{seed}")
            add("historical seed mean minus constituent",f"historical_mean_k{k}",f"window_historical_matched_k{k}_s{seed}")
    dynamic=[]
    for spec in config["rank_ensembles"]:
        for month in config["validation_months"][1:]:
            part=cf[(cf.candidate==spec["id"])&(cf.eval_month==month)]
            recipe=json.loads(part.iloc[0].recipe)
            for kind in config["policies"]:
                a=part[part.policy==kind].iloc[0]
                for ref in [recipe["anchor"],recipe["member"]]:
                    b=cf[(cf.candidate==ref)&(cf.policy==kind)&(cf.eval_month==month)].iloc[0]
                    dynamic.append(dict(candidate=spec["id"],reference=ref,eval_month=month,policy=kind,f2_difference=a.f2-b.f2,
                                        precision_difference=a.precision-b.precision,recall_difference=a.recall-b.recall))
    policies=[]
    for name in summary.candidate.unique():
        for kind in ["latest","rank_fraction"]:
            x,y=idx.loc[(name,kind)],idx.loc[(name,"pooled")]
            policies.append(dict(candidate=name,policy=kind,f2_difference=x.f2-y.f2,precision_difference=x.precision-y.precision,recall_difference=x.recall-y.recall))
    return pd.DataFrame(rows),pd.DataFrame(dynamic),pd.DataFrame(policies)


def markdown(table,columns):
    rows=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for _,r in table.iterrows():rows.append("| "+" | ".join(f"{r[c]:.4f}" if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+" |")
    return "\n".join(rows)


def report(config_path):
    config,selected=load_selection(config_path);out=Path(config["output_dir"]);dest=Path(config["report_dir"]);dest.mkdir(parents=True,exist_ok=True)
    frame=load_data(config);records=audit_bases(config,selected["fingerprint"],frame)
    final_records=audit_bases(config,selected["fingerprint"],frame,"exploratory",selected["fit_specs"])
    expected={(s["id"],m) for s in config["candidates"] for m in config["validation_months"]}
    actual=[(r["candidate"],r["eval_month"]) for r in records]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError("Incomplete base grid")
    history=base_history(config);cf=crossfit(config,history);summary=order_summary(cf)
    stored=pd.read_csv(out/"crossfit.csv")
    for col in ["candidate","policy","eval_month","source_months","recipe","parameters"]:
        if list(cf[col].astype(str))!=list(stored[col].astype(str)):
            raise ValueError("Saved forward strategy/policy provenance mismatch")
    for col in cf.select_dtypes(include="number").columns:
        if not np.allclose(cf[col],stored[col]):raise ValueError("Saved forward metrics mismatch")
    top=summary.iloc[0]
    if (top.candidate,top.policy)!=(selected["selected_candidate"],selected["selected_policy"]) or not np.isclose(top.f2,selected["selected_development_f2"],atol=1e-12):
        raise ValueError("Frozen selection differs from deterministic development winner")
    for spec in config["seed_means"]:
        for month in config["validation_months"]:save_derived(config,spec["id"],month,history[spec["id"]][month],spec,selected["fingerprint"])
    for month in config["validation_months"][1:]:
        for name,(data,past,recipe) in construct(history,config,month).items():
            if recipe["kind"]=="rank_blend":save_derived(config,name,month,data,recipe,selected["fingerprint"])
    raw={s["id"]:prediction(config,"exploratory",config["test_month"],s["id"]) for s in selected["fit_specs"]}
    final_data=selected_data(config,selected,raw);final=pd.read_csv(out/"exploratory.csv")
    metrics=["f2","f1","f3","precision","recall","tp","fp","fn","tn","flagged_fraction","flagged","n","fraud","average_precision","roc_auc","mcc","balanced_accuracy","fpr","accuracy"]
    for r in final.itertuples():
        rule=selected["rules"][r.candidate][r.policy]
        if json.loads(r.parameters)!=rule:raise ValueError("Policy differs from freeze")
        agreement(measure(final_data[r.candidate],rule),r._asdict(),metrics)
        saved=prediction(config,"exploratory",config["test_month"],r.candidate)
        aligned(saved,final_data[r.candidate])
        if not np.array_equal(saved["scores"],final_data[r.candidate]["scores"]):raise ValueError("Final derived scores mismatch")
    access=read_config(out/"exploratory_access.json")
    if access["selection_sha256"]!=digest(out/"selection.json") or not selected["created_at_utc"]<access["access_at_utc"]:raise ValueError("Freeze/access record mismatch")
    dev_ci=intervals_development(config,selected,cf,history)
    matched,ensemble,policy=contrasts(config,summary,cf)
    intervals=pd.read_csv(out/"paired_intervals.csv")
    paired_data={};paired_rules={}
    for name,data in final_data.items():
        for kind,rule in selected["rules"][name].items():
            key=name+"__"+kind;paired_data[key]=data;paired_rules[key]=rule
    rebuilt=pd.DataFrame(paired_f2(paired_data,paired_rules,"baseline_balanced__pooled",config["bootstrap_replicates"],config["bootstrap_seed"]))
    for col in rebuilt.columns:
        if pd.api.types.is_numeric_dtype(rebuilt[col]):
            if not np.allclose(rebuilt[col],intervals[col]):raise ValueError("Saved paired interval mismatch")
        elif list(rebuilt[col])!=list(intervals[col]):raise ValueError("Saved paired comparison mismatch")
    # Add the selected comparison to the stronger observed unweighted reference too.
    sname=selected["selected_candidate"];skind=selected["selected_policy"]
    ci_extra=paired_f2(dict(selected=final_data[sname],unweighted=final_data["baseline_lgbm"]),
                      dict(selected=selected["rules"][sname][skind],unweighted=selected["rules"]["baseline_lgbm"]["pooled"]),"unweighted",config["bootstrap_replicates"],config["bootstrap_seed"])
    ci_extra=pd.DataFrame(ci_extra)
    pooled=cf.groupby(["candidate","policy"])[["tp","fp","fn"]].sum().reset_index();pooled["pooled_f2"]=fbeta(pooled.tp,pooled.fp,pooled.fn)
    for name,table in [("crossfit",cf),("summary",summary),("exploratory",final),("matched_contrasts",matched),("ensemble_contrasts",ensemble),
                       ("policy_contrasts",policy),("development_intervals",dev_ci),("paired_intervals",intervals),("selected_vs_unweighted_interval",ci_extra),("pooled_development",pooled)]:
        table.to_csv(dest/(name+".csv"),index=False)
    write_json(dest/"config.json",config);write_json(dest/"selection.json",selected)
    winner=final[(final.candidate==sname)&(final.policy==skind)].iloc[0]
    balanced=final[(final.candidate=="baseline_balanced")&(final.policy=="pooled")].iloc[0]
    unweighted=final[(final.candidate=="baseline_lgbm")&(final.policy=="pooled")].iloc[0]
    dev_ref=summary[(summary.candidate=="baseline_balanced")&(summary.policy=="pooled")].iloc[0]
    dev_gain=float(selected["selected_development_f2"]-dev_ref.f2)
    audit=dict(base_development_results=len(records),new_development_fits=sum(not r["reused"] for r in records),reused_development_results=sum(r["reused"] for r in records),
               strategy_count=int(summary.candidate.nunique()),crossfit_rows=len(cf),new_exploratory_fits=sum(not r["reused"] for r in final_records),
               reused_exploratory_results=sum(r["reused"] for r in final_records),selected_candidate=sname,selected_policy=skind,
               development_f2_gain=dev_gain,consumed_month7_gain_vs_balanced=float(winner.f2-balanced.f2),consumed_month7_gain_vs_unweighted=float(winner.f2-unweighted.f2),
               performance_goal_achieved=bool(dev_gain>0 and winner.f2>max(balanced.f2,unweighted.f2)),
               saved_predictions_verified_and_reported_metrics_recomputed=True,original_training_hashes_and_class_matches_pass=True,
               identical_evaluation_cases=True,earlier_only_member_and_policy_sources_pass=True,prior_studies_preserved=True,
               source_config_software_identity_pass=True,selection_unchanged_after_access=True,freeze_precedes_exploratory_access=True,month7_previously_consumed=True,
               config_sha256=digest(config_path),selection_sha256=digest(out/"selection.json"),report_source_sha256=digest("baf/recall_extension_report.py"))
    guard(config);write_json(dest/"audit.json",audit)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));plt.rcParams.update({"font.size":11})
    for label,name,kind,end in [("Previous F2 baseline","baseline_balanced","pooled",balanced.f2),("F2-selected strategy",sname,skind,winner.f2),("Unweighted reference","baseline_lgbm","pooled",unweighted.f2)]:
        values=list(cf[(cf.candidate==name)&(cf.policy==kind)].sort_values("eval_month").f2)+[end]
        axes[0].plot([5,6,7],values,marker="o",label=label)
    old_names={s["id"] for s in read_config(config["previous_f2_config"])["candidates"]}
    new_best=summary[~summary.candidate.isin(old_names)].iloc[0]
    new_rows=cf[(cf.candidate==new_best.candidate)&(cf.policy==new_best.policy)].sort_values("eval_month")
    axes[0].plot(new_rows.eval_month,new_rows.f2,marker="s",linestyle="--",label="Best added strategy (development only)")
    axes[0].set(xticks=[5,6,7],xlabel="Evaluation month (7 already consumed)",ylabel="F2",title="Chronological F2");axes[0].legend(fontsize=9)
    pos=np.arange(3)
    axes[1].bar(pos-.18,100*np.array([balanced.precision,balanced.recall,balanced.flagged_fraction]),.36,label="Previous F2 baseline")
    axes[1].bar(pos+.18,100*np.array([winner.precision,winner.recall,winner.flagged_fraction]),.36,label="F2-selected strategy")
    axes[1].set(xticks=pos,xticklabels=["Precision","Recall","Flagged"],ylabel="Percent",title="Consumed month 7: detection and workload");axes[1].legend(fontsize=9)
    fig.tight_layout();fig.savefig(dest/"f2_extension.png",dpi=170);plt.close(fig)
    stronger_gain=float(winner.f2-max(balanced.f2,unweighted.f2))
    interval=ci_extra.iloc[0]
    findings=f'''# Adaptive BAF recall-weighted F2 extension

This study continues the user's unachieved recall-weighted detection goal after the prior F2 search produced no gain. It optimizes positive-class **F2=5TP/(5TP+4FN+FP)**, not support-weighted F1, and imposes no review quota or precision floor. Beta=2 and the finite presets are analyst-selected, not approved group decisions.

## Measured result

The earlier-only chronological selection is **{sname} / {skind}**, with macro month-5/6 F2 **{selected['selected_development_f2']:.4f}** versus **{dev_ref.f2:.4f}** for the previous balanced LightGBM F2 policy (difference **{dev_gain:+.4f}**).

On already-consumed month 7, the frozen strategy's F2 is **{winner.f2:.4f}**, versus balanced **{balanced.f2:.4f}** and unweighted **{unweighted.f2:.4f}**. Its difference from the stronger observed old baseline is **{stronger_gain:+.4f}**. {'The positive exploratory point gain exceeds both previous baseline policies.' if stronger_gain>0 else 'It does not beat the stronger observed prior baseline on this consumed period.'} Its paired conditional difference interval versus unweighted is **[{interval.ci_low:.4f}, {interval.ci_high:.4f}]**. {'The interval includes zero; the gain is uncertain.' if interval.ci_low<=0<=interval.ci_high else 'The interval excludes zero conditional on fixed models and decisions; this is still exploratory evidence.'}

Month 7 has already been consumed; no new holdout was created. Repeated development tuning and adaptive extension can be optimistic. A finished search is not itself an improvement, and a positive conditional point estimate is not independent confirmation.

{'The measured point gain is limited to this exploratory study.' if audit['performance_goal_achieved'] else 'The extension did not establish the requested performance improvement; the goal remains unachieved.'}

## Selected operating results

{markdown(final[(final.policy=='pooled')|final.selected],['candidate','policy','f2','f1','f3','precision','recall','flagged_fraction','tp','fp','fn'])}

The selected strategy catches **{int(winner.tp):,}** frauds, misses **{int(winner.fn):,}** and flags **{int(winner.fp):,}** legitimate applications, reviewing **{winner.flagged_fraction:.2%}**. All selected/baseline policies were fitted on development predictions and frozen before this further access. Other declared policies are diagnostics, not post-hoc replacements for the selected one.

![F2 and operating trade-off](f2_extension.png)

## Search and controls

{audit['new_development_fits']} new development fits and {audit['reused_development_results']} verified reuses cover 39 base models across months 4/5/6. Twelve native LightGBM presets cross weights 1/30/100, leaves 15/31 and rounds 200/600 with all other settings fixed. Fifteen prior F2 recipes remain references. Twelve original recent-K/historical-matched-K recipes retain exact original training rows, class counts and one-hot/imputation preprocessing. The [config](config.json), [protocol](../../docs/recall-weighted-extension-protocol.md) and [audit](audit.json) expose the recipes and identities.

Three historical-seed means and six earlier-selected batch rank ensembles give {audit['strategy_count']} strategies and {len(cf)} forward policy evaluations. Rank percentiles use unlabeled scores from the same application month, never its labels, to put differently weighted models on comparable scales. This requires batch scoring and does not produce calibrated fraud probabilities. Ensemble members can change across folds; [crossfit.csv](crossfit.csv) records their recipes and policy-source months.

{markdown(summary.head(12),['candidate','policy','f2','precision','recall','flagged_fraction'])}

The original unweighted and balanced macro F2 baselines reproduce. Macro monthly F2 is the selection objective; [pooled_development.csv](pooled_development.csv) separately reports pooled confusion-count F2.

## Ablations and uncertainty

[Matched contrasts](matched_contrasts.csv) include weight-1 controls, boosting duration, leaf capacity, recent-versus-size/class-matched history and seed averaging versus constituents. [Rank-ensemble contrasts](ensemble_contrasts.csv) compare each fold's actual selected member and anchor. [Policy contrasts](policy_contrasts.csv) hold model predictions fixed. These retain unsuccessful effects rather than treating every added strategy as beneficial; window comparisons do not establish causal recency effects.

{markdown(dev_ci,['eval_month','reference','difference','ci_low','ci_high'])}

Intervals use 500 paired row-bootstrap draws at fixed decisions, seed 1027. Development macro intervals resample within each month and average differences. They exclude tuning, retraining, temporal and generative uncertainty, have no multiplicity correction, and do not remove adaptive search optimism. Rank-policy uncertainty is conditional on realized decisions, not a bootstrap of reselected rankings. Month-7 policy comparisons are in [paired_intervals.csv](paired_intervals.csv) and the selected comparison to unweighted in [selected_vs_unweighted_interval.csv](selected_vs_unweighted_interval.csv).

## Verification, reproduction and limits

The audit checks base and derived predictions, direct metrics, original training-row hashes and matched class counts, identical future applications, earlier-only policies/members, preserved old artifacts, software/source/config identity and unchanged selection after access. Exploratory transfer needed {audit['new_exploratory_fits']} new dependency fits and {audit['reused_exploratory_results']} verified reuses. Raw data and bulk outputs remain ignored.

Use the [README](../../README.md) sequence: validate, freeze, exploratory, report. Missing caches are refitted into this study's directories using the same frozen recipes; older output directories are never rewritten. Fresh-clone instructions restore the small exported frozen records first. Further tuning requires another separate study.

One synthetic variant, only eight monthly periods, two operating-policy evaluation months, coarse time, unknown label delays and no applicant identifiers limit generalization. Higher F2 can still entail many false positives and missed fraud; workload and precision remain necessary operating judgments. New data is required for independent confirmation.

Codex assisted with design, implementation, verification, plots and interpretation. Acknowledge BAF, scikit-learn, LightGBM and CatBoost in the eventual course report. This is an experiment report, not the final 8-12 page submission.
'''
    (dest/"findings.md").write_text(findings);print(json.dumps(audit,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/recall-weighted-extension.json");report(p.parse_args().config)
