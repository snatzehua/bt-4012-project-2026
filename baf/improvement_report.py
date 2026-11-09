"""Audit and report the completed exploratory improvement study."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .common import budget_name, digest, features, read_config, write_json
from .engineering import engineer
from .improvement_metrics import rank_metrics, threshold_stats
from .improvements import crossfit_policies, fingerprint, guard_preservation, load_selection, policies, pooled, prediction, records
from .report import md_table

BASELINE_MAP = {"baseline_lgbm":("lightgbm",False), "baseline_balanced":("lightgbm",True), "baseline_logistic":("logistic",False)}
CONTRASTS = [
    ("native_small","baseline_lgbm","Native missing/category handling"),
    ("native31","native_small","Longer 31-leaf preset"),
    ("native63","native31","63-leaf capacity preset"),
    ("engineered31","native31","Engineered features, 31 leaves"),
    ("engineered63","native63","Engineered features, 63 leaves"),
    ("weight3","engineered31","Positive weight 3"),
    ("weight10","engineered31","Positive weight 10"),
    ("recency2","engineered31","Two-month recency weighting"),
    ("undersample20","engineered31","20:1 negative undersampling"),
    ("logistic_engineered","baseline_logistic","Logistic features + regularization"),
    ("catboost8","catboost6","CatBoost deeper preset"),
    ("catboost_engineered6","catboost6","CatBoost engineered features"),
    ("catboost_weight3","catboost_engineered6","CatBoost positive weight 3"),
]


def audit(config_path,diagnostic_path="configs/logistic-ablation.json"):
    config, selected = load_selection(config_path)
    out = Path(config["output_dir"])
    validation, final = records(config,"validation"),records(config,"exploratory")
    blends=read_config(out/"blends.json")["specs"]
    names={s["id"] for s in config["candidates"]+blends}
    assert len(validation)==len(names)*len(config["validation_months"])==60
    assert {(r["candidate"],r["eval_month"]) for r in validation}=={(n,m) for n in names for m in config["validation_months"]}
    expected_final={s["id"] for s in selected["fit_specs"]+selected["blend_specs"]}
    assert {r["candidate"] for r in final}==expected_final and len(final)==len(expected_final)
    for phase,rows in [("validation",validation),("exploratory",final)]:
        for month in {r["eval_month"] for r in rows}:
            month_rows=[r for r in rows if r["eval_month"]==month]
            assert len({r["evaluation_ids_sha256"] for r in month_rows})==1
            baseline=next(r for r in month_rows if r["candidate"]=="baseline_lgbm")
            for r in month_rows:
                assert r["fingerprint"]==selected["fingerprint"]
                assert r["train_month_max"]<month
                assert not any("ConvergenceWarning" in w for w in r["warnings"])
                if r["family"]!="blend":
                    assert sum(r["train_month_counts"].values())==r["train_rows"]
                    assert max(map(int,r["train_month_counts"]))<month
                    if r["candidate"]=="undersample20":
                        assert r["train_fraud"]==baseline["train_fraud"]
                        assert r["train_rows"]==r["train_fraud"]*21
                    else:
                        assert r["train_ids_sha256"]==baseline["train_ids_sha256"]
                for q in config["review_fractions"]:
                    suffix="_at_"+budget_name(q)
                    assert r["reviewed"+suffix]==math.ceil(q*r["n_eval"])
                    assert r["tp"+suffix]+r["fn"+suffix]==r["fraud_eval"]
                    assert r["tp"+suffix]+r["fp"+suffix]==r["reviewed"+suffix]
    # Original baselines must actually reproduce v1, not merely keep its files.
    for phase,rows,old_name in [("validation",validation,"validation"),("exploratory",final,"final")]:
        old=pd.read_csv(Path("reports/baf-study-v1")/f"{old_name}_metrics.csv")
        for r in rows:
            if r["candidate"] not in BASELINE_MAP:
                continue
            model,weighted=BASELINE_MAP[r["candidate"]]
            ref=old[(old.model==model)&(old.weighted==weighted)&(old.strategy=="expanding")&(old.eval_month==r["eval_month"])].iloc[0]
            for metric in ["average_precision","roc_auc","recall_at_1pct","precision_at_1pct","default_f1"]:
                np.testing.assert_allclose(r[metric],ref[metric],rtol=0,atol=1e-10)
    crossfit=pd.read_csv(out/"crossfit_policies.csv",dtype={"threshold_source_months":str})
    for name in names:
        for month in config["validation_months"][1:]:
            earlier=[m for m in config["validation_months"] if m<month]
            thresholds=policies(config,pooled(config,name,earlier))
            data=prediction(config,"validation",month,name)
            for policy,threshold in thresholds.items():
                row=crossfit[(crossfit.candidate==name)&(crossfit.eval_month==month)&(crossfit.policy==policy)].iloc[0]
                assert row.threshold_source_months==",".join(map(str,earlier))
                actual=threshold_stats(data["labels"],data["scores"],threshold)
                for key in ["threshold","f1","f2","tp","fp","fn","tn"]:
                    np.testing.assert_allclose(row[key],actual[key],atol=1e-10)
    policy_table=pd.read_csv(out/"exploratory_policies.csv")
    assert len(policy_table)==len(selected["evaluate_names"])*4
    for name in selected["evaluate_names"]:
        np.testing.assert_allclose(list(selected["thresholds"][name].values()),list(policies(config,pooled(config,name,config["validation_months"])).values()),rtol=0,atol=1e-12)
        data=prediction(config,"exploratory",config["test_month"],name)
        run=next(r for r in final if r["candidate"]==name)
        recomputed=rank_metrics(data["labels"],data["scores"],data["row_ids"],config["review_fractions"])
        for key,value in recomputed.items():
            np.testing.assert_allclose(run[key],value,atol=1e-10)
        for policy,threshold in selected["thresholds"][name].items():
            row=policy_table[(policy_table.candidate==name)&(policy_table.policy==policy)].iloc[0]
            actual=threshold_stats(data["labels"],data["scores"],threshold)
            for key,value in actual.items():
                np.testing.assert_allclose(row[key],value,atol=1e-10)
    assert datetime.fromisoformat(selected["exploratory_access_at_utc"])>=datetime.fromisoformat(selected["created_at_utc"])
    intervals=pd.read_csv(out/"paired_intervals.csv")
    assert len(intervals)==(len(selected["evaluate_names"])-1)*3
    assert set(intervals.replicates)=={config["bootstrap_replicates"]}
    diagnostic=read_config(diagnostic_path)
    diagnostic_out=Path(diagnostic["output_dir"])
    diagnostic_audit=read_config(diagnostic_out/"audit.json")
    assert Path(diagnostic["main_config"]).resolve()==Path(config_path).resolve()
    assert diagnostic_audit["fingerprint"]=={**fingerprint(config_path),"diagnostic_config_sha256":digest(diagnostic_path),"diagnostic_source_sha256":digest("baf/logistic_ablation.py")}
    assert diagnostic_audit["fits"]==3 and diagnostic_audit["adaptive_development_only"]
    assert diagnostic["candidate"]["id"] not in selected["evaluate_names"]
    guard_preservation(config)
    checks=dict(base_validation_fits=51,adaptive_logistic_diagnostic_fits=3,validation_blend_comparisons=9,
                exploratory_base_fits=len(selected["fit_specs"]),exploratory_blends=len(selected["blend_specs"]),
                selected_comparators=len(selected["evaluate_names"]),all_run_identities_match=True,
                temporal_boundaries_and_same_cases=True,training_sampling_counts_pass=True,
                baseline_metrics_match_original_study=True,original_files_and_selection_preserved=True,
                crossfit_threshold_sources_and_metrics_verified=True,frozen_thresholds_recomputed_from_development=True,
                exploratory_metrics_recomputed_from_predictions=True,freeze_precedes_exploratory_access=True,
                convergence_warnings=0,month7_previously_consumed=True)
    write_json(out/"audit.json",checks)
    return config,selected,checks


def generate(config_path,diagnostic_path="configs/logistic-ablation.json"):
    config,selected,checks=audit(config_path,diagnostic_path)
    out=Path(config["output_dir"])
    dest=Path("reports")/out.name
    dest.mkdir(parents=True,exist_ok=True)
    for name in ["validation_metrics.csv","validation_summary.csv","crossfit_policies.csv","exploratory_metrics.csv",
                 "exploratory_policies.csv","paired_intervals.csv","selection.json","audit.json","blends.json","original_local_snapshot.json"]:
        shutil.copy2(out/name,dest/name)
    validation=pd.read_csv(out/"validation_metrics.csv")
    summary=pd.read_csv(out/"validation_summary.csv").set_index("candidate")
    final=pd.read_csv(out/"exploratory_metrics.csv").set_index("candidate")
    policies_table=pd.read_csv(out/"exploratory_policies.csv")
    intervals=pd.read_csv(out/"paired_intervals.csv")
    ablations=[]
    for a,b,label in CONTRASTS:
        delta=validation[validation.candidate==a].set_index("eval_month").average_precision-validation[validation.candidate==b].set_index("eval_month").average_precision
        ablations.append(dict(strategy=label,candidate=a,reference=b,ap_difference=summary.loc[a,"average_precision"]-summary.loc[b,"average_precision"],
                              recall_1pct_difference=summary.loc[a,"recall_at_1pct"]-summary.loc[b,"recall_at_1pct"],
                              crossfit_f1_difference=summary.loc[a,"crossfit_f1"]-summary.loc[b,"crossfit_f1"],
                              months_ap_better=int((delta>0).sum()),min_month_ap_difference=delta.min(),max_month_ap_difference=delta.max()))
    ablations=pd.DataFrame(ablations)
    ablations.to_csv(dest/"ablations.csv",index=False)
    logistic_source=Path(read_config(diagnostic_path)["output_dir"])
    logistic_summary=pd.read_csv(logistic_source/"summary.csv")
    shutil.copy2(logistic_source/"summary.csv",dest/"logistic_fixed_c_summary.csv")
    shutil.copy2(logistic_source/"comparisons.csv",dest/"logistic_fixed_c_months.csv")
    logistic_diag=read_config(diagnostic_path)
    diagnostic_cf=crossfit_policies({**config,"output_dir":str(logistic_source)},[logistic_diag["candidate"]["id"]])
    main_cf=pd.read_csv(out/"crossfit_policies.csv")
    logistic_cf=pd.concat([diagnostic_cf,main_cf[main_cf.candidate.isin(["baseline_logistic","logistic_engineered"])]],ignore_index=True)
    logistic_cf.to_csv(dest/"logistic_fixed_c_thresholds.csv",index=False)
    logistic_f1=logistic_cf[logistic_cf.policy=="f1"].groupby("candidate",as_index=False)[["precision","recall","f1","f2","flagged_fraction"]].mean()
    logistic_f1.to_csv(dest/"logistic_fixed_c_f1_summary.csv",index=False)
    feature_base=features(pd.read_csv(config["data_path"],nrows=2))
    feature_engineered=engineer(feature_base)
    pd.DataFrame({"feature":feature_engineered.columns,"engineered":[c not in feature_base for c in feature_engineered]}).to_csv(dest/"feature_inventory.csv",index=False)
    f1=policies_table[policies_table.policy=="f1"].set_index("candidate")
    f2=policies_table[policies_table.policy=="f2"].set_index("candidate")
    display=final[["average_precision","roc_auc","recall_at_1pct","precision_at_1pct","tp_at_1pct","fp_at_1pct","fn_at_1pct"]].copy()
    for metric in ["precision","recall","f1","f2","mcc","balanced_accuracy","flagged_fraction"]:
        display[f"f1_policy_{metric}"]=f1[metric]
        display[f"f2_policy_{metric}"]=f2[metric]
    display.to_csv(dest/"comparison_summary.csv")
    selected_names=sorted(set(selected["selected"].values()))
    plotted=list(dict.fromkeys(["baseline_lgbm",selected["selected"]["average_precision"],selected["selected"]["crossfit_f1"],selected["selected"]["recall_at_1pct"],selected["selected"]["crossfit_f2"]]))
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(10,7))
    chart=summary.sort_values("average_precision")
    colors=["#bd5c35" if n in selected_names else "#366589" for n in chart.index]
    ax.barh(chart.index,chart.average_precision,color=colors)
    ax.set(xlabel="Average precision (fraction)",title="Mean forward-validation AP across months 4-6\nOrange: selected for at least one objective")
    ax.set_xlim(0,max(chart.average_precision)*1.12)
    for i,v in enumerate(chart.average_precision): ax.text(v+.001,i,f"{v:.3f}",va="center",fontsize=8)
    fig.tight_layout();fig.savefig(dest/"validation_candidates.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    chart=ablations.sort_values("ap_difference")
    ax.barh(chart.strategy,chart.ap_difference*100,color=["#366589" if v>=0 else "#bd5c35" for v in chart.ap_difference])
    ax.axvline(0,color="#444444",lw=1)
    ax.set(xlabel="Change in mean validation AP (percentage points)",title="Ablations against their matching reference presets")
    fig.tight_layout();fig.savefig(dest/"ablations.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for name in plotted:
        data=prediction(config,"exploratory",config["test_month"],name)
        p,r,_=precision_recall_curve(data["labels"],data["scores"])
        axes[0].plot(r,p,label=f"{name} (AP={final.loc[name,'average_precision']:.3f})")
        row=final.loc[name]
        axes[1].plot(np.array(config["review_fractions"])*100,[row[f"recall_at_{budget_name(q)}"]*100 for q in config["review_fractions"]],marker="o",label=name)
    axes[0].set(xlabel="Recall",ylabel="Precision",title="Exploratory month-7 precision-recall curves")
    axes[1].set(xlabel="Review capacity (%)",ylabel="Fraud captured (%)",title="Same cases and review capacities")
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/"operating_curves.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    markers={"f1":"o","f2":"s","precision30":"^","recall50":"D"}
    for name in plotted:
        rows=policies_table[policies_table.candidate==name].sort_values("recall")
        line=axes[0].plot(rows.recall*100,rows.precision*100,alpha=.35,label=name)[0]
        for row in rows.itertuples(): axes[0].scatter(row.recall*100,row.precision*100,marker=markers[row.policy],color=line.get_color(),s=60)
    cf=pd.read_csv(out/"crossfit_policies.csv")
    for name in plotted:
        rows=cf[(cf.candidate==name)&(cf.policy=="f1")].sort_values("eval_month")
        axes[1].plot(rows.eval_month,rows.f1,marker="o",label=name)
    axes[0].set(xlabel="Recall (%)",ylabel="Precision (%)",title="Frozen policies: circle F1, square F2, triangle P30, diamond R50")
    axes[1].set(xlabel="Forward validation month",ylabel="F1 (earlier-month threshold)",title="Threshold policy transfer across validation months",xticks=[5,6])
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/"threshold_tradeoffs.png",dpi=160);plt.close(fig)
    # Final selected model importances are diagnostics, not evidence for causality.
    for spec in selected["fit_specs"]:
        bundle=joblib.load(out/"models"/f"{spec['id']}.joblib")
        model,adapter=bundle["model"],bundle["adapter"]
        if hasattr(model,"named_steps"):
            names=model.named_steps["preprocess"].get_feature_names_out()
            fitted=model.named_steps["model"]
            values=fitted.coef_[0] if hasattr(fitted,"coef_") else fitted.feature_importances_
            label="coefficient" if hasattr(fitted,"coef_") else "split_count"
        else:
            names=adapter.columns
            values=model.feature_importances_
            label="prediction_value_change" if spec["family"]=="catboost" else "split_count"
        pd.DataFrame({"feature":names,label:values}).to_csv(dest/f"{spec['id']}_features.csv",index=False)
    base=final.loc["baseline_lgbm"]
    best_ap=selected["selected"]["average_precision"]
    best_f1=selected["selected"]["crossfit_f1"]
    best_f2=selected["selected"]["crossfit_f2"]
    best_capacity=selected["selected"]["recall_at_1pct"]
    selected_table=pd.DataFrame([dict(objective=k,candidate=v) for k,v in selected["selected"].items()])
    capacity_delta=final.loc[best_capacity,"recall_at_1pct"]-base.recall_at_1pct
    f1_delta=f1.loc[best_f1,"f1"]-f1.loc["baseline_lgbm","f1"]
    headlines=[
        f"For logistic regression, the engineered/regularized preset improves mean forward AP from **{summary.loc['baseline_logistic','average_precision']:.4f}** to **{summary.loc['logistic_engineered','average_precision']:.4f}**, mean precision at 1% review from **{summary.loc['baseline_logistic','precision_at_1pct']*100:.2f}%** to **{summary.loc['logistic_engineered','precision_at_1pct']*100:.2f}%**, and mean recall from **{summary.loc['baseline_logistic','recall_at_1pct']*100:.2f}%** to **{summary.loc['logistic_engineered','recall_at_1pct']*100:.2f}%** across months 4-6. Cross-fitted F1 across months 5/6 changes from **{summary.loc['baseline_logistic','crossfit_f1']:.4f}** to **{summary.loc['logistic_engineered','crossfit_f1']:.4f}**. The fixed-C diagnostic below separates the feature effect from regularization.",
        f"The validation-selected AP candidate **{best_ap}** has month-7 AP **{final.loc[best_ap,'average_precision']:.4f}**, versus **{base.average_precision:.4f}** for unchanged unweighted LightGBM.",
        f"The validation-selected capacity candidate **{best_capacity}** captures **{final.loc[best_capacity,'recall_at_1pct']*100:.2f}%** of fraud at 1% review, versus **{base.recall_at_1pct*100:.2f}%** ({capacity_delta*100:+.2f} pp). Precision is **{final.loc[best_capacity,'precision_at_1pct']*100:.2f}%**, versus **{base.precision_at_1pct*100:.2f}%**. Both are reviewed at the same quota, so any capture gain improves both precision and recall at that quota.",
        f"The cross-fitted validation F1 candidate **{best_f1}** achieves month-7 F1 **{f1.loc[best_f1,'f1']:.4f}**, versus **{f1.loc['baseline_lgbm','f1']:.4f}** for the baseline under its own development-selected F1 policy ({f1_delta:+.4f}). Its precision/recall are **{f1.loc[best_f1,'precision']*100:.2f}% / {f1.loc[best_f1,'recall']*100:.2f}%**; baseline precision/recall are **{f1.loc['baseline_lgbm','precision']*100:.2f}% / {f1.loc['baseline_lgbm','recall']*100:.2f}%**.",
        f"The recall-focused validation F2 candidate **{best_f2}** achieves month-7 F2 **{f2.loc[best_f2,'f2']:.4f}**, precision **{f2.loc[best_f2,'precision']*100:.2f}%** and recall **{f2.loc[best_f2,'recall']*100:.2f}%**, flagging **{f2.loc[best_f2,'flagged_fraction']*100:.2f}%** of applications. The baseline's own F2 policy scores **{f2.loc['baseline_lgbm','f2']:.4f}** and flags **{f2.loc['baseline_lgbm','flagged_fraction']*100:.2f}%**. This policy has a different review load from the 1% capacity comparison.",
    ]
    main_intervals=intervals[intervals.candidate.isin(selected_names)]
    positive=main_intervals[main_intervals.ci_low>0]
    interval_sentence=(f"{len(positive)} selected-candidate metric {'interval lies' if len(positive)==1 else 'intervals lie'} above zero under this conditional bootstrap; see the full table for which comparisons. This is descriptive uncertainty, not fresh confirmation or a search-adjusted significance test."
                       if len(positive) else "No selected-candidate metric interval lies wholly above zero under this conditional bootstrap. Point improvements should therefore be described cautiously.")
    target_view=policies_table[policies_table.policy.isin(["precision30","recall50"])].copy()
    target_view["target_met"]=np.where(target_view.policy=="precision30",target_view.precision_target_met,target_view.recall_target_met)
    failures=target_view[~target_view.target_met]
    top_positive=ablations.sort_values("ap_difference",ascending=False).iloc[0]
    worst=ablations.sort_values("ap_difference").iloc[0]
    lines=[
        "# BAF: exploratory strategies, feature engineering and operating metrics",
        f"Report generated on {datetime.now().astimezone().strftime('%d %B %Y')}. Codex assisted with design, implementation, execution and interpretation. Group review is required before submission.",
        "## Main findings",
        *headlines,interval_sentence,
        "**Evidence status:** Month 7 was already evaluated in study v1. These new comparisons are exploratory even though their new selections and thresholds were frozen before this run's month-7 scoring. Search over development folds, a single synthetic variant and one consumed future month prevent a confirmatory generalization claim.",
        "## What was tested",
        "Seventeen presets produced 51 forward fits (months 4/5/6). Three blends contributed nine additional comparisons. The search includes model capacity, native categorical handling, 36 row-local engineered features, moderate positive weights, recency weighting, random undersampling with prior correction, logistic transformations and CatBoost. Original baselines were rerun and their metrics match study v1. Read the [protocol](../../docs/improvement-protocol.md) and [exact configuration](../../configs/improvements.json).",
        "Metric tables use fractions (0.30 means 30%) unless marked pp. AP is non-interpolated average precision. F1 balances precision/recall; F2 emphasizes recall. MCC and balanced accuracy add class-aware context. Fixed review capacity gives a separate operating constraint. Accuracy and ROC AUC alone can obscure poor minority-class detection. Brier scores are descriptive, especially for weighted models whose outputs are not calibrated probabilities.",
        md_table(selected_table),
        "## Forward validation and strategy ablations",
        "Rank metrics average the three forward months equally. Cross-fitted F1/F2 use month 5 thresholds learned from month 4, and month 6 thresholds learned from months 4/5; neither threshold is selected on its evaluation month's labels. Final policy thresholds pool development predictions only. Choosing model presets and blend members with these same development folds can still be optimistic.",
        "Recency weights can also change effective class composition, so their contrast evaluates the practical strategy rather than a pure recency mechanism. Fit times are measured elapsed times under the current host load/power state; they are descriptive and not a controlled speed benchmark.",
        "Blends average raw prediction scores. A weighted member can have a different score scale from an unweighted member, so the nominal blend weights do not imply calibrated fraud probabilities.",
        "A blend's recorded fit_seconds is zero because it adds no new base-model fit; training it from scratch still requires both member models. Their fit times are shown separately, and shared members are reused across blend weights.",
        md_table(summary.reset_index()),"![Forward candidates](validation_candidates.png)",
        f"The largest positive mean-AP ablation is **{top_positive.strategy}** ({top_positive.ap_difference*100:+.2f} pp); the lowest is **{worst.strategy}** ({worst.ap_difference*100:+.2f} pp). These are matched preset contrasts, not claims that every feature or method helps. Native-small changes both missingness handling and categorical encoding; logistic engineering also changes regularization. Model-capacity presets change several hyperparameters together.",
        md_table(ablations),"![Ablation effects](ablations.png)",
        "An additional adaptive development-only diagnostic was added after the first-fold engineered-logistic gain. It fits raw logistic at C=0.1 on the same three folds, isolating feature effects at fixed C=0.1 from the original C=1-to-C=0.1 regularization change. It was excluded from the candidate competition and never evaluated on month 7. Differences below average identical forward months; this diagnostic is not a predeclared search winner.",
        md_table(logistic_summary),
        "The corresponding F1-policy comparison below uses thresholds learned on earlier months for every version, and reports macro means across months 5/6. This isolates feature effects at C=0.1 for threshold transfer as well as ranking; it remains an adaptive development diagnostic.",
        md_table(logistic_f1),
        "## Consumed month 7: ranking at equal capacity",
        f"The evaluation contains {int(base.n_eval):,} applications and {int(base.fraud_eval):,} fraud cases. At 1% capacity, {int(base.reviewed_at_1pct):,} cases are reviewed. Review rounding and row-ID tie breaking are identical across models. Candidate identities below were selected from validation, not by the highest month-7 result.",
        f"An always-legitimate prediction would achieve **{(1-base.fraud_eval/base.n_eval)*100:.2f}% accuracy** here while detecting zero fraud (recall and F1 both zero). This is why accuracy is supporting context rather than the selection objective. Precision measures how many flagged cases are fraud; recall measures how much of all fraud is caught; F1 is their harmonic mean. Different threshold policies can trade precision against recall even when the model's AP is unchanged.",
        md_table(display.reset_index()[["candidate","average_precision","roc_auc","recall_at_1pct","precision_at_1pct","tp_at_1pct","fp_at_1pct","fn_at_1pct"]]),
        "![Precision-recall and capacity curves](operating_curves.png)",
        "## F1, F2 and operating thresholds",
        "These policies compare every model with a threshold learned from development, including the baseline. A gain over threshold 0.5 alone is not credited as a model improvement. Optimizing F2 or targeting 50% recall can increase flags and false positives substantially; this is a different review policy from fixed 1% capacity.",
        md_table(policies_table[["candidate","policy","threshold","precision","recall","f1","f2","mcc","balanced_accuracy","flagged_fraction","tp","fp","fn","tn"]]),
        "![Threshold trade-offs and transfer](threshold_tradeoffs.png)",
        f"There are **{len(failures)} out-of-time precision/recall target violations** among the constraint policies. Targets refer to validation; they are not guaranteed on future applications. The table checks the corresponding target separately for each policy.",
        md_table(target_view[["candidate","policy","precision","recall","flagged_fraction","target_met"]]),
        "## Paired uncertainty",
        "Intervals use 500 paired row-bootstrap replicates on identical month-7 cases, conditional on fixed models and fixed development-selected thresholds. They recompute capacity selection and tie-aware AP, but exclude model selection, retraining and between-month uncertainty and have no multiplicity adjustment. Known holdout outcomes remain a limitation regardless of interval width.",
        md_table(main_intervals[["candidate","baseline","metric","difference","ci_low","ci_high"]]),
        "## Artifacts and checks",
        md_table(pd.DataFrame([checks])),
        "Twenty-five methodological tests cover the original sampling/metric/freeze invariants plus engineered row independence, training-only categories/medians, unknown categories, undersampling, recency/prior correction, threshold objectives, metric agreement, tied-score weighted AP, paired-interval identity/fixed-threshold checks, temporal threshold sources, preservation guards and adaptive-diagnostic scope guards. Full-data audits additionally verify all original baselines, all forward/final identities and boundaries, quota accounting, threshold recomputation and final metrics directly from saved predictions.",
        "The original study's source/config/requirements and exported selection hashes are unchanged; its local selection is preserved when present. No raw data or bulk model predictions are exported to git. Feature inventories and selected-model importance/coefficient CSVs are descriptive diagnostics; a bundled feature gain does not identify causal features. CatBoost importance is PredictionValuesChange, LightGBM is split count, and logistic coefficients depend on preprocessing/scaling.",
        "Reproduce with the [README](../../README.md). Small CSVs include per-month validation, cross-fitted policies, all ablations, selected month-7 policies, paired intervals and audits. Full predictions and serialized selected models remain in ignored outputs. The final course report must acknowledge Codex and external software and be explainable by the group. This Markdown study is supporting evidence, not the final 8-12 page submission.",
        "Sources: [BAF datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf), [LightGBM tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html), [CatBoost categorical handling](https://catboost.ai/docs/en/features/categorical-features), [CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/common).",
    ]
    (dest/"findings.md").write_text("\n\n".join(lines)+"\n")
    print("Report saved to",dest/"findings.md",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/improvements.json")
    parser.add_argument("--diagnostic-config",default="configs/logistic-ablation.json")
    args=parser.parse_args()
    generate(args.config,args.diagnostic_config)
