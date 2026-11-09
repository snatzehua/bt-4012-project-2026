"""Adaptive F2 extension with preserved prior studies and training provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, features, load_data, read_config, write_json
from .engineering import engineer
from .improvements import fit_run, paths, prediction, save_prediction
from .recall_weighted import fingerprint as previous_fingerprint, guard as prior_guard, ids_digest, verify_record
from .recall_weighted_metrics import POLICIES, aligned, learn, measure, paired_f2
from .recall_extension_metrics import construct, leaves, mean_scores, rank_blend
from .study import fit_run as original_fit, identity as original_identity, run_name, train_indices

SOURCES = ["baf/common.py","baf/engineering.py","baf/improvement_metrics.py","baf/improvements.py","baf/study.py",
           "baf/recall_weighted.py","baf/recall_weighted_metrics.py","baf/recall_extension.py","baf/recall_extension_metrics.py"]


def fingerprint(path):
    result=previous_fingerprint(path)
    result["sources"]={p:digest(p) for p in SOURCES}
    return result


def guard(config):
    prior_guard(config)
    if previous_fingerprint(config["previous_f2_config"])!=config["previous_f2_fingerprint"]:
        raise ValueError("Previous F2 source/config/software changed")
    if original_identity(config["original_config"])!=config["original_identity"]:
        raise ValueError("Original window recipe changed")


def expected_population(frame,month):
    ids=frame.index[frame.month==month].to_numpy()
    return dict(row_ids=ids,labels=frame.loc[ids,"fraud_bool"].to_numpy(),scores=np.zeros(len(ids)))


def expected_training(frame,spec,month,config):
    eligible=frame.index[frame.month<month].to_numpy()
    if spec["family"]=="window":
        old=read_config(config["original_config"])
        chosen=train_indices(frame,eligible,spec["original_spec"],old)
        if spec["original_spec"]["strategy"]=="historical_matched":
            recent={**spec["original_spec"],"strategy":"recent"}
            ref=train_indices(frame,eligible,recent,old)
            if len(chosen)!=len(ref) or frame.loc[chosen,"fraud_bool"].sum()!=frame.loc[ref,"fraud_bool"].sum():
                raise ValueError("Historical control is not size/class matched")
        return chosen
    return eligible


def verify_training(record,chosen,frame,month):
    if record["train_ids_sha256"]!=ids_digest(chosen) or record["train_rows"]!=len(chosen):
        raise ValueError("Training identity mismatch")
    if record["train_fraud"]!=int(frame.loc[chosen,"fraud_bool"].sum()):
        raise ValueError("Training class count mismatch")
    if record["train_month_max"]!=int(frame.loc[chosen,"month"].max()) or record["train_month_max"]>=month:
        raise ValueError("Training time mismatch")


def verify_scores(record,data):
    stats=measure(data,dict(kind="pooled",threshold=.5))
    for metric in ["average_precision","roc_auc"]:
        if not np.isclose(record[metric],stats[metric],atol=1e-12,rtol=1e-10):
            raise ValueError("Cached score metrics mismatch")


def load_npz(path):
    with np.load(path) as z:
        return {k:z[k] for k in ["labels","scores","row_ids"]}


def normalize_original(record,spec,identity):
    original=spec["original_spec"]
    if any(record[k]!=v for k,v in original.items()):
        raise ValueError("Original recipe metadata mismatch")
    return {**record,"candidate":spec["id"],"spec":spec,"fingerprint":identity,
            "train_ids_sha256":record["train_row_ids_sha256"],
            "train_month_max":max(map(int,record["train_months"])),"family":"window"}


def run_base(frame,base,engineered,spec,month,phase,config,identity):
    rp,pp=paths(config,phase,month,spec["id"])
    expected=expected_population(frame,month)
    chosen=expected_training(frame,spec,month,config)
    def verify(record,data):
        verify_record(record,identity,spec,data,expected)
        verify_training(record,chosen,frame,month)
        verify_scores(record,data)
    if rp.exists() and pp.exists():
        record=read_config(rp);data=load_npz(pp);verify(record,data)
        if record["prediction_sha256"]!=digest(pp): raise ValueError("Own cached prediction changed")
        return record
    source_rp=source_pp=None
    if spec["family"]=="window":
        old=read_config(config["original_config"])
        old_phase="final" if phase=="exploratory" else phase
        stem=run_name(old_phase,month,spec["original_spec"])
        source_rp=Path(old["output_dir"])/"runs"/(stem+".json")
        source_pp=Path(old["output_dir"])/"predictions"/(stem+".npz")
        reused=source_rp.exists() and source_pp.exists()
        if not reused:
            # Original fitting recipe; new destination, never the original study.
            refit={**old,"output_dir":str(Path(config["output_dir"])/"original-refits")}
            if Path(refit["output_dir"]).resolve()==Path(old["output_dir"]).resolve():
                raise ValueError("Refit would overwrite old outputs")
            original_fit(frame,base,frame.index[frame.month<month].to_numpy(),expected["row_ids"],
                         spec["original_spec"],month,old_phase,refit,config["original_identity"])
            source_rp=Path(refit["output_dir"])/"runs"/(stem+".json")
            source_pp=Path(refit["output_dir"])/"predictions"/(stem+".npz")
        raw=read_config(source_rp)
        if raw["fingerprint"]!=config["original_identity"]: raise ValueError("Original cache identity changed")
        record=normalize_original(raw,spec,identity)
        data=load_npz(source_pp)
        verify(record,data)
    else:
        reused=False; record=data=None
        for cfg_path,expected_id in [(config["previous_f2_config"],config["previous_f2_fingerprint"]),
                                     (config["prior_config"],config["prior_fingerprint"])]:
            old=read_config(cfg_path)
            if spec not in old["candidates"]: continue
            sr,sp=paths(old,phase,month,spec["id"])
            if not(sr.exists() and sp.exists()): continue
            raw=read_config(sr)
            if raw["fingerprint"]!=expected_id or raw["spec"]!=spec: raise ValueError("Prior base cache changed")
            record={**raw,"fingerprint":identity}; data=load_npz(sp)
            verify(record,data)
            source_rp,source_pp=sr,sp;reused=True;break
        if record is None:
            record=fit_run(frame,base,engineered,spec,month,phase,config,identity)
            data=prediction(config,phase,month,spec["id"])
    save_prediction(pp,data["row_ids"],data["labels"],data["scores"])
    record={**record,"phase":phase,"reused":reused,"prediction_sha256":digest(pp)}
    if source_rp is not None:
        record["provenance"]={"record":str(source_rp),"record_sha256":digest(source_rp),
                              "prediction":str(source_pp),"prediction_sha256":digest(source_pp)}
    write_json(rp,record);verify(record,data)
    print("Verified",rp.stem,"reused" if reused else "new fit",flush=True)
    return record


def base_history(config):
    history={s["id"]:{m:prediction(config,"validation",m,s["id"]) for m in config["validation_months"]} for s in config["candidates"]}
    for spec in config["seed_means"]:
        history[spec["id"]]={m:mean_scores([history[n][m] for n in spec["members"]]) for m in config["validation_months"]}
    return history


def save_derived(config,name,month,data,recipe,identity):
    rp=Path(config["output_dir"])/"derived"/f"validation-m{month}-{name}.json"
    pp=paths(config,"validation",month,name)[1]
    if rp.exists() and pp.exists():
        record=read_config(rp);saved=load_npz(pp)
        if record["fingerprint"]!=identity or record["recipe"]!=recipe or record["prediction_sha256"]!=digest(pp):
            raise ValueError("Derived cache changed")
        aligned(data,saved)
        if not np.array_equal(data["scores"],saved["scores"]): raise ValueError("Derived score mismatch")
        return
    save_prediction(pp,data["row_ids"],data["labels"],data["scores"])
    write_json(rp,dict(candidate=name,eval_month=month,recipe=recipe,fingerprint=identity,prediction_sha256=digest(pp)))


def crossfit(config,history=None):
    history=base_history(config) if history is None else history
    rows=[]
    for month in config["validation_months"][1:]:
        strategies=construct(history,config,month)
        if len(strategies)!=config["expected_strategy_count"]: raise ValueError("Strategy count mismatch")
        for name,(data,past,recipe) in strategies.items():
            aligned(data,history["baseline_balanced"][month])
            for kind in POLICIES:
                policy=learn(past,kind,month)
                rows.append(dict(candidate=name,policy=kind,eval_month=month,
                                 source_months=",".join(map(str,policy["source_months"])),
                                 recipe=json.dumps(recipe,sort_keys=True),parameters=json.dumps(policy,sort_keys=True),**measure(data,policy)))
    table=pd.DataFrame(rows)
    if len(table)!=config["expected_crossfit_rows"]: raise ValueError("Forward grid incomplete")
    return table


def order_summary(cf):
    cols=["f2","f1","f3","precision","recall","flagged_fraction","average_precision","tp","fp","fn"]
    table=cf.groupby(["candidate","policy"],as_index=False)[cols].mean()
    return table.sort_values(["f2","recall","precision","candidate","policy"],ascending=[False,False,False,True,True])


def validate(config_path):
    config,identity=read_config(config_path),fingerprint(config_path);guard(config)
    out=Path(config["output_dir"])
    if (out/"selection.json").exists(): raise ValueError("Frozen study cannot be retuned")
    frame=load_data(config);frame=frame[frame.month<config["test_month"]]
    base=features(frame);engineered=engineer(base)
    for month in config["validation_months"]:
        for spec in config["candidates"]: run_base(frame,base,engineered,spec,month,"validation",config,identity)
    history=base_history(config)
    for spec in config["seed_means"]:
        for month in config["validation_months"]:
            save_derived(config,spec["id"],month,history[spec["id"]][month],spec,identity)
    cf=crossfit(config,history);summary=order_summary(cf)
    for month in config["validation_months"][1:]:
        for name,(data,past,recipe) in construct(history,config,month).items():
            if recipe["kind"]=="rank_blend": save_derived(config,name,month,data,recipe,identity)
    old=pd.read_csv("reports/baf-recall-weighted-v1/summary.csv").set_index(["candidate","policy"])
    for name in ["baseline_lgbm","baseline_balanced"]:
        actual=summary[(summary.candidate==name)&(summary.policy=="pooled")].iloc[0].f2
        if not np.isclose(actual,old.loc[(name,"pooled"),"f2"],atol=1e-12): raise ValueError("Original F2 baseline mismatch")
    cf.to_csv(out/"crossfit.csv",index=False);summary.to_csv(out/"summary.csv",index=False)
    write_json(out/"validation_complete.json",dict(fingerprint=identity,crossfit_sha256=digest(out/"crossfit.csv"),summary_sha256=digest(out/"summary.csv")))
    guard(config);print(summary.head(10).to_string(index=False),flush=True)


def audit_bases(config,identity,frame,phase="validation",specs=None):
    specs=config["candidates"] if specs is None else specs
    months=config["validation_months"] if phase=="validation" else [config["test_month"]]
    rows=[]
    for month in months:
        expected=expected_population(frame,month)
        for spec in specs:
            rp,pp=paths(config,phase,month,spec["id"]);record=read_config(rp);data=load_npz(pp)
            verify_record(record,identity,spec,data,expected)
            verify_training(record,expected_training(frame,spec,month,config),frame,month);verify_scores(record,data)
            if digest(pp)!=record["prediction_sha256"]: raise ValueError("Saved score hash mismatch")
            if record["warnings"]: raise ValueError("Inspect fit warnings")
            if "provenance" in record:
                for key in ["record","prediction"]:
                    if digest(record["provenance"][key])!=record["provenance"][key+"_sha256"]: raise ValueError("Reuse source changed")
            rows.append(record)
    return rows


def freeze(config_path):
    config,identity=read_config(config_path),fingerprint(config_path);guard(config)
    out=Path(config["output_dir"])
    if (out/"selection.json").exists(): raise ValueError("Selection is immutable")
    complete=read_config(out/"validation_complete.json")
    if complete["fingerprint"]!=identity or any(complete[k+"_sha256"]!=digest(out/(k+".csv")) for k in ["crossfit","summary"]):
        raise ValueError("Stale development outputs")
    frame=load_data(config);frame=frame[frame.month<config["test_month"]]
    audit_bases(config,identity,frame)
    history=base_history(config);cf=crossfit(config,history);stored=pd.read_csv(out/"crossfit.csv")
    if not np.allclose(cf.f2,stored.f2): raise ValueError("Forward metric mismatch")
    winner=order_summary(cf).iloc[0]
    # No month-7 predictions are needed to select dependencies or policies.
    dummy={name:{**items,config["test_month"]:items[config["validation_months"][-1]]} for name,items in history.items()}
    strategies=construct(dummy,config,config["test_month"])
    names=sorted({str(winner.candidate),"baseline_balanced","baseline_lgbm"})
    recipes={name:strategies[name][2] for name in names}
    means={s["id"]:s for s in config["seed_means"]}
    rules={name:{kind:learn(strategies[name][1],kind,config["test_month"]) for kind in POLICIES} for name in names}
    needed=set()
    for name in names: needed|=leaves(name,config,recipes)
    selected=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),fingerprint=identity,config=config,
                  selected_candidate=str(winner.candidate),selected_policy=str(winner.policy),selected_development_f2=float(winner.f2),
                  evaluate_names=names,recipes=recipes,rules=rules,seed_means=means,
                  fit_specs=[s for s in config["candidates"] if s["id"] in needed],month7_previously_consumed=True,exploratory=True)
    write_json(out/"selection.json",selected)
    print("Frozen",winner.candidate,winner.policy,winner.f2,flush=True)


def load_selection(config_path):
    config=read_config(config_path);guard(config)
    selected=read_config(Path(config["output_dir"])/"selection.json")
    if selected["fingerprint"]!=fingerprint(config_path) or selected["config"]!=config: raise ValueError("Frozen identity changed")
    return config,selected


def selected_data(config,selected,data):
    def resolve(name):
        if name in data: return data[name]
        if name in selected["seed_means"]:
            result=mean_scores([resolve(n) for n in selected["seed_means"][name]["members"]])
        else:
            recipe=selected["recipes"][name]
            result=rank_blend(resolve(recipe["anchor"]),resolve(recipe["member"]),recipe["weight"])
        data[name]=result;return result
    return {name:resolve(name) for name in selected["evaluate_names"]}


def exploratory(config_path):
    config,selected=load_selection(config_path);out=Path(config["output_dir"])
    access=out/"exploratory_access.json"
    if not access.exists(): write_json(access,dict(access_at_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=digest(out/"selection.json"),month7_previously_consumed=True))
    elif read_config(access)["selection_sha256"]!=digest(out/"selection.json"): raise ValueError("Post-access selection changed")
    frame=load_data(config);base=features(frame);engineered=engineer(base);month=config["test_month"]
    for spec in selected["fit_specs"]: run_base(frame,base,engineered,spec,month,"exploratory",config,selected["fingerprint"])
    raw={s["id"]:prediction(config,"exploratory",month,s["id"]) for s in selected["fit_specs"]}
    data=selected_data(config,selected,raw);rows=[];paired_data={};paired_rules={}
    base_names={s["id"] for s in selected["fit_specs"]}
    for name,d in data.items():
        aligned(d,data["baseline_balanced"])
        if name not in base_names:
            save_prediction(paths(config,"exploratory",month,name)[1],d["row_ids"],d["labels"],d["scores"])
        # Base predictions already have a frozen hash in their run record.
        for kind,rule in selected["rules"][name].items():
            rows.append(dict(candidate=name,policy=kind,selected=name==selected["selected_candidate"] and kind==selected["selected_policy"],
                             parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
            key=name+"__"+kind;paired_data[key]=d;paired_rules[key]=rule
    pd.DataFrame(rows).to_csv(out/"exploratory.csv",index=False)
    intervals=paired_f2(paired_data,paired_rules,"baseline_balanced__pooled",config["bootstrap_replicates"],config["bootstrap_seed"])
    pd.DataFrame(intervals).to_csv(out/"paired_intervals.csv",index=False)
    guard(config);print(pd.DataFrame(rows)[["candidate","policy","f2","precision","recall","flagged_fraction","tp","fp","fn"]].to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["validate","freeze","exploratory"])
    parser.add_argument("--config",default="configs/recall-weighted-extension.json");args=parser.parse_args()
    globals()[args.command](args.config)
