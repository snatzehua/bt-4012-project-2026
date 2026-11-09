"""Adaptive development-only diagnostic separating feature and C effects."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .common import digest, features, load_data, read_config, write_json
from .improvement_metrics import rank_metrics
from .improvements import fingerprint, fit_run, guard_preservation, prediction, records, table_records


def run(path):
    diagnostic=read_config(path)
    main_path=diagnostic["main_config"]
    main=read_config(main_path)
    main_identity=fingerprint(main_path)
    original=records(main,"validation")
    expected={(n,m) for n in ["baseline_logistic","logistic_engineered"] for m in main["validation_months"]}
    comparators=[r for r in original if (r["candidate"],r["eval_month"]) in expected]
    if {(r["candidate"],r["eval_month"]) for r in comparators}!=expected:
        raise ValueError("Complete the main logistic validation before this diagnostic")
    if any(r["fingerprint"]!=main_identity for r in comparators):
        raise ValueError("Main-study comparator identity differs")
    if not diagnostic["adaptive_development_only"] or diagnostic["candidate"]["engineered"]:
        raise ValueError("This diagnostic must remain raw and development-only")
    engineered_spec=next(s for s in main["candidates"] if s["id"]=="logistic_engineered")
    if diagnostic["candidate"]["family"]!="logistic" or diagnostic["candidate"]["C"]!=engineered_spec["C"]:
        raise ValueError("The feature contrast must use the same logistic regularization")
    if diagnostic["candidate"]["id"] in {s["id"] for s in main["candidates"]}:
        raise ValueError("Diagnostic cannot enter the main candidate competition")
    config={**main,"output_dir":diagnostic["output_dir"]}
    identity={**main_identity,"diagnostic_config_sha256":digest(path),"diagnostic_source_sha256":digest(__file__)}
    guard_preservation(config)
    frame=load_data(config)
    frame=frame[frame.month<config["test_month"]]
    base=features(frame)
    for month in config["validation_months"]:
        fit_run(frame,base,None,diagnostic["candidate"],month,"validation",config,identity)
    rows=records(config,"validation")
    if len(rows)!=len(config["validation_months"]) or any(r["fingerprint"]!=identity for r in rows):
        raise ValueError("Incomplete or stale diagnostic")
    comparisons=[]
    for row in rows:
        month=row["eval_month"]
        data=prediction(config,"validation",month,diagnostic["candidate"]["id"])
        for name in ["baseline_logistic","logistic_engineered"]:
            other=prediction(main,"validation",month,name)
            assert np.array_equal(data["row_ids"],other["row_ids"])
            assert np.array_equal(data["labels"],other["labels"])
        recomputed=rank_metrics(data["labels"],data["scores"],data["row_ids"],config["review_fractions"])
        for key,value in recomputed.items(): np.testing.assert_allclose(row[key],value,atol=1e-12)
        raw_c1=next(r for r in comparators if r["candidate"]=="baseline_logistic" and r["eval_month"]==month)
        engineered_c01=next(r for r in comparators if r["candidate"]=="logistic_engineered" and r["eval_month"]==month)
        assert row["train_ids_sha256"]==raw_c1["train_ids_sha256"]==engineered_c01["train_ids_sha256"]
        for metric in ["average_precision","recall_at_1pct","precision_at_1pct","roc_auc"]:
            comparisons.append(dict(eval_month=month,metric=metric,raw_c1=raw_c1[metric],raw_c01=row[metric],engineered_c01=engineered_c01[metric],
                                    regularization_difference=row[metric]-raw_c1[metric],
                                    features_at_fixed_c_difference=engineered_c01[metric]-row[metric]))
    out=Path(config["output_dir"])
    table_records(rows).to_csv(out/"validation_metrics.csv",index=False)
    table=pd.DataFrame(comparisons)
    table.to_csv(out/"comparisons.csv",index=False)
    table.groupby("metric",as_index=False).mean(numeric_only=True).drop(columns="eval_month").to_csv(out/"summary.csv",index=False)
    write_json(out/"audit.json",dict(fingerprint=identity,adaptive_development_only=True,fits=len(rows),
                                    same_cases_and_labels_verified=True,metrics_recomputed=True,test_month_excluded=True,
                                    no_promotion_into_main_selection=True,original_study_preserved=True))
    guard_preservation(config)
    dest=Path("reports")/out.name
    dest.mkdir(parents=True,exist_ok=True)
    for name in ["validation_metrics.csv","comparisons.csv","summary.csv","audit.json"]:
        shutil.copy2(out/name,dest/name)
    print("Development-only logistic diagnostic complete",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/logistic-ablation.json")
    run(parser.parse_args().config)
