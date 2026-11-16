"""Run the whole project end to end.

    python run_all.py --data data/Base.csv              # full run
    python run_all.py --data data/Base.csv --quick      # ~10% sample, 1 seed, smaller models
    python run_all.py --data data/Base.csv --only e3    # one experiment (E1 must have run first)

Results go to results/<dataset>/, figures to figures/<dataset>/ (e.g. results/Base/).
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from experiments import (e0_eda_drift, e1_model_comparison, e2_random_vs_temporal,
                         e3_training_window, e4_imbalance, e5_operations, e6_interpretability)
from src.config import Settings, TARGET
from src.data import clean, feature_lists, load_raw
from src.utils import log, save_json

EXPERIMENTS = {
    "e0": e0_eda_drift, "e1": e1_model_comparison, "e2": e2_random_vs_temporal,
    "e3": e3_training_window, "e4": e4_imbalance, "e5": e5_operations, "e6": e6_interpretability,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Settings().data_path)
    ap.add_argument("--quick", action="store_true", help="fast smoke-test settings")
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=None)
    args = ap.parse_args()

    S = Settings(data_path=args.data)
    # Keep each dataset's outputs apart, e.g. results/Base/ and results/Variant_I/
    tag = args.data.stem.replace(" ", "_")
    S.results_dir = S.results_dir / tag
    S.fig_dir = S.fig_dir / tag
    if args.quick:
        S.quick()
    if args.sample_frac is not None:
        S.sample_frac = args.sample_frac
    if args.seeds is not None:
        S.seeds = args.seeds
    S.ensure_dirs()

    log(f"Loading {S.data_path} (sample_frac={S.sample_frac})")
    raw = load_raw(S.data_path, S.sample_frac, seed=0)
    df, clean_log = clean(raw)
    nums, cats = feature_lists(df)
    log(f"{len(df):,} rows, {len(nums)} numeric + {len(cats)} categorical features, "
        f"fraud rate {df[TARGET].mean():.4f}")

    save_json({"settings": vars(S), "rows": len(df), "fraud_rate": float(df[TARGET].mean()),
               "cleaning": clean_log, "numeric_features": nums, "categorical_features": cats,
               "python": sys.version, "platform": platform.platform(),
               "versions": {"pandas": pd.__version__, "numpy": np.__version__,
                            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__}},
              S.results_dir / "run_info.json")

    timings = {}
    for key in args.only or list(EXPERIMENTS):
        t0 = time.time()
        EXPERIMENTS[key].run(df, nums, cats, S)
        timings[key] = round(time.time() - t0, 1)
    save_json(timings, S.results_dir / "timings.json")
    log(f"All done: {timings}")


if __name__ == "__main__":
    main()
