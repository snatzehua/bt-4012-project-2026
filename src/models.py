"""Model factories. Every preprocessing step sits inside the model pipeline,
so it is fitted on the training months only (no leakage into validation/test)."""
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler

from .data import undersample, xy
from .config import TARGET
from .features import BIN_COLS, LOG_COLS, N_BINS, ONEHOT_NUMERIC, log1p_safe

MODEL_NAMES = ["logreg", "random_forest", "lightgbm"]


def _preprocessor(nums, cats, scale):
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])


def _logreg_engineered_preprocessor(nums, cats):
    """Logistic-regression-specific feature engineering (see src/features.py).

    log   : skewed counts -> impute (+ missing flag) -> log1p -> standardise
    num   : remaining numeric columns -> impute (+ missing flag) -> standardise
    level : few-valued numerics (age decade, income decile) -> one-hot, so the
            effect of each level can differ (no monotone assumption)
    bin   : quantile bins (edges fitted on training rows) for strong non-linear
            predictors, added next to their linear term in `num`
    cat   : categorical columns -> one-hot
    """
    log_cols = [c for c in LOG_COLS if c in nums]
    lvl_cols = [c for c in ONEHOT_NUMERIC if c in nums]
    bin_cols = [c for c in BIN_COLS if c in nums]
    other = [c for c in nums if c not in log_cols + lvl_cols]
    impute = lambda: SimpleImputer(strategy="median", add_indicator=True)  # noqa: E731
    return ColumnTransformer([
        ("log", Pipeline([("impute", impute()),
                          ("log1p", FunctionTransformer(log1p_safe, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())]), log_cols),
        ("num", Pipeline([("impute", impute()), ("scale", StandardScaler())]), other),
        ("level", OneHotEncoder(handle_unknown="ignore"), lvl_cols),
        ("bin", KBinsDiscretizer(n_bins=N_BINS, encode="onehot", strategy="quantile",
                                 quantile_method="averaged_inverted_cdf", subsample=200_000,
                                 random_state=0), bin_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])


def make_model(name, nums, cats, settings, weighted=False, seed=0, pos_rate=0.01, params=None):
    """Build an unfitted model.

    weighted=True turns on class weighting (the 'class_weight' imbalance strategy).
    params: optional dict of tuned hyperparameters for this model.
    """
    params = dict(params or {})
    if name == "logreg":
        clf = LogisticRegression(C=params.get("C", 1.0), max_iter=2000,
                                 class_weight="balanced" if weighted else None)
        prep = (_logreg_engineered_preprocessor(nums, cats) if settings.feature_set == "engineered"
                else _preprocessor(nums, cats, scale=True))
        return Pipeline([("prep", prep), ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=settings.rf_trees,
            min_samples_leaf=params.get("min_samples_leaf", 20),
            max_features="sqrt", n_jobs=settings.n_jobs, random_state=seed,
            class_weight="balanced_subsample" if weighted else None)
        return Pipeline([("prep", _preprocessor(nums, cats, scale=False)), ("clf", clf)])
    if name == "lightgbm":
        # LightGBM handles NaN and pandas 'category' columns natively.
        return LGBMClassifier(
            n_estimators=settings.lgbm_trees,
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            min_child_samples=params.get("min_child_samples", 100),
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            scale_pos_weight=((1 - pos_rate) / pos_rate) if weighted else 1.0,
            random_state=seed, n_jobs=settings.n_jobs, verbose=-1)
    raise ValueError(f"unknown model {name}")


# Small grids, chosen on the validation month by PR-AUC.
PARAM_GRID = {
    "logreg": [{"C": 0.1}, {"C": 1.0}],
    "random_forest": [{"min_samples_leaf": 10}, {"min_samples_leaf": 50}],
    "lightgbm": [{"num_leaves": 31, "learning_rate": 0.05},
                 {"num_leaves": 63, "learning_rate": 0.05}],
}


def fit_predict(name, train_df, eval_dfs, nums, cats, settings, strategy="none",
                seed=0, params=None):
    """Fit on train_df using an imbalance strategy and score every frame in eval_dfs.

    strategy: 'none' | 'class_weight' | 'undersample'
    Returns (fitted_model, list_of_score_arrays, seconds_to_fit).
    """
    if strategy == "undersample":
        train_df = undersample(train_df, settings.undersample_ratio, seed=seed)
    pos_rate = float(train_df[TARGET].mean())
    model = make_model(name, nums, cats, settings, weighted=(strategy == "class_weight"),
                       seed=seed, pos_rate=pos_rate, params=params)
    X, y = xy(train_df, nums, cats)
    t0 = time.time()
    model.fit(X, y)
    secs = time.time() - t0
    scores = [model.predict_proba(xy(d, nums, cats)[0])[:, 1] for d in eval_dfs]
    return model, scores, secs
