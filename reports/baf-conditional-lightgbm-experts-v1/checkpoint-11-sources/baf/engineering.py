"""Row-local features and training-only adapters for exploratory experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import CATEGORICAL, make_model, read_config


CROSSES = ["source_device", "payment_employment", "housing_home_phone"]


def engineer(base):
    """Requires already-decoded predictors; never accepts target/month inputs."""
    if {"fraud_bool", "month", "device_fraud_count"} & set(base.columns):
        raise ValueError("Excluded fields supplied to feature engineering")
    x = base.copy()
    for c in ["velocity_6h", "velocity_24h", "velocity_4w", "zip_count_4w",
              "bank_branch_count_8w", "days_since_request", "session_length_in_minutes",
              "intended_balcon_amount", "proposed_credit_limit", "credit_risk_score"]:
        x[f"signed_log_{c}"] = np.sign(x[c]) * np.log1p(np.abs(x[c]))
    x["velocity_6h_to_24h"] = x.velocity_6h / (x.velocity_24h + 1)
    x["velocity_24h_to_4w"] = x.velocity_24h / (x.velocity_4w + 1)
    x["velocity_6h_to_4w"] = x.velocity_6h / (x.velocity_4w + 1)
    x["velocity_short_change"] = x.velocity_6h - x.velocity_24h
    x["velocity_long_change"] = x.velocity_24h - x.velocity_4w
    x["zip_share_4w"] = x.zip_count_4w / (x.velocity_4w * 24 * 28 + 1)
    x["branch_count_to_global_velocity"] = x.bank_branch_count_8w / (x.velocity_4w + 1)
    x["dob_emails_to_zip_count"] = x.date_of_birth_distinct_emails_4w / (x.zip_count_4w + 1)
    for c in ["prev_address_months_count", "current_address_months_count", "bank_months_count",
              "session_length_in_minutes", "device_distinct_emails_8w", "intended_balcon_amount"]:
        x[f"missing_{c}"] = x[c].isna().astype(np.int8)
    x["address_tenure_total"] = x.prev_address_months_count + x.current_address_months_count
    x["bank_to_current_tenure"] = x.bank_months_count / (x.current_address_months_count + 1)
    x["valid_phone_count"] = x.phone_home_valid + x.phone_mobile_valid
    x["neither_phone_valid"] = (x.valid_phone_count == 0).astype(np.int8)
    x["free_email_name_mismatch"] = x.email_is_free * (1 - x.name_email_similarity)
    x["device_email_name_mismatch"] = x.device_distinct_emails_8w * (1 - x.name_email_similarity)
    x["credit_income_decile_interaction"] = x.proposed_credit_limit * x.income
    x["credit_age_interaction"] = x.proposed_credit_limit * x.customer_age
    x["risk_income_interaction"] = x.credit_risk_score * x.income
    for new, cols in zip(CROSSES, [("source", "device_os"), ("payment_type", "employment_status"), ("housing_status", "phone_home_valid")]):
        x[new] = x[cols[0]].astype(str) + "::" + x[cols[1]].astype(str)
    numeric = x.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Nonfinite engineered feature")
    return x


class NativeInputs:
    """Fit categorical levels / numerical medians only on the training subset."""
    def __init__(self, catboost=False):
        self.catboost = catboost

    def fit(self, x):
        self.columns = list(x.columns)
        self.categoricals = [c for c in x.columns if c in CATEGORICAL + CROSSES]
        self.levels = {c: sorted(x[c].dropna().astype(str).unique()) for c in self.categoricals}
        self.numeric = [c for c in x.columns if c not in self.categoricals]
        self.medians = x[self.numeric].median().fillna(0)
        return self

    def transform(self, x):
        if list(x.columns) != self.columns:
            raise ValueError("Predictor schema differs from training")
        result = x.copy()
        for c in self.categoricals:
            values = x[c].fillna("__MISSING__").astype(str)
            result[c] = (values if self.catboost else
                         pd.Categorical(values.where(values.isin(self.levels[c])), categories=self.levels[c]))
        if self.catboost:
            result[self.numeric] = result[self.numeric].fillna(self.medians)
        return result


def make_estimator(spec, config, x):
    family = spec["family"]
    if family == "baseline":
        return make_model(spec["model"], spec["weighted"], read_config(config["baseline_config"]), x.columns), None
    if family == "logistic":
        cats = [c for c in x.columns if c in CATEGORICAL + CROSSES]
        nums = [c for c in x.columns if c not in cats]
        pre = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler())]), nums),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cats),
        ], sparse_threshold=1)
        model = LogisticRegression(C=spec["C"], max_iter=1500, tol=.0001, random_state=config["seed"])
        return Pipeline([("preprocess", pre), ("model", model)]), None
    adapter = NativeInputs(catboost=family == "catboost").fit(x)
    if family == "lightgbm":
        params = dict(spec["params"])
        params.update(random_state=config["seed"], n_jobs=config["threads"], deterministic=True,
                      force_col_wise=True, verbosity=-1, scale_pos_weight=spec.get("pos_weight", 1))
        return LGBMClassifier(**params), adapter
    if family == "catboost":
        model = CatBoostClassifier(**spec["params"], loss_function="Logloss", random_seed=config["seed"],
                                   thread_count=config["threads"], verbose=False, allow_writing_files=False,
                                   class_weights=[1, spec.get("pos_weight", 1)], cat_features=adapter.categoricals)
        return model, adapter
    raise ValueError(family)


def undersample_indices(labels, eligible, negatives_per_positive, seed):
    eligible = np.asarray(eligible)
    y = np.asarray(labels)[eligible]
    pos, neg = eligible[y == 1], eligible[y == 0]
    count = min(len(neg), int(len(pos) * negatives_per_positive))
    if not len(pos) or not count:
        raise ValueError("Undersampling requires both classes")
    selected = np.random.default_rng(seed).choice(neg, count, replace=False)
    return np.sort(np.concatenate([pos, selected]))


def recency_weights(months, half_life):
    months = np.asarray(months)
    if half_life <= 0:
        raise ValueError("Half-life must be positive")
    weight = 2.0 ** ((months - months.max()) / half_life)
    return weight / weight.mean()


def correct_prior(scores, population_prior, sampled_prior):
    """Undo the prior-odds change induced by random class undersampling."""
    if not 0 < population_prior < 1 or not 0 < sampled_prior < 1:
        raise ValueError("Priors must lie in (0,1)")
    p = np.clip(np.asarray(scores), 1e-12, 1 - 1e-12)
    adjustment = population_prior / (1 - population_prior) / (sampled_prior / (1 - sampled_prior))
    return p * adjustment / (1 - p + p * adjustment)
