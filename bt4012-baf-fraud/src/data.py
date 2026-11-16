"""Loading, cleaning and splitting the Bank Account Fraud (BAF) data."""
import numpy as np
import pandas as pd

from .config import CATEGORICAL, MISSING_IF_NEGATIVE, MONTH, TARGET

EXPECTED_COLUMNS = {
    TARGET, MONTH, "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "payment_type", "zip_count_4w", "velocity_6h",
    "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "employment_status", "credit_risk_score",
    "email_is_free", "housing_status", "phone_home_valid", "phone_mobile_valid",
    "bank_months_count", "has_other_cards", "proposed_credit_limit",
    "foreign_request", "source", "session_length_in_minutes", "device_os",
    "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count",
}


def load_raw(path, sample_frac=1.0, seed=0):
    """Read a BAF csv. Optionally keep a stratified sample (same share of
    every month x label cell) for quick runs."""
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing BAF columns: {sorted(missing)}")
    if sample_frac < 1.0:
        df = df.groupby([MONTH, TARGET]).sample(frac=sample_frac, random_state=seed)
    return df.reset_index(drop=True)


def clean(df):
    """Return a cleaned copy plus a log of what was changed.

    * negative sentinel values -> NaN (and we record the missing rate)
    * constant columns are dropped (e.g. device_fraud_count is always 0 in Base)
    * categorical columns get a fixed category set so every split encodes alike
    No step uses the label, so none of this can leak target information.
    """
    df = df.copy()
    log = {"missing_rate": {}, "dropped_constant": []}
    for col in MISSING_IF_NEGATIVE:
        if col in df:
            neg = df[col] < 0
            log["missing_rate"][col] = float(neg.mean())
            df[col] = df[col].astype(float)
            df.loc[neg, col] = np.nan
    for col in list(df.columns):
        if col not in (TARGET, MONTH) and df[col].nunique(dropna=False) <= 1:
            log["dropped_constant"].append(col)
            df = df.drop(columns=col)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).astype("category")
    return df, log


def feature_lists(df):
    """Numeric and categorical feature names. `month` is a time index, not a
    feature: a model that used it could not generalise to future months."""
    cats = [c for c in CATEGORICAL if c in df]
    nums = [c for c in df.columns if c not in cats + [TARGET, MONTH]]
    return nums, cats


def months(df, month_list):
    return df[df[MONTH].isin(month_list)]


def xy(df, nums, cats):
    return df[nums + cats], df[TARGET].to_numpy()


def undersample(df, ratio, seed=0):
    """Keep every fraud and `ratio` legitimate applications per fraud."""
    pos = df[df[TARGET] == 1]
    neg = df[df[TARGET] == 0]
    n_neg = min(len(neg), ratio * len(pos))
    return pd.concat([pos, neg.sample(n=n_neg, random_state=seed)]).sample(frac=1, random_state=seed)
