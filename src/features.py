"""Feature engineering.

Two layers, because logistic regression and tree models need different things:

1. Row-wise engineered features (`add_features`), given to EVERY model.
   Each one uses only the values in the same row, never the label and never
   statistics from other rows, so it cannot leak information across time or
   from the target.

2. Model-specific transforms for logistic regression (`LOG_COLS`, `ONEHOT_NUMERIC`,
   `BIN_COLS`), built inside the sklearn pipeline in src/models.py so that anything
   that is *fitted* (quantile bin edges, medians, scaling) is fitted on the training
   months only. Tree models are invariant to monotone transforms (a log does not
   change where a split can go) and already model non-linear effects, so they get
   the raw columns plus the row-wise features from layer 1.
"""
import numpy as np

# ---- Layer 1: row-wise features for all models -----------------------------
# velocity_* are platform-wide application counts over the last 6h / 24h / 4w.
# Their LEVEL drifts strongly from month to month (PSI up to ~6 in E0), but a
# RATIO compares the short-term rate with the longer-term rate at the same moment,
# which (a) captures a sudden burst of applications and (b) cancels out a shared
# level shift, so it should be more stable over time.
ENGINEERED = [
    "velocity_6h_to_24h",      # burst in the last 6 hours relative to the last day
    "velocity_24h_to_4w",      # busy day relative to the last 4 weeks
    "velocity_6h_to_4w",
    "n_missing",               # how many optional fields were left empty
    "has_prev_address",        # previous address given at all (71% are missing)
    "total_address_months",    # total months at current + previous address
    "phones_valid",            # number of valid phone numbers (0, 1 or 2)
]

# BAF columns where a negative value means "missing" (mirrors config.MISSING_IF_NEGATIVE).
_OPTIONAL = [
    "prev_address_months_count", "current_address_months_count", "intended_balcon_amount",
    "bank_months_count", "session_length_in_minutes", "device_distinct_emails_8w",
]


def add_features(df):
    """Return a copy of a CLEANED frame (sentinels already NaN) with ENGINEERED columns."""
    df = df.copy()
    v6 = df["velocity_6h"].clip(lower=0)  # a few small negative values in the raw data
    v24 = df["velocity_24h"].clip(lower=1)  # denominators: never 0 (min is ~1300 in Base.csv)
    v4w = df["velocity_4w"].clip(lower=1)
    df["velocity_6h_to_24h"] = v6 / v24
    df["velocity_24h_to_4w"] = df["velocity_24h"].clip(lower=0) / v4w
    df["velocity_6h_to_4w"] = v6 / v4w
    opt = [c for c in _OPTIONAL if c in df]
    df["n_missing"] = df[opt].isna().sum(axis=1).astype(float)
    df["has_prev_address"] = df["prev_address_months_count"].notna().astype(float)
    df["total_address_months"] = (df["prev_address_months_count"].fillna(0)
                                  + df["current_address_months_count"].fillna(0))
    df["phones_valid"] = (df["phone_home_valid"] + df["phone_mobile_valid"]).astype(float)
    return df


# ---- Layer 2: logistic-regression-only transforms ---------------------------
# Heavily right-skewed, non-negative counts/durations (skew > 1 in Base.csv).
# A linear model would let a handful of extreme values dominate the coefficient;
# log1p compresses the tail so the effect is roughly linear in log scale.
LOG_COLS = [
    "prev_address_months_count", "current_address_months_count", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "bank_branch_count_8w",
    "proposed_credit_limit", "session_length_in_minutes", "total_address_months",
]

# Numeric columns with only a few distinct values whose effect need not be monotone
# (customer_age is in decades 10..90, income in deciles 0.1..0.9): one-hot them so
# logistic regression can learn a separate effect for each level.
ONEHOT_NUMERIC = ["customer_age", "income"]

# Strong continuous predictors where the fraud risk is plausibly non-linear
# (e.g. very low name/e-mail similarity). Quantile bins (edges fitted on the
# training months) are added NEXT TO the linear term.
BIN_COLS = ["name_email_similarity", "credit_risk_score"]
N_BINS = 10


def log1p_safe(x):
    """log1p for a non-negative array (used inside a FunctionTransformer)."""
    return np.log1p(np.clip(x, 0, None))
