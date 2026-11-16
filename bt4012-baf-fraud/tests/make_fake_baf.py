"""Generate a FAKE file with the BAF schema, for smoke-testing the pipeline only.

The numbers mean nothing. Never report results from this file.

    python tests/make_fake_baf.py --rows 80000 --out data/fake_baf.csv
    python run_all.py --data data/fake_baf.csv --quick --sample-frac 1
"""
import argparse

import numpy as np
import pandas as pd


def make(rows, seed=0):
    rng = np.random.default_rng(seed)
    month = rng.integers(0, 8, rows)
    df = pd.DataFrame({
        "income": rng.choice(np.round(np.linspace(0.1, 0.9, 9), 1), rows),
        "name_email_similarity": rng.random(rows),
        "prev_address_months_count": np.where(rng.random(rows) < 0.7, -1, rng.integers(5, 300, rows)),
        "current_address_months_count": np.where(rng.random(rows) < 0.01, -1, rng.integers(0, 400, rows)),
        "customer_age": rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90], rows,
                                   p=[.02, .24, .3, .22, .12, .06, .02, .01, .01]),
        "days_since_request": rng.exponential(1.0, rows),
        "intended_balcon_amount": np.where(rng.random(rows) < 0.74, -rng.random(rows) * 10, rng.random(rows) * 100),
        "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"], rows),
        "zip_count_4w": rng.integers(1, 6000, rows),
        "velocity_6h": rng.normal(5600, 3000, rows).clip(0) + 300 * month,   # covariate drift
        "velocity_24h": rng.normal(4800, 1500, rows).clip(0),
        "velocity_4w": rng.normal(4850, 900, rows) - 120 * month,
        "bank_branch_count_8w": rng.integers(0, 2400, rows),
        "date_of_birth_distinct_emails_4w": rng.integers(0, 40, rows),
        "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"], rows),
        "credit_risk_score": rng.normal(130, 70, rows).round(),
        "email_is_free": rng.integers(0, 2, rows),
        "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE", "BF", "BG"], rows),
        "phone_home_valid": rng.integers(0, 2, rows),
        "phone_mobile_valid": rng.integers(0, 2, rows),
        "bank_months_count": np.where(rng.random(rows) < 0.25, -1, rng.integers(1, 32, rows)),
        "has_other_cards": rng.integers(0, 2, rows),
        "proposed_credit_limit": rng.choice([200, 500, 1000, 1500, 2000], rows),
        "foreign_request": (rng.random(rows) < 0.03).astype(int),
        "source": rng.choice(["INTERNET", "TELEAPP"], rows, p=[.99, .01]),
        "session_length_in_minutes": np.where(rng.random(rows) < 0.002, -1, rng.exponential(7, rows)),
        "device_os": rng.choice(["linux", "windows", "macintosh", "x11", "other"], rows),
        "keep_alive_session": rng.integers(0, 2, rows),
        "device_distinct_emails_8w": rng.choice([-1, 0, 1, 2], rows, p=[.001, .03, .94, .029]),
        "device_fraud_count": 0,
        "month": month,
    })
    # fraud depends on a few features; the effect of one of them changes over time (concept drift)
    logit = (-5.0 + 0.012 * (df.credit_risk_score - 130) - 1.5 * df.name_email_similarity
             + 0.9 * (df.housing_status == "BA") + 0.7 * (df.device_os == "windows")
             + 0.6 * df.email_is_free + (0.04 * month) * (df.customer_age >= 50))
    df.insert(0, "fraud_bool", (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=80000)
    ap.add_argument("--out", default="data/fake_baf.csv")
    a = ap.parse_args()
    d = make(a.rows)
    d.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(d):,} rows, fraud rate {d.fraud_bool.mean():.4f}")
