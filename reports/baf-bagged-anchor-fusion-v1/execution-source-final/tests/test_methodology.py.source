import json
import math

import numpy as np
import pandas as pd
import pytest

from baf.common import (
    CATEGORICAL, SENTINEL_MINUS_ONE, bootstrap_recall, budget_metrics,
    features, make_model, matched_indices, ranking, read_config, recent_indices,
)
from baf.study import final


def test_recent_sampling_honors_cutoff_and_month_priority():
    months = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    eligible = np.flatnonzero(months < 3)
    recent = recent_indices(months, eligible, 3, 2026)
    assert set([4, 5]).issubset(recent)
    assert months[recent].max() == 2
    assert months[recent].min() == 1
    assert np.array_equal(recent, recent_indices(months, eligible, 3, 2026))
    with pytest.raises(ValueError, match="Infeasible"):
        recent_indices(months, eligible, 7, 2026)


def test_historical_controls_match_classes_without_future_rows_or_replacement():
    y = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
    eligible = np.arange(6)
    recent = np.array([3, 4, 5])
    chosen = matched_indices(y, eligible, recent, 17)
    assert len(chosen) == 3 and len(np.unique(chosen)) == 3
    assert y[chosen].sum() == y[recent].sum() == 1
    assert set(chosen).issubset(eligible)
    with pytest.raises(ValueError, match="ineligible"):
        matched_indices(y, eligible, np.array([3, 7, 8]), 17)


def test_capacity_rounding_ties_and_confusion_accounting():
    y = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.9, 0.8, 0.2, 0.1])
    ids = np.array([12, 10, 14, 16, 18])
    metrics = budget_metrics(y, scores, ids, 0.21)
    assert metrics["reviewed"] == 2
    assert metrics["tp"] == 1 and metrics["fp"] == 1
    assert metrics["fn"] == 1 and metrics["tn"] == 2
    assert budget_metrics(y, scores, ids, 0.2)["tp"] == 0  # smaller ID wins a score tie
    assert sum(metrics[k] for k in ["tp", "fp", "fn", "tn"]) == len(y)


def test_weighted_bootstrap_equals_explicit_resampling_and_reranking():
    y = np.array([0, 1, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.8, 0.5, 0.2])
    ids = np.array([10, 11, 12, 13, 14])
    sampled = np.array([0, 1, 1, 2, 4, 4, 4])
    weights = np.bincount(sampled, minlength=len(y))
    order = ranking(scores, ids)
    for q in [0.01, 0.3, 0.7, 1.0]:
        explicit = budget_metrics(y[sampled], scores[sampled], ids[sampled], q)["recall"]
        assert bootstrap_recall(y, order, weights, q) == pytest.approx(explicit)


def fixture_frame():
    frame = pd.DataFrame({c: [0., 1., 2., 3., 4., 5.] for c in SENTINEL_MINUS_ONE})
    for c in CATEGORICAL:
        frame[c] = ["a", "a", "b", "b", "a", "b"]
    frame["fraud_bool"] = [0, 1, 0, 1, 0, 1]
    frame["month"] = [0] * 6
    frame["device_fraud_count"] = [0] * 6
    frame["income"] = [1., 3., np.nan, 1., 3., np.nan]
    frame["intended_balcon_amount"] = [-2., 1., 2., 3., 4., 5.]
    frame["credit_risk_score"] = [-10., 1., 2., 3., 4., 5.]
    frame["velocity_6h"] = [-10., 1., 2., 3., 4., 5.]
    return frame


def test_missing_codes_do_not_erase_valid_negative_values():
    frame = fixture_frame()
    frame.loc[0, "prev_address_months_count"] = -1
    cleaned = features(frame)
    assert np.isnan(cleaned.loc[0, "prev_address_months_count"])
    assert np.isnan(cleaned.loc[0, "intended_balcon_amount"])
    assert cleaned.loc[0, "credit_risk_score"] == -10
    assert cleaned.loc[0, "velocity_6h"] == -10
    assert not set(["month", "fraud_bool", "device_fraud_count"]) & set(cleaned.columns)


def test_preprocessing_does_not_learn_test_medians_or_categories():
    train = fixture_frame()
    x = features(train)
    estimator = make_model("logistic", False, read_config(), x.columns)
    estimator.fit(x, train.fraud_bool)
    numeric_columns = estimator.named_steps["preprocess"].transformers_[0][2]
    imputer = estimator.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[numeric_columns.index("income")] == 2
    unseen = x.iloc[[0]].copy()
    unseen["income"] = 100000
    unseen["source"] = "unseen"
    assert np.isfinite(estimator.predict_proba(unseen)).all()
    assert imputer.statistics_[numeric_columns.index("income")] == 2
    encoder = estimator.named_steps["preprocess"].named_transformers_["categorical"]
    assert "unseen" not in encoder.categories_[CATEGORICAL.index("source")]


@pytest.mark.parametrize("stale", [False, True])
def test_final_evaluation_requires_a_current_selection_freeze(tmp_path, stale):
    config = read_config()
    config["output_dir"] = str(tmp_path / "results")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    if stale:
        destination = tmp_path / "results"
        destination.mkdir()
        (destination / "selection.json").write_text(json.dumps({"fingerprint": {}, "data_sha256": config["data_sha256"]}))
    with pytest.raises(ValueError, match="stale" if stale else "Freeze"):
        final(config_path)
