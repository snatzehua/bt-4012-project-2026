"""Checks for presentation evidence integrity; no fitting or raw-data access."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("submission_evidence", ROOT / "scripts/build_submission_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def test_export_matches_frozen_selection_and_confusion_counts():
    manifest, files = evidence.collect()
    assert manifest["validated"] == {"original_final_rows": 14, "improvement_final_rows": 5,
                                      "f2_consumed_roles": 5, "model_fits": 0}
    assert "0.33088096180021687" in files["f2_comparison.csv"]
    assert "0.3722397476340694" in files["f2_comparison.csv"]
    assert "test-context-feature-ablation-f2" in manifest["excluded_incomplete_studies"]


def test_export_is_byte_deterministic():
    first_manifest, first_files = evidence.collect()
    second_manifest, second_files = evidence.collect()
    assert first_files == second_files
    assert json.dumps(first_manifest, sort_keys=True) == json.dumps(second_manifest, sort_keys=True)


def test_source_corruption_is_rejected_before_export(tmp_path):
    for name in evidence.SOURCE_SHA256:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    target = tmp_path / "reports/baf-study-v1/final_metrics.csv"
    target.write_text(target.read_text().replace("319,650", "320,649", 1))
    with pytest.raises(ValueError, match="Frozen evidence hash mismatch"):
        evidence.collect(tmp_path)


def test_missing_evidence_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing required evidence"):
        evidence.collect(tmp_path)


@pytest.mark.parametrize("change", [{"precision": "0.5"}, {"tp": "708.5"},
                                     {"fn": "719"}, {"f2": "0.4"}, {"tp": "nan"}])
def test_inconsistent_or_nonfinite_confusion_is_rejected(change):
    row = {"tp": "708", "fp": "3090", "fn": "720", "tn": "92325", "n": "96843", "fraud": "1428",
           "precision": str(708 / 3798), "recall": str(708 / 1428), "f2": str(5 * 708 / (5 * 708 + 4 * 720 + 3090))}
    row.update(change)
    with pytest.raises(ValueError):
        evidence.validate_confusion(row, n_key="n", fraud_key="fraud")


def test_export_needs_only_committed_evidence_not_raw_data_or_model_cache(tmp_path):
    for name in evidence.SOURCE_SHA256:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    manifest, files = evidence.collect(tmp_path)
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "outputs").exists()
    assert len(files) == 10
    assert manifest["validated"]["model_fits"] == 0
