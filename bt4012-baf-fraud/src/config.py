"""Project-wide settings. Change values here rather than inside the experiments."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- Columns in the BAF dataset -------------------------------------------
TARGET = "fraud_bool"
MONTH = "month"

CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

# BAF encodes missing values as negative numbers in these columns
# (see the dataset datasheet). We turn them into NaN and let the models
# handle them (imputation + missing indicators for LR/RF, native NaN for LightGBM).
MISSING_IF_NEGATIVE = [
    "prev_address_months_count",
    "current_address_months_count",
    "intended_balcon_amount",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

# Protected attribute used for the fairness check (as in the BAF paper).
AGE_COL = "customer_age"
AGE_THRESHOLD = 50  # groups: < 50 and >= 50


@dataclass
class Settings:
    data_path: Path = ROOT / "data" / "Base.csv"
    results_dir: Path = ROOT / "results"
    fig_dir: Path = ROOT / "figures"

    # Fraction of rows to keep (stratified by month and label). 1.0 = full data.
    sample_frac: float = 1.0
    seeds: list = field(default_factory=lambda: [0, 1, 2])

    # Standard BAF protocol trains on months 0-5 and tests on 6-7.
    # We hold month 5 out as a validation month, used ONLY to tune
    # hyperparameters and choose decision thresholds, so the test months
    # are never touched until the final evaluation.
    train_months: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    val_months: list = field(default_factory=lambda: [5])
    test_months: list = field(default_factory=lambda: [6, 7])

    # Training-window study: for each test month t, compare expanding
    # (all months < t) with rolling windows of the last k months.
    window_test_months: list = field(default_factory=lambda: [4, 5, 6, 7])
    window_ks: list = field(default_factory=lambda: [1, 2, 3])

    # Operating points
    fpr_target: float = 0.05                                        # BAF benchmark: recall at 5% FPR
    review_capacities: list = field(default_factory=lambda: [0.01, 0.02, 0.05])  # share of applications reviewed
    # Cost of a missed fraud relative to a false alarm (review cost + customer friction).
    # These are ASSUMPTIONS; the report should present them as a sensitivity analysis.
    cost_ratios: list = field(default_factory=lambda: [10, 25, 50, 100])

    # Model sizes
    rf_trees: int = 200
    lgbm_trees: int = 500
    undersample_ratio: int = 10   # negatives kept per positive in the "undersample" strategy

    n_bootstrap: int = 200
    n_jobs: int = -1

    def quick(self):
        """Small, fast settings for smoke tests."""
        self.sample_frac = min(self.sample_frac, 0.1)
        self.seeds = [0]
        self.rf_trees = 50
        self.lgbm_trees = 150
        self.n_bootstrap = 50
        return self

    def ensure_dirs(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
