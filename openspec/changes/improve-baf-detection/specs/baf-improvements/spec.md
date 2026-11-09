## Purpose

Test and explain fraud-detection improvements across multiple metrics while preserving the completed benchmark and preventing future-data leakage.

## ADDED Requirements

### Requirement: Separate exploratory study
Improvement experiments SHALL use separate configuration and outputs, preserve the first-study source/configuration/selection hashes and label month-7 comparisons as exploratory because its outcomes are already known.

#### Scenario: Run the improvement study
- **WHEN** improvement experiments execute
- **THEN** the original study records remain unchanged and the new report states the consumed-holdout limitation

### Requirement: Leakage-safe strategy comparisons
The study SHALL test model, feature-engineering, imbalance and temporal-weighting strategies on identical chronological evaluation cases. Features SHALL use only row-local inputs or mappings learned from eligible training cases, excluding target/time/constant fields.

#### Scenario: Transform an unseen category
- **WHEN** an evaluation category was absent from training
- **THEN** transformation uses a documented fallback without fitting on evaluation cases

### Requirement: Multiple operating objectives
The study SHALL report AP, ROC AUC, precision, recall, F1, F2, MCC, balanced accuracy, confusion counts and fixed-capacity detection. Thresholds SHALL be selected from earlier validation predictions and frozen before month-7 scoring; precision/recall target policies SHALL report target violations honestly.

#### Scenario: Select a threshold for a forward validation fold
- **WHEN** threshold-dependent validation performance is computed
- **THEN** its threshold is fitted only on predictions from earlier validation months, rather than that fold's labels

### Requirement: Measured gains and reproducible evidence
The study SHALL compare selected strategies with unchanged baselines, report paired uncertainty and ablations including unsuccessful strategies, and export inspectable findings, tests, dependencies and execution commands.

#### Scenario: A strategy does not improve performance
- **WHEN** a strategy underperforms its matched baseline
- **THEN** the findings retain its measured results and do not claim improvement or fresh confirmation
