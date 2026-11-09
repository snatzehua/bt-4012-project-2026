# BAF empirical study

## Purpose

Provide reproducible development-only EDA and measured, auditable comparisons for the BAF research hypotheses.

## Requirements

### Requirement: Development-only exploration
The study SHALL export schema, quality, missingness, monthly prevalence and feature-distribution summaries without using final-test cases to select features or settings.

#### Scenario: Run EDA
- **WHEN** the development EDA command executes
- **THEN** outputs contain only months 0-6 and include encoded missingness separately from literal nulls

### Requirement: Controlled chronological experiments
The study SHALL evaluate expanding, recent-K and size/class-matched historical-K training on identical future cases, with recorded seeds and sample audits. Transformations SHALL be fitted only on each training subset.

#### Scenario: Build matched samples
- **WHEN** the historical control is constructed
- **THEN** it matches the recent sample's exact size and class counts, has no future rows and rejects infeasible controls

### Requirement: Frozen final evaluation
The study SHALL select settings from forward validation and persist data, config and source hashes before final scoring. It SHALL reject missing or stale selections.

#### Scenario: Evaluate without a valid freeze
- **WHEN** the final command has no selection manifest or its hashes differ
- **THEN** execution stops before final model scoring

### Requirement: Auditable findings and reproducibility
The study SHALL report capacity-based detection and false positives, average precision, threshold metrics, runtime, variation and conditional uncertainty. It SHALL label random diagnostics and limitations, provide reproduction commands and retain small readable results without committing raw data.

#### Scenario: Inspect reported recency differences
- **WHEN** the findings report compares recent and historical training
- **THEN** it cites measured matched-size results, sampling-seed variation and conditional paired uncertainty without claiming causal or production performance
