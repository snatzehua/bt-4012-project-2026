# BAF recall-weighted study

## Purpose

Optimize and explain recall-weighted fraud detection on chronological BAF applications, with reproducible evidence and explicit precision and review-workload consequences.

## Requirements

### Requirement: Explicit recall-weighted objective
The study SHALL optimize positive-class F-beta with beta=2, report its formula and distinguish this objective from class-support-weighted F1. Selection SHALL use macro chronological F2 without an imposed review-capacity or precision constraint. Precision, recall, F1, F3, AP, confusion counts and flagged fraction SHALL accompany F2.

#### Scenario: Evaluate a candidate operating policy
- **WHEN** a candidate produces fraud decisions
- **THEN** F2 equals 5TP/(5TP+4FN+FP), and the report exposes both missed fraud and legitimate applications flagged

### Requirement: Comparable chronological search
The study SHALL declare a finite training and policy search before new fitting, predict identical development applications with all competitors, and fit preprocessing and models only on earlier months. Operating-policy parameters SHALL use only earlier development prediction labels than the scored fold. Any blend-member selection SHALL obey that same time boundary.

#### Scenario: Score month 5
- **WHEN** a policy is evaluated on month 5
- **THEN** its model uses months before 5 and its policy parameters and selected blend members use development predictions from month 4 only

### Requirement: Separate preserved study identities
The study SHALL use separate configuration and outputs, verify the existing frozen data/source/config/software and selection records before reuse, and reject incompatible cached predictions. It SHALL keep month 7 out of tuning and label any subsequent comparison on it as exploratory on an already-consumed holdout.

#### Scenario: Reuse a development prediction
- **WHEN** a saved prediction has an incompatible identity or different evaluation cases
- **THEN** the study rejects reuse rather than silently mixing artifacts

### Requirement: Verified measured improvement
The study SHALL compare training and policy changes against the previous unweighted and balanced LightGBM F2 policies on identical cases, report monthly and macro differences, and provide paired F2 uncertainty conditional on fixed models and policies. An improvement claim SHALL require a positive measured difference on the named evaluation scope; uncertainty overlapping zero SHALL preclude claims of a clear or confirmed gain. Failed comparisons SHALL remain visible.

#### Scenario: Development winner underperforms on consumed month 7
- **WHEN** the frozen winner has lower month-7 F2 than its baseline
- **THEN** findings state that development improvement did not transfer and do not claim out-of-time improvement

### Requirement: Auditable reproduction and completion
The study SHALL export exact configuration, source/software/data identity, selection provenance, metric tables, findings, preservation checks and execution commands. Tasks SHALL remain incomplete until their outputs exist and have been checked; completing an experiment without measured improvement SHALL NOT be described as achieving the user's performance goal.

#### Scenario: Finish the finite search without a gain
- **WHEN** all configured candidates have been measured but none improves the target scope
- **THEN** the report records that result and further work remains explicit rather than asserting that the performance goal was achieved
