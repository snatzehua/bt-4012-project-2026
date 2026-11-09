## Purpose

Test calibrated batch decisions against empirical recall-weighted policies with strict chronological provenance, explicit approximation limits and verified actual F2 outcomes.

## ADDED Requirements

### Requirement: Separate preserved policy study
The study SHALL predeclare a finite policy grid, preserve all five completed studies and their negative findings, retain positive-class F2 with beta=2 and impose no review quota or precision floor. Compatible score reuse SHALL require source/config/software/data and prediction provenance checks; missing scores SHALL be reproducible in a new destination without rewriting prior studies.

#### Scenario: Reuse a chronological stack score
- **WHEN** its member rows, source identity, combiner training boundary or saved prediction differs from the frozen source
- **THEN** the policy study rejects reuse rather than accepting the candidate name alone

### Requirement: Earlier-only calibration and unlabeled batch choices
Calibration SHALL use only eligible out-of-time predictions and labels from months before the evaluated month. Batch flagged-count selection SHALL depend only on frozen calibration, current scores and stable IDs, never current labels. The study SHALL disclose batch inference and distinguish its plug-in F2 surrogate from actual F2 and exact expected F2.

#### Scenario: Score month 6
- **WHEN** a calibrated batch policy produces month-6 decisions
- **THEN** its pooled calibrator uses month-4/5 predictions and labels or its latest calibrator uses month 5, while month-6 labels affect neither calibration nor batch choice

### Requirement: Comparable actual performance and operating controls
Selection SHALL use actual chronological macro F2 on identical future cases, with both old empirical baseline policies and a same-model empirical comparator. The study SHALL report monthly and pooled metrics, calibration diagnostics, precision/recall, workload, false positives, missed fraud and paired conditional uncertainty. Surrogate or calibration improvements alone SHALL NOT establish performance-goal achievement.

#### Scenario: A surrogate improves while measured F2 falls
- **WHEN** a calibrated policy estimates a better operating point but its actual forward F2 is worse
- **THEN** findings retain that failure and do not replace the user's objective with the surrogate, Brier score or AP

### Requirement: Diagnostic boundary and immutable transfer
Any label-aware best-ranking-prefix calculation SHALL be isolated as an optimistic development diagnostic, computed after development selection freeze and excluded from selection and operating-rule fitting. Selected model/policy/calibrator/dependencies SHALL be frozen before further consumed-month access, and month 7 SHALL remain exploratory.

#### Scenario: Inspect a current-month optimal prefix
- **WHEN** its F2 exceeds the frozen policy's F2
- **THEN** it is identified as label-dependent hindsight rather than an achieved deployable result, and no frozen policy is changed

### Requirement: Verified evidence and reproduction
The study SHALL export exact recipes and identities, complete policy tables including failures, checked predictions/metrics/intervals, inspected figures, selection/access provenance and reproduction instructions. Tasks SHALL remain incomplete until actual outputs are checked. Without a measured gain on the named scope, the performance objective SHALL remain unachieved.

#### Scenario: Finish the finite policy grid without improvement
- **WHEN** all declared policies have been measured but none beats the required references
- **THEN** the report states that result and leaves the goal active rather than claiming that completed calibration work achieved it
