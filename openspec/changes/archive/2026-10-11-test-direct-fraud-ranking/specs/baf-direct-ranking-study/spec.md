## Purpose

Evaluate direct fraud-ranking learning with auditable artificial queries, matched classification controls and strictly chronological recall-weighted decisions.

## ADDED Requirements

### Requirement: Separate finite preserved ranking study
The study SHALL predeclare its finite ranking/control grid, configuration, destinations and identities, retain positive-class F2 beta=2 without an imposed quota or precision floor, and preserve all seven completed studies and their findings. Reference reuse SHALL verify recipe, source/config/software/data and population compatibility; missing caches SHALL be reproducible in new directories under frozen recipes.

#### Scenario: Reuse an incompatible reference
- **WHEN** a cached reference has a changed identity or different evaluation cases
- **THEN** reuse is rejected without modifying earlier study artifacts

### Requirement: Target-independent complete query grouping
Artificial training queries SHALL depend only on eligible training months, stable row IDs, declared size and seed. Query formation SHALL preserve every training row exactly once, keep month boundaries, handle singleton tails without dropping rows, and retain all-negative queries. Label-dependent regrouping, balancing or removal SHALL NOT occur. Group sizes, ordered training identities and descriptive class counts SHALL be auditable, with artificial-query limitations stated.

#### Scenario: Change labels while keeping eligible rows fixed
- **WHEN** the same grouping recipe is applied after permuting training labels
- **THEN** ordered row membership and query boundaries are unchanged, although descriptive fraud counts can change

### Requirement: Verified raw-score model interface
Ranking predictions SHALL remain finite real-valued scores, preserve stable evaluation IDs and reproduce from saved models and training-fitted preprocessing. The study SHALL reject nonfinite or incompatible artifacts, verify the installed ranking/group contract before full fitting, and avoid treating ranking scores as probabilities or reporting probability-only losses for them.

#### Scenario: Produce scores outside the probability range
- **WHEN** a valid saved ranking model emits negative values or values above one
- **THEN** empirical thresholds and ranking metrics use those raw scores without clipping, sigmoid saturation or probability-calibration claims

### Requirement: Matched chronological actual-F2 evaluation
All candidates SHALL predict identical future cases using models/preprocessing fitted on strictly earlier data and operating rules learned only from earlier out-of-time predictions. Ranking/control comparisons SHALL hold declared feature handling, training rows/order, tree settings and regularization fixed. Selection SHALL use actual macro forward F2, retain both old pooled F2 references, and distinguish ranking-surrogate gains from actual fraud-decision gains.

#### Scenario: Evaluate month 6
- **WHEN** a candidate produces month-6 operating decisions
- **THEN** its model uses months 0-5 and its policy uses only earlier out-of-time predictions, excluding month-6 labels from both fitting stages

### Requirement: Frozen exploratory transfer and measured improvement
Selected model/policy, same-model pooled rule, both old references and predeclared matched controls SHALL be frozen before further access to the consumed evaluation month. Transfer SHALL exclude losing-preset sweeps and hindsight threshold optima. Findings SHALL expose actual F2, precision/recall, workload, confusion counts and paired conditional uncertainty; failed contrasts SHALL remain visible. A positive improvement claim SHALL require a positive measured difference on its named scope, and intervals overlapping zero SHALL preclude clear or confirmed gain claims.

#### Scenario: A ranking development gain fails transfer
- **WHEN** the frozen ranking policy has lower F2 than its reference on the consumed period
- **THEN** findings report that failure and do not claim out-of-time improvement or fresh independent confirmation

### Requirement: Verified evidence and honest completion
The study SHALL export all declared monthly/macro/pooled comparisons, grouping and prediction audits, exact frozen identities, inspected figures, uncertainty draws, reproduction instructions and assistance attribution. Tasks SHALL be checked only after actual outputs are verified, and finishing an unsuccessful search SHALL NOT satisfy the performance objective.

#### Scenario: Complete the grid without an actual-F2 gain
- **WHEN** all fits and checks finish but no selected policy improves the required references
- **THEN** the negative evidence is retained and the performance goal remains unachieved
