# BAF focal-loss study

## Purpose

Test custom focal-loss fraud score learning against matched ordinary-loss controls with verified derivatives, strictly chronological operating policies and honest measured F2 outcomes.

## Requirements

### Requirement: Separate finite preserved loss study
The study SHALL predeclare a finite loss/control grid in a separate configuration and destination, retain positive-class F2 beta=2 with no quota or precision floor, and preserve all six completed studies and their negative findings. Missing reference predictions SHALL be reproducible under frozen recipes in new directories, while incompatible cached identities or populations SHALL be rejected.

#### Scenario: Reuse an earlier reference
- **WHEN** its recipe, source/software/data identity, training boundary or prediction population differs
- **THEN** the study rejects reuse rather than silently treating a model name as evidence of compatibility

### Requirement: Audited loss and numerical behavior
Custom losses SHALL consume raw binary margins and return verified first/second derivatives with explicit numerical safeguards. The study SHALL distinguish its true loss derivatives from any curvature approximation, report approximation use and verify ordinary-loss limiting behavior and controlled built-in equivalence before interpreting focal effects. Training weight SHALL remain distinct from evaluation beta and bank costs.

#### Scenario: A focal Hessian is nonpositive
- **WHEN** the configured optimizer replaces its curvature with a positive floor
- **THEN** the fit records that replacement and the report identifies the approximate optimization step instead of claiming exact Newton optimization

### Requirement: Chronological matched evaluation
All compared models SHALL predict identical future cases, fit feature handling and training on strictly earlier months, and learn operating policies only from earlier out-of-time predictions. Focal-versus-ordinary contrasts SHALL match training populations, tree parameters, class-weight conventions and initialization. Selection SHALL use actual macro forward F2, with both old F2 references and ordinary-loss controls retained.

#### Scenario: Evaluate month 6
- **WHEN** a candidate produces month-6 decisions
- **THEN** its model/preprocessing use months 0-5 and its policy uses only earlier development predictions, excluding month-6 labels from both fitting stages

### Requirement: Immutable exploratory transfer and actual improvement
The selected model recipe and policy SHALL be frozen before further consumed-month access. Transfer SHALL evaluate only predeclared selected/control dependencies and old references, remain exploratory, retain failed comparisons and report actual F2, precision/recall, workload, false positives, missed fraud and paired conditional uncertainty. Better training loss or AP alone SHALL NOT satisfy the performance objective.

#### Scenario: A focal candidate fails to beat ordinary loss
- **WHEN** its matched control has better actual forward F2 or its development gain fails consumed-period transfer
- **THEN** findings state that limitation and do not claim a focal-loss or out-of-time improvement

### Requirement: Verified reproducible evidence and completion
The study SHALL export all declared comparisons, numerical audits, exact identities, checked predictions/metrics/intervals, inspected figures, reproduction instructions and assistance attribution. Tasks SHALL remain incomplete until outputs exist and are checked. Finishing a search without a measured gain SHALL leave the performance goal unachieved.

#### Scenario: Complete fitting without improvement
- **WHEN** every declared fit and check finishes but actual F2 does not improve the required references
- **THEN** the report retains the negative outcome and does not equate completed implementation with performance-goal achievement
