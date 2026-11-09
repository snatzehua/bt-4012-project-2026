## Purpose

Evaluate the fixed additive-versus-linear fraud modelling question with independently checked Newton optimization, preserving the stopped attempt and measuring chronological recall-weighted performance against all existing references.

## ADDED Requirements

### Requirement: Separate prospective preserved study
The study SHALL declare its complete recipes, numerical gates, controls, policies and destinations before fitting. It SHALL preserve all ten completed study identities and the stopped spline attempt's source/configuration and available artifacts, including failure evidence. Missing optional stopped-model caches SHALL be disclosed without recreating the stopped attempt. Required reference caches SHALL be reused only after checking recipes and populations; wholly absent required caches SHALL be reproduced under their frozen recipes in the new destination, with actual fits counted separately. Partial or incompatible required caches SHALL be rejected. No stopped or completed study SHALL be overwritten or retroactively declared successful.

#### Scenario: Start with available stopped artifacts
- **WHEN** the successor study starts with some stopped-attempt artifacts available locally
- **THEN** it verifies their recorded identities, captures their presence and preserves them throughout execution without using partial old results as its selectable grid

#### Scenario: Reproduce on a fresh clone
- **WHEN** optional stopped model caches are absent but tracked failure records and mandatory frozen dependencies are intact
- **THEN** the study records unavailable solver diagnostics, leaves the stopped experiment untouched and reproduces only genuinely absent required references using their original recipes

### Requirement: Identical training-only representation controls
The study SHALL compare the declared original linear and additive representations on identical earlier training populations, crossing the same regularization and class-weight settings. Predictor roles, missing handling, observed-only knots, duplicate/degenerate handling, scaling, categorical inference and sparse dimensions SHALL be training-only and match the preserved representation contract. Current labels or unrelated inference rows SHALL NOT change a row's transformed features or score. Serialized inference SHALL retain the fitted feature identity and numerical output.

#### Scenario: Predict future missing or out-of-range values
- **WHEN** future rows contain unseen categories, new numeric missingness or values beyond training knots
- **THEN** the fixed training schema handles them according to the declared rules, reports diagnostics and does not refit preprocessing

### Requirement: Genuine checked Newton optimization
Every new selectable model SHALL use the declared Newton optimizer without internal alternative-solver fallback. Acceptance SHALL require finite independent training objective and gradient, the declared absolute gradient bound, valid iterations below the fixed limit, no fit warnings, valid class/dimension identities and independently reconstructed persisted scores. Actual fallback entry SHALL be checked even when it emits no warning. Optional old-solver objective comparisons SHALL remain numerical diagnostics outside selection. A failed gate SHALL retain the actual evidence and stop completion without dropping the recipe, relaxing the gate or changing the solver.

#### Scenario: Internal fallback without a warning
- **WHEN** the declared solver enters its internal alternative-solver path without emitting a warning
- **THEN** the execution records that event and rejects the model as a valid new result

#### Scenario: Full-data fit fails after miniature checks pass
- **WHEN** a full-data recipe fails any declared numerical gate
- **THEN** the failure and completed work remain visible, selection and transfer remain unavailable, and the grid is not reported as a completed negative performance result

### Requirement: Complete comparable chronological actual-F2 selection
All declared candidates SHALL predict identical month-4/5/6 populations from strictly earlier training data. Policies SHALL learn only from earlier out-of-time predictions and labels. Selection SHALL use actual macro month-5/6 positive-class F2 with beta=2, deterministic ties and no imposed review quota or precision floor. Every declared strategy/policy result SHALL be retained alongside precision, recall, confusion counts, workload and supplementary metrics; AP or training loss SHALL NOT substitute for F2.

#### Scenario: Evaluate month 6
- **WHEN** a candidate produces month-6 decisions
- **THEN** its model and transform use months 0-5, its policy uses only eligible month-4/5 predictions and labels, and month-6 outcomes do not enter the rule

### Requirement: Complete matched numerical and performance comparisons
The study SHALL retain all declared reference, policy, representation, class-weight and regularization contrasts, including ties and negative differences, with reconstructible factors and arithmetic. New-solver fits SHALL NOT be substituted with old converged solver predictions. Available old-solver diagnostics SHALL prove matched training populations, representations and objectives and SHALL NOT expand the selectable search.

#### Scenario: Attribute a representation difference
- **WHEN** a spline-versus-linear contrast is reported
- **THEN** both sides share training/evaluation populations, C, weighting and policy, and the report exposes the representation dimension and regularization geometry limitation

### Requirement: Frozen bounded exploratory transfer and honest gain claims
Before further access to already-consumed month 7, the study SHALL freeze the development winner, three named reference policies, same-strategy pooled rule and the predeclared winner-type-matched roles with dependencies and final operating rules. Only those roles SHALL transfer, with duplicate fits removed and actual fit/reuse counts exposed. No losing-setting sweep, hindsight threshold or independent-confirmation claim SHALL be permitted. Achievement of the broader performance goal SHALL require positive measured F2 differences against all three references on both development and consumed transfer; clear-gain claims additionally require positive conditional interval bounds on the named scope.

#### Scenario: Improve development but fail a transfer comparison
- **WHEN** a development-selected candidate fails to improve any required reference on consumed month 7
- **THEN** the findings expose that comparison and leave the broader performance goal unachieved

### Requirement: Verified evidence and reproducibility
Completion SHALL require checked configuration/protocol, complete score/operating/contrast/feature/solver/provenance outputs, immutable freeze/access records, independent gradient and inference checks, conditional paired uncertainty, inspected figures, findings, runnable reproduction instructions and assistance attribution. Uncertainty SHALL disclose adaptive search, selection, retraining, temporal and generative exclusions. Task completion SHALL follow actual checked outputs, and planning or finite-search completion SHALL NOT establish a measured gain.

#### Scenario: Declare the study complete
- **WHEN** implementation and reporting are declared complete
- **THEN** each specified output exists and has been checked, preservation and methodological tests pass, strict OpenSpec validation passes, and goal status follows measured acceptance rather than artifact existence
