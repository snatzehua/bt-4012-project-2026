## Purpose

Evaluate training-only additive nonlinear fraud features against matched linear controls, with reproducible chronological recall-weighted decisions and honest exploratory transfer claims.

## ADDED Requirements

### Requirement: Separate finite preserved representation study
The study SHALL declare the complete representation, regularization, training-weight and operating-policy grid before fitting. It SHALL retain positive-class F2 beta=2 without an imposed review quota or precision floor and preserve all ten earlier data/source/config/software/selection identities and findings. Cached references SHALL prove identical recipes, training populations, preprocessing and inference outputs; missing artifacts SHALL be reproduced only under the original recipes in the new destination. Partial or incompatible caches SHALL be rejected.

#### Scenario: Reuse a completed reference
- **WHEN** an earlier prediction or saved model is reused
- **THEN** its provenance, population and score identities are verified and no earlier study artifact changes

### Requirement: Training-only additive features and matched controls
Nonlinear models SHALL use the declared univariate smooth basis, numeric/discrete/categorical roles, missingness treatment, training-only knot deduplication and out-of-range behavior. Matched linear models SHALL retain the same original predictors, eligibility, missing indicators, categorical handling, scaling convention, regularization and training weights, changing only the smooth representation. All fitted statistics and feature schemas SHALL depend only on eligible earlier predictors. Degenerate training columns and previously unseen categories or missingness SHALL have declared deterministic behavior. Labels, month and the excluded constant field SHALL NOT enter feature construction.

#### Scenario: Repeated knots or degenerate training values
- **WHEN** an eligible numeric predictor has repeated proposed knots, one observed value or no observed values
- **THEN** the declared training-only deduplication or fallback is recorded and inference remains finite under the same saved feature schema

#### Scenario: Future predictor extremes and new missingness
- **WHEN** a future batch contains an out-of-range numeric value, unseen category or missing value absent from training
- **THEN** the saved transform applies its declared inference behavior without refitting or changing another application's output

### Requirement: Verified solver and persisted inference
Every new fitted recipe SHALL record its actual solver parameters, convergence status, iterations, warnings, transformed dimension and feature identity. A nonconverged or numerically invalid fit SHALL NOT silently become a selected result or trigger a new recipe. Saved preprocessing and model artifacts SHALL reproduce positive-class predictions within the declared tolerance and reject changed schemas or identities. Numerical coefficient, intercept and additive-contribution checks SHALL support the claimed model interpretation; predictive nonlinear effects SHALL NOT be described as causal.

#### Scenario: Optimizer does not converge
- **WHEN** fitting produces a convergence warning or fails the declared numerical gate
- **THEN** the study records the failure and stops the affected execution before selection, without silently changing iterations, regularization or solver

### Requirement: Comparable chronological actual-F2 selection
All selectable strategies SHALL predict the same future development applications with models fitted strictly earlier. Policies SHALL learn exclusively from earlier out-of-time predictions and labels. Selection SHALL maximize actual macro month-5/6 F2 with the existing deterministic tie-breaking and retain the previously selected fusion and both original pooled LightGBM references. Reports SHALL retain the entire grid, monthly and macro metrics, pooled-confusion summaries, precision, recall, workload, missed fraud and legitimate flags. Every declared representation, weight, regularization and policy contrast SHALL be reconstructible with its other factors matched.

#### Scenario: Compare a spline with its linear control
- **WHEN** a nonlinear result is attributed to its feature representation
- **THEN** the control uses the identical training/evaluation population, regularization, weighting and policy, and its metric difference is exported even if zero or negative

### Requirement: Frozen limited exploratory transfer
The development winner, all three old reference policies, same-strategy pooled rule and winner-type-determined matched controls SHALL be frozen with their full dependencies before further month-7 access. Only those frozen roles SHALL be evaluated on the already-consumed period, with deduplicated model fits and explicit fit/reuse counts. No losing-setting sweep, hindsight threshold or independent-confirmation claim SHALL be permitted. A performance-goal achievement claim SHALL require positive measured F2 differences against all three named references on both development and consumed transfer; clear-gain claims additionally require positive conditional interval bounds on the named scope.

#### Scenario: Beat balanced while trailing a stronger reference
- **WHEN** the selected strategy improves balanced LightGBM but does not improve the selected fusion or unweighted reference on a required scope
- **THEN** the report identifies the failed comparison and leaves the broader performance goal unachieved

### Requirement: Reproducible checked evidence
Completion SHALL require checked configuration/protocol, complete operating/contrast/feature/solver/provenance outputs, data/source/config/software identities, immutable freeze/access records, persisted models, paired fixed-decision uncertainty, inspected figures, findings, runnable reproduction instructions and assistance attribution. Conditional uncertainty SHALL disclose adaptive search, selection, retraining, temporal and generative exclusions. Tasks SHALL remain incomplete until their real outputs are verified; planning completion or a finished finite search alone SHALL NOT establish a performance gain.

#### Scenario: Finish implementation and reporting
- **WHEN** the study is declared complete
- **THEN** actual outputs satisfy each requirement, methodological checks and strict OpenSpec validation pass, and the performance goal status follows measured evidence rather than artifact existence
