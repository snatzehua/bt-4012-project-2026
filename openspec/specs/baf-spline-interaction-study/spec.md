# baf-spline-interaction-study Specification

## Purpose

Test whether a fixed small set of nonlinear pair effects improves chronological recall-weighted fraud detection under matched additive controls and independently checked optimization.

## Requirements

### Requirement: Separate finite preserved interaction study
The study SHALL declare all pairs, feature rules, recipes, policies, numerical gates, references and destinations before fitting. It SHALL preserve all twelve completed identities, the separate Newton fit/analysis histories and the stopped attempt. Reused models/scores SHALL prove compatible recipes, preprocessing, populations, software and saved inference. Only wholly absent required caches SHALL be reproduced under their frozen recipes in the new destination with actual fits counted; partial/incompatible required caches SHALL be rejected. Unrelated prior grids SHALL NOT be required or rerun.

#### Scenario: Reproduce without historical bulk caches
- **WHEN** ignored required caches are wholly absent but mandatory tracked ancestry and restored selections are intact
- **THEN** the study reproduces only required exact recipes in its separate destination and leaves historical studies unchanged

#### Scenario: Encounter a partial required cache
- **WHEN** a required model/record/prediction set is incomplete or incompatible
- **THEN** the study rejects it rather than overwriting it or silently substituting another recipe

### Requirement: Fixed training-only nonlinear pair representation
The interaction representation SHALL retain the unchanged additive main-effect representation and add only the declared centered, scaled tensor-product pair blocks. Knots, medians, vocabularies, pair moments, inactive-column decisions and scaling SHALL use eligible training predictors only. Degenerate pairs/columns SHALL retain declared dimensions and zero inactive contributions. Missing, unseen-category and out-of-range inference SHALL follow frozen rules. Labels and unrelated inference rows SHALL NOT change a row's features or score. Saved metadata and coefficients SHALL permit independent feature and margin reconstruction.

#### Scenario: Infer missing or out-of-range pair values
- **WHEN** future applications contain missing values, unseen categories or values beyond training knots
- **THEN** the saved additive and pair transformations apply without refitting and expose the declared diagnostics

#### Scenario: Change labels or unrelated rows
- **WHEN** one application's predictors remain fixed while labels or other applications change
- **THEN** its transformed pair features and fraud score remain unchanged

### Requirement: Checked numerical fitting and complete failure evidence
New fits SHALL use the declared Newton optimization without alternative-solver fallback and pass independent objective/gradient, finite-class/dimension, iteration, warning and saved-inference checks. Resource accounting SHALL expose actual dimensions, storage, timings and process memory. A failed declared gate SHALL preserve evidence and leave the study incomplete without dropping pairs/recipes, relaxing the gate or substituting a solver after outcomes.

#### Scenario: Full fit fails after miniature integration passes
- **WHEN** any full interaction fit fails a numerical or allocation gate
- **THEN** its failure and completed work remain recorded, selection/transfer remain unavailable and no completed negative-performance claim is made

### Requirement: Comparable chronological actual-F2 selection
All declared strategies SHALL predict identical future applications from earlier-only base models/preprocessing. Operating policies SHALL use only earlier out-of-time predictions and labels. Selection SHALL use actual macro month-5/6 positive-class F2 beta=2 with deterministic ties and no imposed quota or precision floor. The latest previously selected fusion and three older references SHALL remain visible at their original operating policies, alongside all declared strategy/policy results and supplementary metrics. AP, pooled F2 or pair diagnostics SHALL NOT replace the macro objective.

#### Scenario: Evaluate an interaction model on month 6
- **WHEN** month-6 decisions are scored
- **THEN** feature/model fitting uses months 0-5 and policy fitting uses only eligible month-4/5 predictions and labels

### Requirement: Complete matched feature and operating comparisons
The study SHALL retain every declared reference, own-policy, interaction-versus-additive, class-weight and regularization comparison with reconstructible factors and arithmetic, including negative results and ties. Pair effects SHALL remain descriptive associations and SHALL NOT become post-outcome pair selection, causal claims or new inference rules. Equal nominal regularization SHALL NOT be asserted equal effective complexity across representations.

#### Scenario: Attribute an interaction gain
- **WHEN** an interaction model is compared with an additive control
- **THEN** training/evaluation populations, C, class weighting and operating policy are matched, with feature dimensions and regularization geometry disclosed

### Requirement: Frozen bounded exploratory transfer
Before further consumed-month access, the actual development winner, all four original reference policies, same-model pooled rule and predetermined matched model controls SHALL be frozen with exact dependencies and final earlier-only policies. Only these roles SHALL transfer with deduplicated fits and actual fit/reuse counts. Losing settings, hindsight cutoffs and independent-confirmation claims SHALL be excluded. Broader goal achievement SHALL require positive measured F2 differences against all four references on both development and consumed transfer; clear-gain language SHALL additionally require positive conditional interval bounds on the named scope.

#### Scenario: Fail a required reference comparison
- **WHEN** the frozen winner fails any required development or consumed comparison
- **THEN** the report preserves that failure and leaves the broader performance goal unachieved

### Requirement: Verified evidence and reproducible completion
Completion SHALL require checked prospective declarations, all declared score/operating/contrast/feature/numerical/provenance outputs, independently reconstructed features/inference/policies/metrics/factors, immutable freeze/access records, conditional paired uncertainty, inspected operating and pair-contribution figures, findings, runnable reproduction instructions and assistance attribution. Adaptive choices, repeated search and consumed-holdout limitations SHALL be explicit. Tasks SHALL remain incomplete until their outputs verify; planning or finite-search completion SHALL NOT establish a performance gain.

#### Scenario: Complete the finite experiment
- **WHEN** the study is declared complete
- **THEN** every requirement has checked output evidence, preservation/tests/strict OpenSpec validation pass and goal status follows measured acceptance rather than artifact existence
