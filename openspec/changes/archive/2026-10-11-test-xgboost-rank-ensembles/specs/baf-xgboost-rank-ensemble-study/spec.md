## Purpose

Evaluate categorical fraud models and label-free percentile ensembles using preserved study identities, matched representation controls and auditable chronological recall-weighted decisions.

## ADDED Requirements

### Requirement: Separate finite preserved ensemble study
The study SHALL declare its finite models, score strategies, operating policies, dependency versions and destinations before full fitting. It SHALL retain positive-class F2 beta=2 without a fixed review quota or precision floor and preserve all eight earlier source/config/software/selection identities and measured findings. Reference reuse SHALL verify identity and identical evaluation populations; absent caches SHALL be reproducible under frozen recipes only in new destinations.

#### Scenario: Install an additional model dependency
- **WHEN** the study adds its declared model package
- **THEN** earlier package versions and frozen identities remain unchanged, and incompatible references are rejected before fitting

### Requirement: Training-only categorical models and reproducible inference
All model preprocessing and fitting SHALL use strictly earlier eligible training rows. Category vocabularies and missing-value behavior SHALL be explicit, preserve feature identity, and exclude target/time/leakage fields from predictors. Models, adapters, parameters and prediction populations SHALL be saved with verified hashes. Reloaded models SHALL reproduce finite scores, including when evaluation data has missing or unseen categories; incompatible or partial artifacts SHALL be rejected.

#### Scenario: Predict applications with unseen categories
- **WHEN** an evaluation category did not occur in eligible training data
- **THEN** it follows the declared unknown-category rule without expanding the training vocabulary or fitting on evaluation rows

### Requirement: Label-free auditable score composition
Percentile representations SHALL depend only on finite aligned scores in the current unlabeled batch, with a declared tie convention. Ensembles SHALL use predeclared members and weights and expose their dependency hashes. Raw and percentile single-member controls SHALL accompany blends. Batch composition dependence SHALL be explicit, and percentile or blended scores SHALL NOT be presented as calibrated fraud probabilities.

#### Scenario: Change labels without changing current scores
- **WHEN** the same current score batch is supplied with different labels
- **THEN** percentile representations, ensemble scores and inference-time composition are unchanged

### Requirement: Comparable forward actual-F2 selection
All strategies SHALL predict identical future applications. Policy learning SHALL use only earlier out-of-time predictions and labels; blend membership and weights SHALL be fixed before fitting rather than selected from the scored fold. Selection SHALL use actual macro month-5/6 F2 with deterministic tie-breaking. Findings SHALL distinguish model, representation, blending and policy changes, retain both old pooled references, and report precision, recall, workload and confusion counts alongside F2.

#### Scenario: Evaluate a blend on month 6
- **WHEN** the blend generates month-6 operating decisions
- **THEN** its base models use only months 0-5, its membership and weights follow the predeclared grid, its percentile transform uses no current labels, and its operating rule learns only from earlier out-of-time predictions

### Requirement: Frozen limited transfer and honest improvement claims
The actual development winner, same-strategy pooled rule, both old pooled references and all declared representation/member controls SHALL be frozen with their dependency identities before further access to consumed month 7. Transfer SHALL evaluate only those frozen roles and SHALL NOT sweep losing settings or hindsight thresholds. Positive improvement claims SHALL require positive measured differences on their named scope; overlapping-zero conditional intervals SHALL preclude clear or confirmed gain claims without becoming a new success gate. A failed transfer SHALL remain visible and SHALL NOT be called independent confirmation.

#### Scenario: A development blend gain fails transfer
- **WHEN** the frozen selected blend has lower consumed-period F2 than a named baseline
- **THEN** findings expose the failed comparison and do not claim improvement over that baseline on future applications

### Requirement: Checked evidence and completion
The study SHALL export every declared operating, macro, pooled, matched-contrast, score and diversity table, dependency provenance, fixed-decision paired uncertainty, inspected figures, exact identities and reproduction instructions. Uncertainty SHALL state its conditioning and exclusions. Assistance sources SHALL be acknowledged. Tasks SHALL remain incomplete until actual outputs are verified, and a completed search without measured improvement SHALL NOT satisfy the performance goal.

#### Scenario: Finish an unsuccessful grid
- **WHEN** all models, score strategies and checks finish without the required F2 gain
- **THEN** negative results and frozen artifacts are preserved, and the improvement goal remains unachieved
