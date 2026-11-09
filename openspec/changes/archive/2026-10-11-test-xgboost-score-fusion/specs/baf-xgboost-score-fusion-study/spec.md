## Purpose

Evaluate fixed raw-score and log-odds fraud-score fusion with preserved model identities, matched controls and auditable chronological recall-weighted operating decisions.

## ADDED Requirements

### Requirement: Separate finite preserved fusion study
The study SHALL declare its finite members, composition methods, weights, numerical safeguards, operating policies and separate destinations before new evaluation. It SHALL retain positive-class F2 beta=2 without an imposed review quota or precision floor and preserve all nine earlier source/config/software/selection identities and findings. Reused artifacts SHALL prove identical recipes and evaluation populations; genuinely absent caches SHALL be reproduced only under frozen recipes in the new destination. Partial or incompatible artifacts SHALL be rejected.

#### Scenario: Reuse a frozen base model
- **WHEN** the new study uses an existing base prediction
- **THEN** its model, training population, preprocessing, software and prediction identities are verified before reuse and the original artifacts remain unchanged

#### Scenario: Reproduce a missing cache
- **WHEN** a required old base cache is wholly absent
- **THEN** the exact frozen recipe is reproduced in the new destination and its fit is counted separately from verified reuse

### Requirement: Label-free finite row-wise score composition
Raw fusions SHALL use the declared weighted mean of aligned member scores. Log-odds fusions SHALL use the same fixed weights after the declared finite endpoint clipping and logit transformation. Composition SHALL use no current labels or batch-derived ranks and SHALL expose member provenance, clipping counts and score ranges. Raw and transformed single-member controls SHALL accompany each composition type. Non-finite, out-of-range, misaligned or undeclared inputs SHALL be rejected, and fused scores SHALL NOT be asserted to be calibrated fraud probabilities.

#### Scenario: Change current labels or unrelated applications
- **WHEN** a current application's member scores remain fixed while its label or unrelated applications change
- **THEN** that application's composed score remains unchanged and no current labels enter the inference calculation

#### Scenario: Supply an endpoint score
- **WHEN** a valid member score equals zero or one
- **THEN** the declared clipping rule produces a finite log-odds score and its clipping is visible in diagnostics

### Requirement: Comparable chronological actual-F2 selection
All declared strategies SHALL predict identical development applications using base models and preprocessing fitted strictly earlier than each prediction month. Operating policies SHALL learn only from earlier out-of-time predictions and labels. Composition membership and weights SHALL be fixed before evaluation. Selection SHALL use actual macro month-5/6 F2 with deterministic tie-breaking, retaining both old pooled references. Reports SHALL distinguish fusion, transformation and policy effects and report precision, recall, workload, confusion counts and other declared metrics alongside F2.

#### Scenario: Evaluate month 6
- **WHEN** a fusion generates month-6 operating decisions
- **THEN** its base models use months 0-5, its composition follows the predeclared grid without month-6 labels, and its operating rule learns only from eligible earlier prediction months

### Requirement: Complete matched comparisons
The study SHALL retain every declared strategy and policy result, including failures and ties. It SHALL export comparisons against both original pooled references, non-pooled versus own-pooled policies, transformed versus raw singles, fusion versus each corresponding member, and log-odds versus raw fusion with all other factors matched. Every contrast SHALL identify its factors and be reconstructible from saved metrics and decisions.

#### Scenario: Attribute a log-odds fusion result
- **WHEN** a log-odds fusion is reported
- **THEN** its comparison with each transformed member and its otherwise matched raw fusion is available without substituting a different model, weight or policy

### Requirement: Frozen limited exploratory transfer
The actual development winner, same-strategy pooled rule, both old pooled references and the member/representation controls predetermined by the winner's declared strategy type SHALL be frozen with their dependencies and operating rules before further access to consumed month 7. Transfer SHALL evaluate only those roles, deduplicate base fits while retaining role comparisons, and SHALL NOT sweep losing settings or compute hindsight thresholds. Positive improvement claims SHALL require positive measured differences on the named scope; overlapping-zero conditional intervals SHALL preclude clear or confirmed gain claims. Failed transfer SHALL remain visible and SHALL NOT be called independent confirmation.

#### Scenario: A development gain fails transfer
- **WHEN** the frozen winner has lower consumed-month F2 than a named baseline
- **THEN** findings expose that failed comparison and do not claim out-of-time improvement over that baseline

### Requirement: Verified evidence and reproducible completion
The study SHALL export all declared operating, macro, pooled, contrast, score and clipping tables; dependency provenance; exact data/source/config/software and freeze/access identities; paired fixed-decision uncertainty; inspected figures; findings; assistance attribution; and runnable reproduction instructions. Uncertainty SHALL state its conditioning and exclusions. Tasks SHALL remain incomplete until actual outputs are checked, and completing a search without measured performance improvement SHALL NOT satisfy the user's performance goal.

#### Scenario: Finish the finite grid without improvement
- **WHEN** all declared strategies have been checked and none improves the target evaluation scope
- **THEN** the report records the negative result, preserves the study identity and leaves the performance goal unachieved
