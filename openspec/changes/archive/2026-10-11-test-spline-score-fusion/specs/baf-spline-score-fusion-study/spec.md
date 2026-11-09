## Purpose

Test whether preserved additive fraud scores complement the existing tree fusion through fixed common-margin composition, with complete chronological F2 controls and honest exploratory transfer.

## ADDED Requirements

### Requirement: Separate finite preserved study
The study SHALL prospectively declare its members, transformations, weights, policies, numerical rules and destinations. It SHALL preserve all eleven completed study identities, including separate Newton fit/analysis histories, and the stopped L-BFGS attempt. Required model/score reuse SHALL verify recipes, preprocessing, populations and provenance. Wholly absent required caches SHALL be reproduced only under frozen recipes in the new destination with actual fits counted separately; partial or incompatible required caches SHALL be rejected.

#### Scenario: Reuse a Newton score
- **WHEN** a preserved Newton model supplies a development score
- **THEN** its original fit and corrected analysis identities, training-only representation, numerical validity and saved inference are checked without modifying the source study

#### Scenario: Reproduce without ignored caches
- **WHEN** required cached models and predictions are wholly absent
- **THEN** only declared required recipes are reproduced in the new destination, with actual fitting disclosed and all old studies unchanged

### Requirement: Domain-correct label-free fixed composition
Composition SHALL combine the declared clipped-log-odds Newton score and the existing fusion's finite margin using fixed weights. It SHALL NOT treat the fusion margin as a probability or apply a probability logit to it. Composition SHALL use no current labels, learned weights or batch-dependent statistics. Original/transformed single-member controls, clipping diagnostics and complete parent identities SHALL accompany blends. Invalid domains, dimensions, alignment, weights or non-finite outputs SHALL be rejected. Composed margins SHALL NOT be asserted calibrated fraud probabilities.

#### Scenario: Receive an unbounded fusion margin
- **WHEN** a valid fusion member score lies outside zero to one
- **THEN** composition accepts it as a finite margin, transforms only the Newton probability-like score and reconstructs the declared weighted sum exactly

#### Scenario: Change labels or unrelated applications
- **WHEN** a row's member scores remain fixed while current labels or unrelated rows change
- **THEN** that row's composed score remains unchanged

### Requirement: Comparable chronological actual-F2 selection
All declared strategies SHALL predict identical month-4/5/6 applications using earlier-only base models and preprocessing. Policies SHALL learn only from earlier out-of-time predictions and labels. Actual macro month-5/6 positive-class F2 beta=2 SHALL select the winner with deterministic ties and no imposed review quota or precision floor. All strategies/policies and declared supplementary metrics SHALL remain visible; AP and diagnostic complementarity SHALL NOT replace actual F2 selection.

#### Scenario: Evaluate a month-6 blend
- **WHEN** month-6 decisions are evaluated
- **THEN** every base model uses months 0-5 and the operating policy uses only eligible month-4/5 predictions and labels, with fixed composition independent of month-6 outcomes

### Requirement: Complete matched controls and diagnostics
The study SHALL retain all declared reference, policy, transformation, member, representation, weighting and composition-weight comparisons, including zeros and negative results. Each contrast SHALL expose reconstructible matched factors and arithmetic. Score correlations and joint-error diagnostics SHALL be reporting evidence only and SHALL NOT alter composition or retrospective selection criteria.

#### Scenario: Attribute a blend result
- **WHEN** a blend is reported as improving a comparator
- **THEN** its otherwise matched member, representation, training-weight and composition-weight comparisons remain available, and the claim identifies its actual evaluation scope

### Requirement: Frozen limited exploratory transfer
Before further consumed-month access, the actual development winner, all three pooled references, same-strategy pooled rule and predeclared winner-type controls SHALL be frozen with complete dependencies and final earlier-only policies. Only those roles SHALL transfer with deduplicated work and actual fit/reuse counts. No losing-grid sweep, hindsight threshold or independent-confirmation claim SHALL be allowed. Broader performance-goal achievement SHALL require positive measured F2 differences against all three references on both development and consumed transfer; clear-gain claims additionally require positive conditional interval bounds on the named scope.

#### Scenario: Improve balanced but fail a stronger reference
- **WHEN** the selected blend fails any required comparison on development or consumed transfer
- **THEN** the findings retain that failure and leave the broader performance goal unachieved

### Requirement: Verified reproducible completion
Completion SHALL require checked configuration/protocol, complete score/operating/contrast/clipping/diversity/provenance outputs, exact source/config/software identities, immutable freeze/access records, independent composition/policy/metric reconstruction, conditional paired uncertainty, inspected figures, findings, runnable reproduction instructions and assistance attribution. Adaptive source selection and consumed-holdout limitations SHALL be explicit. Tasks SHALL remain incomplete until their actual outputs verify; finishing a finite search SHALL NOT establish a performance gain.

#### Scenario: Finish the finite experiment
- **WHEN** the study is declared complete
- **THEN** each requirement has checked evidence, preservation/tests/strict OpenSpec validation pass and goal status follows measured comparisons rather than artifact completion
