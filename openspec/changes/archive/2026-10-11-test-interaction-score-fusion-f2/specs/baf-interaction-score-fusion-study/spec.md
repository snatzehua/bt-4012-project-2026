## Purpose

Test whether preserved nonlinear interaction scores complement tree-based fraud scores through fixed composition, with matched chronological F2 comparisons and independently verified exploratory transfer.

## ADDED Requirements

### Requirement: Separate finite preserved composition study
The study SHALL declare its complete members, transformations, weights, aliases, policies, references, gates and destinations before execution. It SHALL preserve all thirteen completed identities, separate Newton fit/analysis histories and the stopped attempt. Required model reuse SHALL verify frozen recipes, earlier training populations, preprocessing, software, numerical validity and saved inference. Wholly absent required individual caches SHALL reproduce only under their exact frozen recipes in the new destination, with actual fits counted; partial/incompatible caches SHALL be rejected. Unrelated historical grids SHALL NOT be required or rerun.

#### Scenario: Reproduce without ignored required models
- **WHEN** mandatory ancestry and thirteen original/exported selections are intact but required model caches are wholly absent
- **THEN** the study reproduces only its exact required individual recipes and leaves historical studies unchanged

#### Scenario: Encounter a partial required source cache
- **WHEN** a required model, adapter, record or prediction set is incomplete or incompatible
- **THEN** the study rejects that cache without overwriting it or silently falling back to another recipe

### Requirement: Fixed domain-correct label-free composition
The study SHALL combine declared clipped-log-odds member scores and the existing tree-fusion margin using only fixed declared weights. It SHALL retain native/transformed singles, clipping diagnostics and complete parent identities. The tree-fusion margin SHALL NOT be treated as a probability or passed through a probability logit. Invalid domains, alignment, weights, dimensions and non-finite values SHALL be rejected. Current labels, unrelated rows, learned weights and batch statistics SHALL NOT affect a row's composed score. Composed scores SHALL NOT be asserted calibrated probabilities.

#### Scenario: Compose an unbounded tree margin
- **WHEN** the valid tree-fusion margin lies outside zero to one
- **THEN** composition accepts it as a finite margin, transforms only the probability-like member and reconstructs the declared weighted sum

#### Scenario: Change labels or unrelated applications
- **WHEN** one row's member scores remain fixed while aligned labels or unrelated applications change
- **THEN** that row's composed score remains unchanged

### Requirement: Canonical historical alias and reference precedence
The already-tested additive C=0.1/unweighted/25% blend SHALL retain its original recent-fusion identity and SHALL NOT become a second selectable strategy. That identity SHALL participate in the declared matched comparisons while retaining its original reference policy. A winner with any original reference identity SHALL use the reference-winner transfer roles even if its score also has a blend representation. Other declared role aliases SHALL remain visible while sharing identical work.

#### Scenario: Select the canonical recent reference
- **WHEN** the original recent-fusion identity wins development selection
- **THEN** the study retains its original policy, freezes only the six reference-winner roles and claims no gain from renaming its identical score

### Requirement: Comparable chronological actual-F2 selection
Every declared strategy SHALL predict identical month-4/5/6 applications from earlier-only models and preprocessing. Operating policies SHALL use only eligible earlier out-of-time predictions and labels. Selection SHALL use actual macro month-5/6 positive-class F2 beta=2 with deterministic ties and no imposed review quota or precision floor. All four original reference policies and every strategy/policy result SHALL remain visible alongside supplementary metrics. AP, pooled F2, correlation or joint-error diagnostics SHALL NOT replace the macro objective.

#### Scenario: Evaluate a month-6 composition
- **WHEN** a blend produces month-6 decisions
- **THEN** its models/preprocessing use months 0-5, its policy uses only eligible month-4/5 predictions and labels, and its fixed score composition uses no month-6 labels

### Requirement: Complete matched comparisons and diagnostic evidence
The study SHALL retain all declared original-reference, own-policy, native/transformed/member/tree, representation, class-weight, regularization and blend-weight contrasts with reconstructible factors and arithmetic, including ties and negative effects. Complete clipping, diversity, original-feature, pair-feature, coefficient, numerical, resource and provenance evidence SHALL accompany the results. Diagnostics SHALL remain descriptive and SHALL NOT change the finite search, selected metric or inference rules. Equal nominal C SHALL NOT be asserted equal effective complexity across representations.

#### Scenario: Attribute a blend improvement
- **WHEN** a blend is reported as improving a comparator
- **THEN** its otherwise matched source, transformation, representation, weighting, C, alpha and policy comparisons remain available and the claim identifies its actual evaluation scope

### Requirement: Frozen bounded exploratory transfer and honest acceptance
Before further consumed-month access, the actual winner, four original reference policies, same-strategy pooled rule and predetermined winner-type controls SHALL be frozen with all dependencies and final earlier-only policies. Only the six/eleven/fourteen declared roles SHALL transfer, with deduplicated work, type/cache-specific fit bounds and actual counts. Losing-grid sweeps, hindsight cutoffs and independent-confirmation claims SHALL be excluded. Broader goal achievement SHALL require positive measured F2 differences against all four references on both development and consumed transfer; clear-gain language SHALL additionally require positive conditional interval bounds on the named scope.

#### Scenario: Transfer a new blend winner
- **WHEN** a blend outside the four original reference identities wins development selection
- **THEN** the study freezes the fourteen declared roles and exact deduplicated source dependencies before access, and other alpha roles reuse those same sources without extra base fits

#### Scenario: Fail a required reference comparison
- **WHEN** the frozen selected policy fails any required development or consumed comparison
- **THEN** the findings preserve that failure and leave the broader performance goal unachieved

### Requirement: Verified reproducible completion and retained numerical failures
Completion SHALL require checked prospective declarations, complete outputs and provenance, independently reconstructed source features/inference, compositions, earlier-only policies, metrics, factors, winner and conditional paired uncertainty, inspected figures, runnable clean-reproduction instructions and assistance attribution. Reproduced logistic fits SHALL retain the frozen numerical/resource gates without alternative-solver fallback; a failed gate SHALL preserve evidence and leave completion unavailable without changed recipes or relaxed gates. Adaptive choices, repeated search and consumed-holdout limitations SHALL be explicit. Planning or finite-search completion SHALL NOT establish a performance gain.

#### Scenario: A required reproduction fails its numerical gate
- **WHEN** an exact required logistic reproduction fails a declared numerical or allocation gate
- **THEN** the failure and completed work remain recorded, selection/transfer cannot proceed to completed-study status, and no recipe or gate is changed after outcomes

#### Scenario: Complete the finite search
- **WHEN** the experiment is declared complete
- **THEN** every requirement has verified output evidence, preservation/tests/strict OpenSpec checks pass and goal status follows measured reference comparisons rather than artifact existence
