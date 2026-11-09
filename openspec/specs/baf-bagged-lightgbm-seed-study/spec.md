# baf-bagged-lightgbm-seed-study Specification

## Purpose

Evaluate randomized training-row LightGBM models and fixed seed ensembles under chronological recall-weighted evaluation, with preserved references and independently verified exploratory transfer.

## Requirements

### Requirement: Prospective isolated preserved study
The study SHALL declare its finite recipes, seeds, arithmetic, policies, references, destinations, identities and verification gates before fitting. It SHALL preserve all fifteen completed studies, separate Newton fit/analysis histories and the stopped attempt. Individual reference caches SHALL be verified against exact recipes, training/preprocessing/software identities and saved inference. Wholly absent required references SHALL reproduce only their exact individual recipes in the new destination; partial or incompatible sets SHALL fail without fallback or overwrite. Unrelated historical grids SHALL NOT be required.

#### Scenario: Find an incomplete required reference
- **WHEN** a required model, adapter, prediction or provenance set is partial or incompatible
- **THEN** the study rejects it without changing that origin or switching recipes

### Requirement: Effective reproducible randomized training
New models SHALL use the declared earlier training population and train-only original preprocessing, with effective uniform row sampling and an explicit bagging seed. Remaining source settings SHALL stay fixed. Balanced weighting SHALL use eligible whole-training class counts, not evaluation labels or a newly tuned cost. Same-data/same-seed inference SHALL reproduce after serialization; tiny integration evidence SHALL distinguish enabled sampling from an ineffective seed-only change. Actual fits, reused models, iterations, warnings and resource evidence SHALL be recorded separately.

#### Scenario: Enable a randomized source
- **WHEN** a new model is trained
- **THEN** both its row fraction and sampling frequency enable sampling, only its declared seed varies between replicas, and neither preprocessing nor training uses future labels

#### Scenario: Repeat a declared seed
- **WHEN** the same eligible data, recipe, seed and software are used again
- **THEN** predictions and saved-model inference match within the declared tolerance

### Requirement: Fixed aligned seed aggregation
Every ensemble SHALL contain exactly its three declared same-weighting seed members with equal fixed weights and declared arithmetic. Probability means SHALL validate member probabilities; logit means SHALL clip and transform each member before averaging and SHALL remain finite margins. Current labels, unrelated rows, batch statistics and learned weights SHALL NOT affect a row's composed score. Invalid domains, missing/duplicated members, alignment differences and non-finite results SHALL be rejected. Ensemble scores SHALL NOT be asserted calibrated probabilities or operational cost estimates.

#### Scenario: Change labels without changing member scores
- **WHEN** aligned labels or unrelated applications change while one row's three member scores remain fixed
- **THEN** that row's aggregate score is unchanged

### Requirement: Comparable chronological actual-F2 selection
All strategies SHALL predict identical future applications using models/preprocessing fitted on earlier months. Operating policies SHALL learn from earlier out-of-time predictions only. Selection SHALL maximize actual macro month-5/6 positive-class F2 beta=2 with deterministic ties and no review quota or precision floor. The five original reference policies SHALL remain fixed and visible. AP, pooled F2, seed diagnostics or consumed-month outcomes SHALL NOT choose a different winner.

#### Scenario: Evaluate month 6
- **WHEN** a strategy produces month-6 decisions
- **THEN** its models use months 0-5 and its operating rule uses eligible month-4/5 predictions and labels only

### Requirement: Complete matched training and ensemble effects
Every declared single seed, both aggregation forms and both training weightings SHALL remain visible with all fixed-reference, own-policy, randomized-versus-unbagged, ensemble-versus-member, aggregation and matched-weighting contrasts. Seed diversity, decision overlap, clipping and training/preprocessing/resource evidence SHALL accompany results. Ties and negative effects SHALL be retained. Uniform sampled bags SHALL NOT be described as exactly class-matched subsets, and conditional row uncertainty SHALL NOT be described as uncertainty over training seeds.

#### Scenario: Attribute an ensemble gain
- **WHEN** a seed ensemble improves a comparator
- **THEN** all three constituent results, its unbagged control, alternative aggregation and otherwise-matched weighting remain available at the named policy

### Requirement: Frozen bounded transfer and honest goal acceptance
Before further consumed-month access, the actual winner, original references, same-strategy pooled rule and predetermined member/aggregation/weighting controls SHALL be frozen with exact dependencies, earlier-only rules and cache/type-specific fit bounds. Only seven reference-winner roles or fourteen new-winner roles SHALL transfer with deduplicated work. Losing-grid sweeps, hindsight cutoffs and independent-confirmation claims SHALL be excluded. Broader goal achievement SHALL require positive measured F2 differences against all five references on both development and consumed transfer; clear-gain language SHALL additionally require positive conditional interval lower bounds on the named scope.

#### Scenario: Transfer an ensemble winner
- **WHEN** a new ensemble wins development selection
- **THEN** its fourteen roles and exact unique model/score dependencies are frozen before access and no unselected seed or recipe is evaluated outside that dependency set

#### Scenario: Fail a required reference comparison
- **WHEN** the frozen winner fails any required development or consumed comparison
- **THEN** findings retain the failure and the broader improvement goal remains unachieved

### Requirement: Independently verified reproducible completion
Completion SHALL require complete declared outputs and independent reconstruction of saved inference, preprocessing identities, aggregates, earlier-only policies, masks, metrics, contrast factors, actual winner and conditional paired uncertainty. Failed numerical/resource/persistence gates SHALL retain evidence without outcome-driven relaxation. Figures SHALL be inspected against their underlying values. Reproduction SHALL include pinned dependencies/data, fifteen selection restores that reject changes, exact individual reference reproduction, ordered execution, assistance attribution and preservation checks. Source/configuration/selection/test/access identities and closed-log report hashes SHALL be verified before synchronization/archive.

#### Scenario: Complete a negative experiment
- **WHEN** the declared finite study finishes without meeting its performance comparisons
- **THEN** verified tasks can complete while the negative result and active broader goal remain explicit

#### Scenario: Fail a required fit gate
- **WHEN** an exact required fit or persistence check fails
- **THEN** its completed work and failure evidence remain preserved and the study cannot be declared complete by silently changing the recipe
