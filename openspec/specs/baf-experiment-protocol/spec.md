# BAF Experiment Protocol Specification

## Purpose

Specify the documented experiment protocol needed to implement interpretable BAF comparisons motivated by the revised proposal and feedback.

## Requirements

### Requirement: Data and feature inventory
The protocol SHALL name the chosen BAF variant and cite official metadata. It SHALL document target, time field, predictors, missing-value conventions and exclusions, and justify preprocessing for each selected model family.

#### Scenario: Prepare model inputs
- **WHEN** the protocol defines logistic regression and tree-based baselines
- **THEN** it specifies their categorical encoding, numerical scaling and missing-value handling and requires transformations to be fitted only on training data

### Requirement: Chronological evaluation
The protocol SHALL specify training cutoffs, forward validation periods and an untouched final test period. Compared strategies SHALL predict the same future applications. Model, preprocessing, threshold and experiment choices SHALL be frozen before final testing.

#### Scenario: Compare strategies at a cutoff
- **WHEN** recent and historical models are evaluated
- **THEN** both use only eligible past training information and the same future evaluation cases, with label-availability assumptions stated

### Requirement: Matched-size recency controls
The protocol SHALL compare most-recent-K and historical-K training sets for multiple feasible K values and include an expanding-history reference. It SHALL specify historical-pool eligibility, sampling seeds, class-composition policy and a target-independent rule for month-boundary ties.

#### Scenario: A requested K is infeasible
- **WHEN** the eligible pool cannot supply K rows or the required class counts
- **THEN** the configuration is recorded as infeasible rather than silently changing size or composition

#### Scenario: Interpret recency effects
- **WHEN** recent and historical samples have equal size but different class proportions
- **THEN** the analysis states that class composition remains an uncontrolled difference and does not attribute the full performance difference to recency alone

### Requirement: Defined evaluation and tuning policies
The protocol SHALL define imbalance treatments, model/tuning budgets, metric conventions, fixed review capacities, rounding and score-tie rules. It SHALL include precision, recall, F1, PR-AUC or average precision, confusion matrices, budget metrics, missed fraud and legitimate applications flagged. Any random-split benchmark SHALL be labelled diagnostic.

#### Scenario: Report performance at review capacity
- **WHEN** applications are ranked for a stated review budget
- **THEN** the protocol defines the exact number selected and requires detection and false-positive outcomes on identical evaluation cases

### Requirement: Reproducible protocol and scope
The protocol SHALL record splits, seeds, dataset provenance, proposed run configuration, uncertainty summaries and compute limits. It SHALL identify core versus optional experiments, unresolved choices, limitations and assistance sources. Execution-critical settings SHALL be resolved before pipeline implementation.

#### Scenario: Protocol is ready for implementation
- **WHEN** the protocol is finalized
- **THEN** another group member can identify the precise planned comparisons and configuration without choosing unstated settings, and README links to the protocol
