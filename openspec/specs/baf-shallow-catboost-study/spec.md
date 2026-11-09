# baf-shallow-catboost-study Specification

## Purpose

Evaluate a finite set of shallower, shorter native-categorical fraud models and fixed margin blends using chronological recall-weighted selection, matched controls, immutable ancestry and independently verified results.

## Requirements

### Requirement: Prospective isolated preservation and exact source reuse
The study SHALL declare all recipes, feature roles, arithmetic, policies, contrasts, transfer cases, gates, destinations and acceptance criteria before new evaluation. It SHALL preserve all eighteen completed studies, separate Newton fit/analysis identities and the stopped attempt, with eighteen identical original/exported selection pairs. Required individual reference sources SHALL retain their exact earlier populations, fitted preprocessing, recipes, software and saved inference. Wholly absent required sets SHALL reproduce only their exact individual recipes in the new destination under unchanged gates. Partial, corrupt or incompatible inspected sets SHALL fail without overwrite, silent fallback or unrelated historical-grid execution. Cache presence alone SHALL NOT prove source validity.

#### Scenario: Inspect an incomplete required model set
- **WHEN** an inspected source has a prediction or provenance record but lacks a required serialized model or other member
- **THEN** execution rejects that set and retains it without silently searching another origin

#### Scenario: Reproduce an entirely absent reference
- **WHEN** all members of a required individual source set are absent and mandatory ancestry is valid
- **THEN** only its original recipe is reproduced in the new destination and its actual fit is counted separately

### Requirement: Earlier-only native feature handling
Every model SHALL train on applications strictly earlier than its evaluation month and predict the same complete future population as its comparators. New native models SHALL use the original 29 row-level predictors, excluding target, month and the constant device-fraud-count field. Sentinel decoding, numerical medians, feature order and categorical treatment SHALL be explicit and fitted only on eligible training rows. Missing categorical values SHALL receive a fixed token; unseen categories SHALL use native inference without evaluation-time fitting. Current labels, row order or unrelated future rows SHALL NOT change a row's representation or score.

#### Scenario: Score an unseen or missing category
- **WHEN** a future application has a categorical value unseen during training or a missing value
- **THEN** it receives a finite native score using the fixed training representation without learning from that application or its label

### Requirement: Finite matched model recipes and valid persistence
The new grid SHALL contain exactly twelve recipes crossing depths 3/4/6, 200/400 rounds and positive-class costs 1/100. All other declared settings and earlier training populations SHALL remain matched. Costs SHALL be applied exactly once with negative-class cost one; cost 100 SHALL NOT be described as exact balanced weighting. There SHALL be no early stopping, evaluation-label-driven best iteration, undeclared tuning or duplicate cost mechanism. Requested/effective parameters, actual rounds, warnings, training identities and saved inference SHALL be verified. Native model and adapter persistence SHALL reproduce scores to absolute tolerance 1e-12 with zero relative tolerance; failures SHALL retain evidence without relaxing recipes.

#### Scenario: Train the weighted short model
- **WHEN** a depth-4, 200-round, positive-cost-100 recipe is fitted
- **THEN** it uses exactly those settings and one class-cost mechanism, and its persisted inference passes the same gates as its positive-cost-one control

#### Scenario: Reject duplicated class costs
- **WHEN** a native recipe combines its fixed class costs with sample weights or another class-weight mechanism
- **THEN** execution rejects the undeclared recipe before fitting and retains the invalid declaration as failure evidence

### Requirement: Canonical finite margins and fixed blends
The selectable union SHALL contain six original references, twelve native margin strategies and thirty-six fixed anchor blends: 54 strategies and 162 strategy/policy cases. New stored scores SHALL be finite raw margins, with blends evaluated as `(1-alpha)*anchor + alpha*native_margin` in declared member order at alpha 0.10/0.25/0.50. Zero and unit endpoints SHALL alias the original anchor and pure native strategy. New selectable scores and compositions SHALL NOT use sigmoid, logit, clipping, batch normalization, calibration or learned composition. Original reference domains SHALL remain unchanged, and margins SHALL NOT be called calibrated fraud probabilities. Misaligned populations, invalid domains or nonfinite arithmetic SHALL fail.

#### Scenario: Display a blend endpoint
- **WHEN** a diagnostic curve includes native weight zero or one
- **THEN** it uses the existing anchor or pure native strategy without a new candidate, fit or renamed gain

### Requirement: Actual chronological F2 selection without a quota
The pooled, latest and learned-rank-fraction policies SHALL use only eligible earlier out-of-time predictions and labels. Selection SHALL maximize actual positive-class macro F2 beta=2 on months 5/6, followed by macro recall, macro precision and deterministic identity ordering. All six reference operating policies SHALL remain fixed for reference comparisons. AP, pooled F2, hindsight thresholds, a 1% review quota, an imposed precision floor and consumed-month outcomes SHALL NOT replace the objective or winner. Learned rank fractions SHALL derive from earlier F2 optimization rather than a fixed review budget.

#### Scenario: Decide for month six
- **WHEN** a strategy is evaluated on month 6
- **THEN** its model uses months 0-5, pooled/rank policies use prediction months 4/5 and latest uses month 5, with month-6 labels excluded from fitting and decisions

### Requirement: Complete matched effects and work diagnostics
The study SHALL retain all 183 full development scores, 324 forward operating rows, 162 macro and pooled summaries, and 1728 unique declared contrast identities, including negative effects and ties. Effects SHALL cover all fixed references, own pooled policy, all depth pairs, round count, class cost, native-versus-anchor, blend-versus-native/anchor and higher interior blend weights. Score, preprocessing, model, diversity, decision-overlap and actual-work diagnostics SHALL be independently reconstructed. Precision, recall, F1/F2/F3, confusion counts, missed fraud, legitimate flags, flagged fractions, AP and ROC AUC SHALL accompany operating results. Nominal depth/round comparisons SHALL NOT be claimed to equalize runtime, stored trees, categorical statistics or inference cost.

#### Scenario: Observe a shallow-model advantage
- **WHEN** a new shallow strategy outperforms a deeper strategy
- **THEN** the comparison retains the same round count, class cost, blend weight and policy, with all corresponding effects and actual work visible

### Requirement: Frozen bounded exploratory transfer
Before further consumed-month scoring, the actual winner, six references, same-strategy pooled policy, predetermined winner-type controls, exact deduplicated dependencies, earlier-only rules and actual cache/type-specific fit bounds SHALL be frozen. Reference winners SHALL retain eight visible roles and thirteen scores with no new native fits. Native or blended winners SHALL retain twenty-two visible roles and exactly five required native recipes, with at most twenty-one or twenty-five scores respectively. All 162 prospective winner/policy cases SHALL be checked. Visible aliases SHALL share work. Losing-grid sweeps, hindsight consumed-month policies, winner replacement and independent-confirmation claims SHALL be excluded.

#### Scenario: Transfer a blended winner
- **WHEN** a blended strategy wins chronological development selection
- **THEN** only its frozen twenty-two roles, five required native recipes, eight reference bases and deduplicated score dependencies are transferred to the already-consumed month

### Requirement: Independent gates and reproducible completion
Completion SHALL require valid actual-host resource evidence, unchanged source gates, a fourteen-fit new native miniature gate, independently audited saved models and exact reconstruction of preprocessing, parameters, scores, arithmetic, policies, masks, metrics, complete factor sets, selection and work bounds. All 1500 paired fixed-decision bootstrap draws and role-specific intervals SHALL be independently reconstructed. Figures SHALL be verified against plotted values and visually inspected. Reproduction SHALL include pinned data/software, eighteen exact selection restores rejecting changes, whole-absence source handling, clean-copy methodological tests, separate current/historical/production fit accounting, ordered commands and assistance attribution. Frozen execution/access identities, preserved history and closed-log run/report manifests SHALL be checked before synchronization/archive.

#### Scenario: Fail a miniature or numerical gate
- **WHEN** a recipe fails effective-parameter, repeatability, warning, persistence, numerical or resource checks
- **THEN** execution retains completed work and failure evidence without relaxing the gate or claiming the study complete

#### Scenario: Reproduce without historical caches
- **WHEN** the documented setup and complete methodological suite run in a disposable copy without prior model/prediction caches
- **THEN** mandatory selection restores, changed-record rejection and actual tiny-format tests pass, while full-size retraining is claimed only if actually executed

### Requirement: Measured acceptance and honest uncertainty
Broader performance achievement SHALL require strictly positive measured selected-minus-reference F2 differences against all six fixed references on both macro development and consumed exploratory transfer. Aliases and exact ties SHALL NOT count as gains. Clear-gain language SHALL additionally require positive conditional interval lower bounds in the named scope. Findings SHALL disclose adaptive recipe/source/policy/selection search, synthetic-data limitations, repeated consumed-month access and interval exclusions for retraining, temporal/generative variation and multiplicity. Completing the finite study SHALL NOT itself satisfy the broader performance goal.

#### Scenario: Finish a verified negative experiment
- **WHEN** all required experimental outputs pass verification but any required performance comparison fails
- **THEN** the finite study can be completed and archived while the failed comparison and unachieved broader goal remain explicit
