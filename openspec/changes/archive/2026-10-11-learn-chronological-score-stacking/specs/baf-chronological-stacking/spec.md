## Purpose

Evaluate learned combinations of complementary fraud-model scores under strict chronological training and matched controls, seeking auditable recall-weighted improvement while preserving prior negative results.

## ADDED Requirements

### Requirement: Separate finite recall-weighted study
The study SHALL predeclare its finite model, score-representation and policy grid in a separate configuration, retain positive-class F2 with beta=2, impose no review quota or precision floor, and preserve all completed studies' identities and negative findings. It SHALL label reused development evidence and further month-7 evaluation as adaptive and exploratory.

#### Scenario: Start after the unsuccessful extension
- **WHEN** the stacking study begins
- **THEN** its outputs have a distinct identity and destination, and the previous 0.3278 baseline and zero-gain extension remain intact

### Requirement: Chronological base and combiner predictions
Every training row for a learned score combiner SHALL contain base-model predictions produced without fitting those base models or their preprocessing on that row's label or any later month's data. The combiner and its preprocessing SHALL use only prediction rows and labels from months before the month it scores. Evaluation identities and labels SHALL align across every member and comparator, with incompatible caches rejected and absent predictions reproducible in new directories.

#### Scenario: Produce the month-4 combiner prediction
- **WHEN** the combiner is fitted to prediction rows from months 1/2/3
- **THEN** each member's month-1 scores originate from month-0 training, each later prediction follows the same earlier-only rule, and no month-4 labels train the combiner or its preprocessing

### Requirement: Earlier-only decision policies
Operating policies SHALL be fitted on earlier out-of-time combiner predictions rather than in-sample combiner outputs or current evaluation labels. Any batch percentile features SHALL depend on unlabeled scores only, preserve deterministic ties and disclose the requirement to score the application batch together.

#### Scenario: Evaluate decisions on month 6
- **WHEN** a month-6 combiner and operating policy are evaluated
- **THEN** the combiner uses prediction rows from months 1 through 5, and the policy uses only out-of-time combiner predictions from months 4/5, whose combiners were each trained on still earlier months

### Requirement: Matched controls and verified performance evidence
The study SHALL compare learned combinations against matched single-model score controls and both previous LightGBM F2 policies on identical cases. It SHALL report monthly/macro and pooled F2, precision, recall, false positives, missed fraud, workload and conditional paired uncertainty, retaining failed comparisons. Selection SHALL use chronological development only, and selected dependencies, any matched control and policies SHALL be frozen before further consumed-month transfer. If no gain is measured, the study SHALL leave the performance goal unachieved.

#### Scenario: Recalibration helps but stacking does not
- **WHEN** a learned combination improves on the old operating policy but underperforms its matched single-model control
- **THEN** the findings separate the operating/calibration effect from the combination effect and do not credit stacking with that gain

### Requirement: Auditable reproduction and honest completion
The study SHALL export exact recipes, training/prediction provenance, data/source/software identity, frozen selection, checked metrics/uncertainty, inspected figures, findings and reproduction commands. Tasks SHALL remain incomplete until outputs are verified. Finishing the declared search SHALL NOT by itself establish the performance objective or independent month-7 confirmation.

#### Scenario: An incompatible or missing prediction cache
- **WHEN** a source prediction has changed identity or is unavailable
- **THEN** the changed cache is rejected, and an unavailable cache can be refitted under the frozen recipe in the new study's destination without modifying historical outputs
