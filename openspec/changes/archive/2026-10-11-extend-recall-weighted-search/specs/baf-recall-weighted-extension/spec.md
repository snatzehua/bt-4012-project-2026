## Purpose

Continue the unachieved recall-weighted fraud-detection goal through controlled training and ensemble comparisons while preserving failed experiments and chronological provenance.

## ADDED Requirements

### Requirement: Separate adaptive extension
The extension SHALL declare its finite candidate grid before new fitting, preserve all three completed studies' sources/configuration/software/selections, retain beta=2 and avoid imposed review quotas. It SHALL report its adaptive origin and keep month 7 out of tuning.

#### Scenario: Begin the extension after a failed search
- **WHEN** the new study starts
- **THEN** the previous zero-gain result remains intact and the new run has a distinct configuration, identity and output directory

### Requirement: Verified historical prediction reuse
Reused or reproduced chronological-window predictions SHALL have verified source/configuration identities, training-row hashes, class counts and identical future evaluation cases. Missing caches SHALL be reproducible without changing old output directories. Matched historical controls SHALL retain the original recent-K size and class-count matches.

#### Scenario: Reuse a historical-K prediction
- **WHEN** its rows, labels, training population or source identity differ from the original frozen specification
- **THEN** the extension rejects reuse rather than accepting a superficially matching filename

### Requirement: Earlier-only batch rank ensembles
Ensemble-member selection and operating-rule fitting SHALL use only prediction labels from months before the scored month. Rank ensembles SHALL normalize members using the same unlabeled application batch, document this batch requirement and preserve deterministic score ties.

#### Scenario: Score the month-6 ensemble
- **WHEN** its member and policy are selected
- **THEN** only months 4/5 labels are eligible, and month-6 labels never determine the member, normalization or operating parameters

### Requirement: Controlled F2 evidence and reproduction
The extension SHALL report matched weight/boost-duration/capacity contrasts and seed-ensemble effects, macro monthly and pooled F2, precision/recall, false positives, missed fraud, workload and paired conditional uncertainty. It SHALL reproduce the prior baseline, freeze before exploratory transfer, publish verified findings and reproduction instructions, and leave the performance goal unachieved if no gain is measured.

#### Scenario: A candidate beats development but not consumed-period baselines
- **WHEN** the frozen selected strategy transfers poorly
- **THEN** findings identify that failure and do not claim a new independent or overall performance improvement
