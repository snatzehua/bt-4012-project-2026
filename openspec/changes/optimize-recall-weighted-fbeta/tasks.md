## 1. Separate reproducible protocol

- [x] 1.1 Add configs/recall-weighted.json and docs/recall-weighted-protocol.md with the declared beta, finite candidate/policy grid, temporal boundaries, seed, preservation identities and output paths; verify exact candidate counts and agreement with design/specs before fitting.
- [x] 1.2 Implement additive run-identity, preservation and cached-prediction checks; verify modified source/config/selection identities and mismatched row IDs or labels are rejected, and both prior studies remain unchanged.

## 2. Metrics and chronological operating policies

- [x] 2.1 Implement generic F-beta metrics, pooled/latest threshold and learned-rank-fraction policies; verify direct confusion arithmetic, F2 agreement with sklearn, ties, rounding, zero cases and beta validation with methodological tests.
- [x] 2.2 Implement forward policy evaluation and earlier-only blend-member selection; verify month 5 uses only month 4 prediction labels and month 6 only months 4/5, with tests that fail on ineligible sources.
- [x] 2.3 Implement checkpointed new model fitting and safe read-only reuse of eligible old predictions; verify training boundaries, predictor exclusions, adapters and exact evaluation-case alignment in a smoke run.

## 3. Development search and selection

- [x] 3.1 Run the declared development grid and policies; verify every required model/month result exists or has validated reuse provenance, and recompute monthly/macro F2 and confusion counts from predictions.
- [x] 3.2 Export matched training-weight/capacity, policy-only and blend comparisons against old F2 baselines; verify the old balanced/unweighted macro F2 values reproduce and retain unsuccessful rows.
- [x] 3.3 Freeze the development-selected model/policy, final dependency members and fitted policy parameters with data/source/config/software identity; verify deterministic selection, preservation checks and freeze-before-further-month-7-access ordering.

## 4. Exploratory transfer and evidence

- [x] 4.1 Score only frozen selected dependencies and baseline comparators on consumed month 7; verify unchanged selection identity and identical cases, and label every resulting comparison exploratory.
- [x] 4.2 Compute paired conditional F2 uncertainty at fixed policies, supporting metrics and precision/recall/workload figures; verify bootstrap pairing, recomputed metric agreement and rendered plot legibility.
- [x] 4.3 Write reports/baf-recall-weighted-v1/findings.md and audit.json distinguishing measured development gain, month-7 transfer, policy versus training effects, failures and limitations; verify every numerical claim against exported tables and keep the performance goal unachieved if evidence does not establish a gain.

## 5. Reproduction and completion checks

- [x] 5.1 Update README commands, AGENTS, OpenSpec context and assistance acknowledgment to reflect actual implemented outputs; verify the documented command sequence and preserved prior-study hashes.
- [x] 5.2 Run relevant methodological tests and openspec validate --all --strict --no-interactive; inspect required outputs and synchronize current requirements only after implemented behavior is verified, retaining incomplete task status for anything missing.
