## Why

The revised BAF proposal establishes the research direction, but implementation needs a concrete protocol. `feedback.md` identifies a confound between training recency and sample size and asks for feature/preprocessing justification. Resolving these before model development will make the comparisons interpretable.

## What Changes

- Produce `docs/experiment-protocol.md` documenting data selection, actual variables, model-specific preprocessing, chronological splits, training-set comparisons, imbalance treatments, tuning and metrics.
- Define most-recent-K versus historical-K controls at several feasible K values, alongside an expanding-history reference.
- Record unresolved choices and their resolution; identify core experiments and optional diagnostics from the broader proposal.
- Define reproduction and final-test rules before implementation.

The protocol is implemented in `docs/experiment-protocol.md`, with executable settings in `configs/study.json` and acceptance evidence in `reports/baf-study-v1/protocol-checks.json`. Model implementation and measured findings are tracked separately in `run-baf-eda-and-hypothesis-tests`.

## Capabilities

### New Capabilities
- `baf-experiment-protocol`: A documented, leakage-aware and reproducible protocol for controlled BAF comparisons.

### Modified Capabilities
None.

## Impact

Adds the research protocol and README links. It fixes first-study choices for Base v2, forward validation, matched-K sampling, model presets and review capacities. These analyst-selected choices are open to group review; amendments after final-test access must preserve the consumed-holdout record.
