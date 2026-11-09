## Why

The user requested EDA and empirical tests of the revised BAF proposal. The lecturer feedback requires comparisons that separate training recency from size and justify preprocessing.

## What Changes

- Implement the documented first-study protocol using development-only EDA, chronological experiments, matched class/size controls and a labelled random diagnostic.
- Freeze decisions before month-7 evaluation; export audit records, conditional uncertainty intervals, figures and a readable findings report.
- Add a pinned environment, meaningful methodological tests and README reproduction commands.

## Capabilities

### New Capabilities
- `baf-study`: Reproducible, auditable EDA and fraud-model hypothesis testing.

### Modified Capabilities
None.

## Impact

Adds Python modules, configuration, dependencies, tests and small report artifacts. Raw datasets, predictions and run checkpoints stay ignored. The study is an initial empirical evaluation, not a final submission or deployable fraud model. See `docs/experiment-protocol.md` for the frozen study design and limitations.
