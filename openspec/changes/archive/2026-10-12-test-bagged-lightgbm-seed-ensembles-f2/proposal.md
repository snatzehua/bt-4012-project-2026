## Why

The completed fixed-addition study left the existing development winner unchanged at F2 0.330793 and did not beat unweighted LightGBM on consumed month 7. This separate experiment tests whether randomized training rows and averaging independently fitted seeds improve F2, continuing the revised BAF proposal's model/imbalance investigation and lecturer feedback on justified model-specific choices; no improvement is assumed.

## What Changes

- Train the original 200-round, 15-leaf LightGBM recipe with only row bagging enabled, using balanced/unweighted class weights and three declared bagging seeds. Retain identical eligible training rows, original train-only preprocessing and unchanged remaining tree parameters.
- Compare every single seed, equal probability means and equal clipped-logit means against five frozen reference policies. Select actual macro chronological positive-class F2, without review quota or precision floor. This changes the source models, rather than recombining the same deterministic predictions again.
- Declare 18 new development fits, 15 individual reference-base results, 54 physical scores, 15 selectable strategies, 90 forward operating rows and complete matched seed/weighting/aggregation controls before fitting. Record reference fits/reuses separately.
- Freeze only the actual winner and predetermined reference/member/aggregation controls before any bounded exploratory month-7 transfer. Preserve conditional uncertainty, negative outcomes, exact training/preprocessing/software identities and all fifteen completed studies plus the stopped attempt.
- Export independently checked results, provenance, figures, findings and reproduction instructions. A finite negative result leaves the broader improvement goal active.

## Capabilities

### New Capabilities

- `baf-bagged-lightgbm-seed-study`: randomized-row LightGBM training, fixed seed ensembles, chronological actual-F2 controls and bounded exploratory transfer.

### Modified Capabilities

None. Previous study requirements and frozen outputs remain unchanged.

## Impact

Planned isolated implementation: `configs/bagged-lightgbm-seeds.json`, `docs/bagged-lightgbm-seeds-protocol.md`, `baf/bagged_boosting.py`, `baf/bagged_boosting_scores.py`, `baf/bagged_boosting_report.py` and methodological tests. Destinations: `outputs/baf-bagged-lightgbm-seeds-v1/` and `reports/baf-bagged-lightgbm-seeds-v1/`. The previous 49-source closure gains three modules; existing pinned dependencies suffice. Implementation updates README/current context and assistance attribution. No feature, tree-size, row-fraction or seed search, learned ensemble weights, new constraints or losing-grid month-7 evaluation is included. Settings are analyst-selected and adaptive; month 7 remains consumed.
