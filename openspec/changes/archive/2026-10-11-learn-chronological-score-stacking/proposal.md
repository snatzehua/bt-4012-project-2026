## Why

The two dedicated recall-weighted searches have not improved macro chronological F2 0.3278; the latest 36-fit extension retained the old baseline. Test whether learning a combination of complementary model scores improves F2, with single-model controls to distinguish stacking from recalibration and threshold effects.

This continues the revised proposal's model/imbalance/threshold questions and the lecturer's request for model-specific feature justification. It is an adaptive follow-up on reused development periods, not independent confirmation from already-consumed month 7.

## What Changes

- Add a separate finite study using positive-class F2 with beta=2, no imposed review quota and no precision floor; preserve all four completed studies and their failed comparisons.
- Generate strictly chronological early-month predictions for four fixed, existing model recipes, then train score combiners only on predictions made before their labels were used in base-model fitting.
- Compare learned four-model combinations with matched balanced-LightGBM-only score controls, two score representations and fixed regularization/class-weight presets. Evaluate both old baseline policies on identical future applications.
- Fit each decision policy on earlier out-of-time combiner predictions, so neither base training, combiner training nor threshold fitting uses the scored month's labels.
- Freeze development-selected recipes and policies before exploratory transfer; export direct F2/precision/recall/workload comparisons, conditional uncertainty, provenance audits, negative results and reproduction instructions.

## Capabilities

### New Capabilities

- `baf-chronological-stacking`: Leakage-controlled learned score combinations and matched single-model controls for the recall-weighted detection objective.

### Modified Capabilities

None. Existing studies remain frozen and their main requirements remain intact.

## Impact

Add a new config/protocol, additive stacking runner/report modules, methodological tests and small findings exports. Reuse pinned dependencies and compatible existing predictors read-only; absent caches are refitted into the new study's directories. With current caches, the declared development search needs 12 early base-model fits, 15 verified base prediction reuses and 72 combiner/control fits. Further month-7 access uses only the frozen selected strategy, its matched single-member control when a stack is selected, their dependencies and the two baseline comparators. No new dataset, raw-data commit, submission deliverable or independent holdout claim is included.
