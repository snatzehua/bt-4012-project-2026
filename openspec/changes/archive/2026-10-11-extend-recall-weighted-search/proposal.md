## Why

The dedicated F2 study completed 30 new fits and 108 forward policy evaluations without improving the prior macro F2 of 0.3278. The performance goal remains unachieved: test stronger minority-class emphasis, shorter boosting and the already-available chronological training-window predictions, rather than claim that a completed search is a gain.

This continues the revised BAF proposal's imbalance/window/threshold questions and retains the lecturer's matched-size historical controls. It is an adaptive exploratory follow-up motivated by development failure, not new confirmation from consumed month 7.

## What Changes

- Preserve all three completed studies and run a separate, finite F2 extension without a review quota or precision floor.
- Test a native LightGBM grid crossing positive weights 1/30/100, leaf counts 15/31 and boosting rounds 200/600, with matched controls and training-only preprocessing.
- Reuse or reproduce the original LightGBM recent-K and class/size-matched historical-K predictions, including three-seed historical means, on identical chronological cases.
- Compare score-percentile ensembles with the balanced baseline to avoid mixing incompatible raw class-weight score scales; select members and operating policies from earlier development predictions only.
- Export failed and successful comparisons, F2/precision/recall/workload, conditional uncertainty, preservation audits, a frozen selection and exact reproduction instructions.

## Capabilities

### New Capabilities

- `baf-recall-weighted-extension`: Adaptive F2 search across stronger training weights, boosting duration, verified historical predictions and comparable batch rank ensembles.

### Modified Capabilities

None. The completed F2 study and its negative result remain unchanged.

## Impact

Add a separate configuration/protocol, additive extension module(s), methodological tests, ignored outputs and a small findings export. Existing modelling and metric modules are imported read-only. No new dataset, report submission, raw-data commit or independent month-7 claim is included. New fits are finite: 12 presets across three development months (36 fits), plus only frozen selected dependencies for exploratory transfer; existing predictors are reused.
