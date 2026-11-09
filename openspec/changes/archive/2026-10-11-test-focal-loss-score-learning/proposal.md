## Why

Earlier model, stacking and calibration studies did not improve recall-weighted F2 0.3278. The fixed-ranking diagnostic leaves little cutoff-only headroom, so test whether focal-loss learning improves the ranking and actual chronological F2 rather than another threshold adjustment.

This follows the revised BAF proposal's class-imbalance question and the user's performance goal. It is an adaptive experiment on reused development evidence; the method's detection-paper results do not establish a fraud-data gain.

## What Changes

- Preserve six completed studies and declare a separate finite native-LightGBM loss study, retaining positive-class F2 beta=2 with no quota or precision floor.
- Compare four focal presets, two custom ordinary-logistic-loss controls and two built-in logistic-loss controls with matched training data, tree parameters and initialization.
- Verify raw-margin derivatives, numerical stability and custom-versus-built-in integration on controlled fixtures before interpreting focal comparisons.
- Compare all eight new presets with the old balanced/unweighted LightGBM and native31 references on identical forward cases under three existing earlier-only F2 policies.
- Freeze selection before further exploratory consumed-month transfer; export checked actual metrics, matched loss contrasts, fixed-decision uncertainty, provenance, failures and reproduction instructions.

## Capabilities

### New Capabilities

- `baf-focal-loss-study`: Audited custom-loss score learning with ordinary-loss controls and chronological recall-weighted evaluation.

### Modified Capabilities

None. Existing study sources, requirements, configurations, selections and results remain frozen.

## Impact

Add a new configuration/protocol, loss/runner/report modules, methodological tests and small checked exports. The finite grid requires 24 new development fits and nine verified reference results, with missing references reproducible in new directories, producing 66 forward policy rows. Use the pinned existing environment; add no dependency or raw dataset. There is no promised gain, new submission artifact or independent month-7 holdout.
