## Why

The chronological stack improved average precision in both forward months but failed to improve F2 0.3278. Test whether fitting probability calibration on earlier predictions and choosing a batch decision from a smooth F2 estimate improves the actual recall-weighted metric; better ranking alone does not establish that result.

This continues the revised BAF proposal's operating-threshold question and the user's recall-weighted performance goal. It is an adaptive follow-up on reused development data, not independent month-7 confirmation.

## What Changes

- Preserve all five completed studies and declare a separate finite policy study with beta=2, no review quota and no precision floor.
- Reuse three fixed chronological score sources: balanced LightGBM, unweighted LightGBM and the best four-member stack from the completed development search. Reproduce missing predictions under their frozen recipes in new directories.
- Fit earlier-only sigmoid-logit or isotonic calibration using pooled or latest eligible prediction months. Choose the current unlabeled batch's flagged count using a plug-in F2 surrogate, explicitly distinguished from exact expected F2.
- Compare the four calibrated batch policies with the three existing empirical F2 policies, keeping identical cases, source boundaries and baseline reproduction.
- Export calibration/decision diagnostics, measured F2/precision/recall/workload, conditional intervals, preservation audits and reproduction instructions. Label any current-month optimal ranking-prefix calculation as an optimistic diagnostic, compute it only after selection freeze, and exclude it from selection.

## Capabilities

### New Capabilities

- `baf-calibrated-f2-policy`: Earlier-only calibration and unlabeled batch decisions for recall-weighted detection, with explicit surrogate limitations and empirical-policy controls.

### Modified Capabilities

None. Earlier sources, requirements, choices and negative results remain unchanged.

## Impact

Add a separate configuration/protocol, additive calibration/policy runner and report, methodological checks and small findings exports. With current caches, reuse nine model/month score sources and fit 24 development calibrators, producing 42 forward operating comparisons. No new base or combiner development fit is initially needed. Missing caches remain reproducible under the existing recipes in the new destination. Further consumed-month transfer uses only the frozen selected policy/model, its same-model empirical comparator and both old baselines. No new dependency, raw-data commit, submission deliverable or independent holdout claim is included.
