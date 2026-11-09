## Why

Nineteen completed studies leave chronological positive-class F2 at 0.330880962. Controlled removal of application-velocity and local-count predictors remains untested: these features shift across months and might hurt future detection or add useful fraud information, which a matched ablation can distinguish.

This follows the revised BAF proposal's temporal-drift question and the lecturer's request for model-specific feature justification. Distribution shift alone does not establish harmful concept drift or justify a performance claim.

## What Changes

- Plan a fixed two-factor removal experiment using the original LightGBM representation, populations and 200-round recipe: retain both groups, remove velocity only, remove local counts only, or remove both; compare original balanced and unweighted costs.
- Reuse the two full-feature baseline models, fit six reduced-feature recipes on three earlier training populations (18 new full development fits), and retain all six fixed operating references. Evaluate eight pure native margins and twenty-four fixed margin blends with the current anchor: 38 strategies and 114 strategy/policy cases.
- Independently verify feature exclusion, training-only preprocessing, underlying saved-model inference, matched effects, actual F2 policies, work and uncertainty. Freeze only predetermined winner-dependent transfer roles before further consumed-month access.
- Preserve all nineteen completed studies, separate Newton histories, the stopped attempt and original report/run identities. Snapshot mutable project context before future documentation edits so historical manifests remain verifiable as historical versions.
- Export negative results, exact controlled comparisons, conditional uncertainty, checked figures, fit ledgers and reproduction commands. There are no optional extensions; different groups, engineered replacements or hyperparameters require another study.

## Capabilities

### New Capabilities

- `baf-context-feature-ablation-study`: finite feature-group removal and fixed-margin composition study with chronological F2 selection, matched full-feature controls, immutable ancestry and independently checked completion.

### Modified Capabilities

None. Earlier requirements and frozen scientific identities remain unchanged.

## Impact

The apply phase will add `configs/context-feature-ablation.json`, `docs/context-feature-ablation-protocol.md`, `baf/context_ablation.py`, `baf/context_ablation_study.py`, `baf/context_ablation_report.py`, methodological tests and a separate fit-accounting runner. New outputs will use ignored `outputs/baf-context-feature-ablation-v1/`; exports will use `reports/baf-context-feature-ablation-v1/`. The current pinned Python/LightGBM environment suffices. This proposal creates planning artifacts only; no new model, score, configuration or protocol is implemented here. Settings are analyst-selected, F2 keeps beta=2 without quota/floor, month 7 remains consumed, and planning completion does not achieve the performance goal.
