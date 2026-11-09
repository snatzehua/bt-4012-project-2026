## Why

The user wants a better recall-weighted fraud F-score. Eighteen completed exploratory studies leave the best chronological development positive-class F2 at 0.330880962, while the latest conditional-expert study's best new strategy scores 0.324795452. Earlier CatBoost candidates used depth 6/8 and 800/1000 rounds; shallower, shorter native-categorical models are an untested, finite way to investigate whether reduced capacity improves generalization or complements the current score.

This follows the revised BAF proposal's model/imbalance questions and the lecturer's request for model-specific preprocessing and controlled comparisons. It does not assume an improvement, and does not add a review quota or precision floor.

## What Changes

- Declare a separate, adaptive analyst-selected study of twelve raw-feature CatBoost recipes: depth 3/4/6, 200/400 rounds, fixed positive class cost 1/100. Train on the same earlier populations, with matched depth, round-count and class-cost controls.
- Evaluate twelve native margin scores and thirty-six fixed margin blends with the existing bagged-anchor score, retaining six fixed operating references: 54 strategies and 162 strategy/policy cases. Use actual chronological macro F2, beta=2, as the selection objective.
- Add an isolated configuration, protocol, three implementation modules, methodological tests, miniature/persistence/resource gates, independent reconstruction and bounded winner-dependent transfer. Preserve the eighteen completed studies, original source recipes, separate Newton identities and the stopped L-BFGS attempt.
- Export all declared operating results, matched effects, score/model/work diagnostics, conditional uncertainty, checked figures and reproduction evidence, including negative results and ties. Month 7 remains consumed exploratory transfer.
- Core work includes the complete fixed grid and all declared controls. There are no optional extensions in this change; additional depths, rounds, costs, representations, calibration or tuning require another study.

## Capabilities

### New Capabilities

- `baf-shallow-catboost-study`: finite, chronological native-margin CatBoost and fixed-anchor blending study with matched capacity/cost controls, immutable ancestry, independent verification and honest F2 acceptance criteria.

### Modified Capabilities

None. Existing study requirements and frozen execution identities remain unchanged.

## Impact

Planned new files are `configs/shallow-catboost.json`, `docs/shallow-catboost-protocol.md`, `baf/shallow_catboost.py`, `baf/shallow_catboost_study.py`, `baf/shallow_catboost_report.py`, `tests/test_shallow_catboost.py` and a separate verification runner. New ignored outputs use `outputs/baf-shallow-catboost-v1/`; exported evidence uses `reports/baf-shallow-catboost-v1/`. README, AGENTS and current OpenSpec context will reflect verified progress. CatBoost 1.2.10 and the existing pinned environment suffice; no dependency upgrade, raw-data commit or historical producer modification is proposed. Planning completion is distinct from implementation, experimental completion and achievement of the broader performance goal.
