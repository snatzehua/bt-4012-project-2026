## Why

The best development-selected score fusion improved macro F2 to 0.328907, but its consumed-month F2 of 0.372694 still trails unweighted LightGBM at 0.376884. Can training-only smooth additive feature effects improve actual recall-weighted F2, rather than continuing to rearrange the same tree scores? This follows the revised BAF proposal's imbalance/model experiments and the lecturer's request for model-specific feature justification; improvement is a hypothesis, not an expected result.

## What Changes

- Plan a separately identified spline-logistic study with positive-class F2 beta=2, no imposed review quota or precision floor, identical earlier-only populations and preserved identities for all ten completed studies.
- Compare six additive spline-logistic recipes with six matched linear-logistic controls: C in {0.01, 0.1, 1}, crossed with unweighted/balanced training. Declare feature roles, knot deduplication, missingness, unseen-category handling, extrapolation, scaling and convergence gates before fitting.
- Retain the frozen selected fusion and both original LightGBM references. Select by actual macro month-5/6 F2 under the existing pooled/latest/rank-fraction policies; preserve every result and report matched representation, regularization, weighting and policy effects.
- Freeze the selected strategy and predetermined controls before limited exploratory transfer to already-consumed month 7. Report failures against the stronger reference; no new independent confirmation is available.
- Export reproducible models, feature/knot and solver diagnostics, provenance, chronological decisions, conditional uncertainty, findings and inspected figures. All implementation tasks remain pending until real outputs are checked.
- Core scope is this finite representation experiment. Interactions, further knot/solver searches, engineered-feature expansion, new blends and additional datasets are excluded and would require a separate change.

## Capabilities

### New Capabilities

- `baf-spline-logistic-study`: auditable training-only additive spline versus linear modelling, matched actual-F2 selection and frozen exploratory transfer.

### Modified Capabilities

None. Existing studies and operating-policy requirements remain unchanged.

## Impact

Planned new modules `baf/spline_inputs.py`, `baf/spline_study.py` and `baf/spline_report.py`; new configuration `configs/spline-logistic.json`, protocol `docs/spline-logistic-protocol.md`, methodological tests and separate ignored outputs plus tracked reports under `baf-spline-logistic-v1`. Reuse existing pinned scikit-learn, SciPy and logistic machinery without dependency changes or edits to frozen modules/configs. README, AGENTS and project context will distinguish planned versus verified state. Substantive Codex assistance and official scikit-learn definitions will be attributed.
