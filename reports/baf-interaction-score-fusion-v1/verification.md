# Checked interpretation and completion evidence

The development-selected policy is the fixed 10% **additive C=1, balanced spline** / 90% existing tree-fusion margin, with the latest-month threshold policy. Its strategy ID starts with `interaction_blend_`, but its source is additive. Selection remains unchanged after consumed transfer.

Macro chronological F2 improves from the recent reference's **0.329010 to 0.330793**. All four development point differences are positive. The intervals against the recent fusion, tree fusion and balanced LightGBM include zero; only the development comparison against unweighted LightGBM has a positive conditional lower bound.

On already-consumed month 7, selected F2 is **0.371534**, precision **19.49%**, recall **48.04%**, with **686** frauds caught, **742** missed and **2,834** legitimate applications flagged. It beats the recent-fusion and balanced references as point estimates, but trails tree fusion and unweighted LightGBM. Every original-reference consumed interval includes zero. The broader performance goal therefore remains unachieved.

| Reference | Development F2 difference (95% conditional interval) | Consumed-month difference (95% conditional interval) |
|---|---:|---:|
| recent_fusion | +0.001783 [-0.001322, +0.005006] | +0.001286 [-0.003943, +0.005301] |
| fusion | +0.001887 [-0.001847, +0.006004] | -0.001160 [-0.004637, +0.001898] |
| balanced | +0.002992 [-0.001442, +0.008094] | +0.007125 [-0.000200, +0.013949] |
| unweighted | +0.008140 [+0.003097, +0.013272] | -0.005351 [-0.014823, +0.003437] |

The same blend with its predeclared pooled policy scores consumed F2 **0.374974**, above the selected latest policy but still below unweighted LightGBM. Its development F2 was lower, so it cannot replace the winner after seeing month 7. Other predeclared matched controls remain visible, including higher consumed point estimates; none authorize hindsight selection. Score composition and policy transfer both matter, and these data do not isolate a universal causal explanation for their differences.

The complete development evidence includes 33 checked base reuses, zero new full development fits, three canonical derived reuses, 156 scores, 306 forward rows and 1,386 independently reconstructed contrasts. Bounded transfer checked eight models and eighteen scores for fourteen frozen roles: three new fits (one interaction and two additive) and five base reuses. No numerical failures, warnings or solver fallbacks occurred in the required new full fits. The 619-test receipt matches the frozen code/config/software/test identity; the focused/full invocations made eight new tiny cache-format test fits, separate from historical tests. The source eight-fit miniature gate was reused, with zero new production miniature fits.

All 1,000 development and 500 consumed paired fixed-decision draws were independently reconstructed by the frozen reporting implementation; all 39 development and 13 consumed intervals were checked again after CSV round-trip. The three-panel operating figure was inspected for readability. Its 153 scatter points, 32 matched-alpha points and two tree-reference lines were checked against the audited tables, and independent regeneration produced identical PNG bytes. No extra alpha endpoints were evaluated. All 174 score populations were directly compared with raw monthly IDs and labels.

These intervals condition on fixed models/policies. They exclude adaptive source/C/alpha/pair choices, repeated study search, selection, retraining, temporal/generative uncertainty and multiplicity correction. Month 7 has already been consumed and supplies no independent confirmation. Analyst-selected settings remain open to group review.

The pinned Python 3.12.9 environment, all 27 package versions, compatibility check and Base v2 SHA256 were verified. The README's exact thirteen-selection restore script was executed against unchanged existing records; it checks hashes before any copy and rejects changed records. Four genuine tiny absent-cache format tests cover exact required reproduction without rerunning unrelated historical grids. Ordered full test/receipt, resource preflight, development preparation, validation with independent audit, freeze, transfer and report stages have all completed. Never overwrite the completed frozen run with those commands.

See `reporting_verification.json`, `population_identity_verification.json`, `transfer_verification.json`, `environment_reproduction_verification.json`, the original test receipt and the final completion audit/manifest for evidence. Codex assisted with planning, implementation, execution, numerical/feature/metric/provenance verification, uncertainty, figure inspection and interpretation using preserved repository interfaces and pinned scikit-learn/SciPy/XGBoost/LightGBM APIs. No public notebook was copied.
