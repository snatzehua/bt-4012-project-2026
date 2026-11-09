# Interaction study: implementation checks, not performance results

Checked on 11 October 2026. The study remains incomplete. Tasks **1.1 and 1.3** are verified; twelve tasks remain unchecked. No new study miniature/full model fits, cached BAF model exports, selection or further month-7 access have occurred. No performance gain is established. The broader F2 goal remains active.

## Verified declaration (task 1.1)

`configs/interaction-spline-logistic.json` and `docs/interaction-spline-logistic-protocol.md` declare the three exact ordered pairs, eight recipes, twelve selectable strategies, all four original reference policies, 204 contrast factors, chronology, beta=2, seeds, numerical gates, finite counts and destinations. The source closure has exactly 43 paths. All 366 mandatory frozen hashes pass. No review quota or precision floor is imposed. The config guard rejects changed pairs, recipes, output counts, source closure and a newly imposed quota.

## Verified feature representation (task 1.3)

`baf/interaction_inputs.py` fits an unchanged `AdditiveInputs('spline')` and appends centered/scaled tensor products from the original **unscaled** marginal bases. Only training predictors determine moments, medians, knots, vocabularies and inactive flags. Stable merged centered population moments avoid subtracting large raw moments. Exactly constant columns and degenerate pairs retain dimensions and zero future contributions. There is no near-zero inactivity cutoff.

The independent auditor in `baf/interaction_report.py` fits the original main representation separately, checks saved spline knots/coefficients, evaluates direct SciPy bases and computes two-pass extended-precision moments. Explicit product-column loops independently verify left-major/right-minor ordering. It reconstructs the complete feature matrix, coefficient margins, sigmoid inference and normalized weighted objective/full gradient without calling the new pair transformation methods.

Fixtures check ordinary and degenerate/duplicate-knot data, all-missing training columns, missing future values, unseen categories, constant tails, unrelated-row/permutation invariance, the absence of a label fit API, serialization, corrupted metadata and spline state. They compare chunk sizes one, seven, 257 and 8,192 and observe the actual 8,192-row construction limit on an 8,800-row frame. Small positive variance remains active. The original main-effect matrix remains unchanged. Feature/moment tolerance is atol 1e-12/rtol 1e-10; saved inference tolerance is atol=rtol=1e-12.

## Implemented interfaces with execution tasks still open

The checked new-representation fitting/auditing interfaces delegate only compatible frozen Newton matrix functions. Analytical zero-optimum fixtures exercise persisted artifacts, independent gradients, corruption detection and preserved failure evidence without fitting an estimator. Arbitrary nonzero coefficients independently check both weighted and unweighted objective/gradient formulas at both C values. Tests reject changed predictions, population records, coefficients, gradients, fallback counts and partial caches; a retained failed gate cannot automatically refit.

Individual cache dispatch, exact additive/tree preparation, recent-fusion arithmetic, chronological operating-table exports, miniature and resource-preflight interfaces are implemented. Their full execution/audit tasks remain open. The complete development command, immutable selection/transfer stages, complete independent output audit and final uncertainty/figure/report pipeline still need implementation and verification before freezing the execution identity. No `declaration.json`, `selection.json` or `exploratory_access.json` exists in the new output directory.

## Executed checks

- `.venv/bin/python -m pytest -q tests/test_spline_interactions.py`: first 71 checks passed after correcting a bitwise floating-point assertion to the declared inference tolerance. Seven subsequent declaration/cache-dispatch checks were included in the complete suite below.
- `.venv/bin/python -m pytest -q tests`: **366 passed in 26.82s**, including all **78** new checks. The new checks fit **zero estimators**; mocked analytical artifacts are not model fits. Existing historical test modules may run their own tiny integration fixtures; those are not new study development/miniature fits.
- `.venv/bin/python -m compileall -q baf/interaction_inputs.py baf/interaction_study.py baf/interaction_report.py tests/test_spline_interactions.py`: exit zero.
- `openspec validate --all --strict --no-interactive`: **20 passed, zero failed**.
- Direct SHA-256 comparison of all **2,860** entries in the completed-study preservation snapshot: **zero changed or missing files**. The snapshot was copied to ignored `outputs/baf-interaction-spline-logistic-v1/preimplementation_preservation.json`; this is preimplementation evidence, not a frozen selection/execution declaration.

The exact new source/config/protocol/test hashes and check summaries are recorded in `implementation_audit.json`. Main specifications are unchanged and this change remains active; it must not be archived as a completed experiment.

Codex assisted with the declaration, implementation and verification above. No public notebook was copied. Acknowledge generative AI assistance and scikit-learn/SciPy software in the final course report, and keep the mathematical representation and checks explainable by the group.
