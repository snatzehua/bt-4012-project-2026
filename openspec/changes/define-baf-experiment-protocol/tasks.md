## 1. Confirm data and scope

- [x] 1.1 Review official BAF metadata, select the main variant and document the variable inventory and missing-value conventions.
- [x] 1.2 Reconcile the revised proposal and feedback into core versus optional experiments; record model and imbalance-treatment choices and rationale.

## 2. Specify controlled experiments

- [x] 2.1 Define chronological folds, final holdout and label-availability assumptions; check feasibility using metadata or a documented local profile.
- [x] 2.2 Select multiple feasible K values, historical-pool eligibility, sampling seeds, class-composition controls and monthly tie handling.
- [x] 2.3 Document model-specific preprocessing, training-only fitting and comparable tuning budgets.
- [x] 2.4 Fix metric definitions, review budgets, rounding, score-tie rules, threshold selection and uncertainty summaries.

## 3. Publish and verify the protocol

- [x] 3.1 Write `docs/experiment-protocol.md` with decisions, configuration, limitations and AI/external-source acknowledgment notes.
- [x] 3.2 Check the protocol against each acceptance scenario and resolve execution-critical open questions; retain evidence of the check.
- [x] 3.3 Link the finalized protocol from README, synchronize affected project guidance and run `openspec validate --all --strict --no-interactive`.
