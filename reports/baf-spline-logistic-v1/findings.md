# Additive spline-logistic execution stopped at convergence gate

This attempt is **incomplete** and produced no selected model or chronological macro F2 result. It does not achieve the performance goal. The declared full-data L-BFGS gate stopped execution on month 4 for `spline_c0p1_balanced`: the optimizer emitted a ConvergenceWarning after reaching the fixed 2,000-iteration limit. The model was not saved as a valid result, no selection was frozen, and month 7 was not accessed by this attempt.

## Actual execution

- Nine new full-data fits completed: all six month-4 linear controls and three month-4 spline recipes (`C=0.01` unweighted/balanced and `C=0.1` unweighted).
- One additional attempted fit failed its numerical gate. The 26 other declared full-grid fits were not attempted. The failed fit is not removed from the intended grid or replaced with a converged setting.
- Nine earlier base model/prediction results were verified and reused, with three fixed reference fusion scores. There were no new reference fits.
- The separate miniature preflight completed 24 deterministic converged fits, demonstrating that passing a small fixture does not prove full-data convergence.
- The full methodological suite passed 192 tests in 7.73 seconds, compileall passed and strict OpenSpec validation passed 17 items. These checks do not substitute for a completed experiment.

`failed_fit.json` retains the warning and exact training/config/source/software identity. `stopped.json` records counts and the preservation snapshot; `run_provenance.json` contains the completed dependency/model records. All 61 earlier preservation hashes still agree. The current source/config identity and stopped-attempt artifacts are retained for any separate successor study.

## Interpretation and next step

This is a numerical failure of one declared optimizer recipe, not evidence that spline features improve or fail the F2 objective. The complete matched development comparison, selection, uncertainty, transfer and figures remain unperformed. Do not rank the nine available fits as if they were the declared full search, silently raise the iteration limit or claim performance improvement from their default-threshold metrics.

A separate prospective study can test the same regularized representation grid with a solver appropriate for many rows and a modest transformed dimension. Its numerical choices must be declared before new fitting, preserve this failure and all ten completed studies, and retain actual F2 selection and consumed-holdout limitations. All settings remain analyst-selected. Codex assisted with planning, implementation and verification; the official scikit-learn APIs provide the feature/solver definitions, and no public notebook was copied.
