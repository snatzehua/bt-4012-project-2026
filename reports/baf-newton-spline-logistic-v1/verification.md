# Newton spline-logistic verification

The study is complete, with no new F2 gain. The best new spline policy is `spline_c0p1_unweighted / rank_fraction`, macro F2 0.322264054745, against selected existing fusion 0.328906570178. All 18 matched representation contrasts are positive, ranging from 0.006968434155 to 0.019469969443. This establishes an improvement over the tested linear controls, not over the stronger existing references or a performance ceiling.

## Actual checked work

- 36 new full development fits, nine reference base reuses and three fixed reference fusions; zero reference refits. All 48 score artifacts and 90 forward decisions were reconstructed independently.
- All 237 contrast rows, 45 macro/pooled summaries, 36 solver and numerical records, 36 feature-block records, 2,088 feature diagnostics and coefficient exports were checked against actual saved models/inputs.
- Maximum independently reconstructed absolute training gradient: 9.96895934559e-07, within 1.1e-6. Maximum iterations: nine. Observed solver fallback entries: zero. The separate preflight made 24 fits; the full methodological suite passed 236 tests in 9.66 seconds, and compileall/pinned environment checks passed.
- Nine available converged old L-BFGS models were compared on identical training objectives outside model selection. Three optional old models were unavailable because the original run stopped before they were saved; none were refitted.
- Five exact roles were frozen before the recorded further consumed-month access. Three final reference bases were reused with zero new final model fits. Selected consumed F2 is 0.3726941226941227, below unweighted 0.3768844221105528.
- All 1,500 fixed-decision paired draws were independently reconstructed by direct sampled Boolean rows. Twelve development and four consumed interval summaries retain comparator aliases and declared exclusions. The existing fusion's development interval versus balanced overlaps zero; its consumed difference versus unweighted is negative. No new improvement or independent confirmation is claimed.
- Both operating and additive-effect PNGs were visually inspected: titles, axes, legends and panel labels are legible without clipping. The 1,717 effect-curve rows are coefficient/margin checked; the losing spline remains a development-only diagnostic and was not transferred.

## Traceable verifier correction

The first audit failed on inherited DataFrame indices, after all full-data numerical fits passed. Resetting indices proved identical exported summary values. Original fit sources/configuration, all model/score/run hashes and 35-source fit identities remain unchanged. Separate corrected analysis entry points, 37-source analysis identities and `analysis_declaration.json` expose the repair and its retrospective timing. The original source/config/error snapshot remains in the ignored output root. No numerical setting, operating policy, selection criterion or holdout boundary was changed.

The independent audit script/result and latest methodological record are copied alongside these findings and frozen in selection evidence. `report_hashes.json` checks exported/copy identities. All ten earlier completed study identities and the stopped L-BFGS attempt remain intact; its change remains at 5/14 and incomplete. Analyst-selected settings remain open to group review. Codex assisted with the implementation, checks and repair; no public notebook was copied.
