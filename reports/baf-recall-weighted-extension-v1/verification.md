# Extension verification

The experiment is complete; the performance goal is not achieved. The selected macro month-5/6 F2 remains 0.3278015781, with no gain over the previous balanced LightGBM policy. The frozen policy also trails the unweighted reference on consumed month 7.

| Requirement | Checked evidence |
| --- | --- |
| Finite separate search | config.json; 39 base models, three seed means, six rank ensembles; 117 base month/results |
| Actual execution | audit.json: 36 new development fits, 81 verified reuses, 288 forward policy rows, zero new month-7 fits, two baseline reuses |
| Training/cache provenance | Runner and report audits reconstructed original training hashes and class matches; checked input identities, evaluation IDs/labels, score hashes and AP/ROC from predictions |
| Earlier-only rules | crossfit.csv contains month-5 rules sourced from month 4 and month-6 rules from 4/5; ensemble choices follow the same boundary |
| Baseline reproduction | Both original balanced/unweighted pooled-policy macro F2 values recomputed and matched the previous study |
| Contrasts and failures | matched_contrasts.csv (114 rows), ensemble_contrasts.csv (72), policy_contrasts.csv (96); pooled_development.csv preserves summed confusion counts |
| Immutable selection | Exported selection.json matches the local frozen record; its digest matches exploratory_access.json; freeze precedes further access |
| Direct metric/interval checks | Every saved forward/final metric and final policy was checked; paired intervals were rebuilt with the same seed and 500 draws |
| Visual verification | f2_extension.png was rendered and inspected for labels, clipping and agreement with the tables |
| Reproduction | README gives pinned environment, Base v2 access, small freeze-record restoration and sequential validate/freeze/exploratory/report commands; missing-window-cache refit test verifies a new destination |
| Preservation | All four studies' exported/local freeze records agree; source/config/software/data guards pass |
| Assistance | Findings and README acknowledge Codex; course submission still needs the stated software/data acknowledgments |

Final checks: `python -m pytest -q` passed all 47 methodological tests; `python -m compileall -q baf tests` passed; strict OpenSpec validation passed all 11 entries before archive and all 10 active entries after archive. Four implemented requirements were synchronized to the main extension spec. The completed change is at `openspec/changes/archive/2026-10-11-extend-recall-weighted-search/`, with 12/12 tasks checked.

These checks establish experiment completion and the negative result. They do not establish a performance improvement or independent confirmation on month 7.
