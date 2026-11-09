# Stacking verification

The finite experiment is complete. The performance goal remains unachieved: macro F2 stays 0.3278015781, and the selected baseline has no consumed-month-7 gain over the stronger unweighted reference. Higher stack AP is not credited as achieving F2 improvement.

- Configuration/protocol declared 27 base results, 24 learned presets, 72 combiner/control fits and 156 chronological policy rows before fitting. All exist and were checked.
- Actual base execution: 12 early fits and 15 verified reuses, followed by 72 learned fits. Every recorded training month precedes its prediction month; fit warnings are empty.
- Runner and report audits reconstructed training identities/class counts and checked source/config/software/data, aligned future IDs/labels, original-cache provenance, source-score hashes and model hashes.
- The report rebuilt every combiner's forecast from its saved model and fixed member scores, verified its scaler against eligible earlier score rows, recomputed all forward/final metrics and generated paired fixed-decision intervals directly from predictions.
- Both original macro F2 baselines reproduced. Exact actual confusion counts reproduce F2=5TP/(5TP+4FN+FP) for all 156 rows.
- Exports retain 156 matched-factor contrasts, 52 policy contrasts, 78 macro summaries, 78 pooled summaries, six development interval rows and two consumed-month interval rows. No losing preset was scored on month 7.
- Selection remained baseline_balanced/pooled. There were no final combiner fits or new month-7 base fits, and two verified baseline reuses. The access record's digest matches selection; freeze precedes further access.
- All five exported/local freeze records agree. Their preservation guards and the report-source digest pass.
- The PNG was rendered and visually inspected for labels, clipping and agreement with the tables.
- README includes pinned dependencies, data access, small freeze-record restoration and sequential validate/freeze/exploratory/report commands. Missing-cache tests verify refits into new destinations; the full early fits exercised that route with actual models.
- Final checks: 57 methodological tests pass; `python -m compileall -q baf tests` passes; strict OpenSpec validation passes all 12 entries before archive. Five verified requirements were synchronized to the main stacking spec. All 12 tasks are checked.

These checks establish execution, provenance and the negative result. They do not establish the requested performance improvement, acceptable operating costs or independent month-7 confirmation.
