# Implementation checkpoint: bagged-anchor fusion

**Three of fourteen tasks are verified: 1.1, 2.1 and 2.2. No new full-data performance result exists. The broader F2 improvement goal remains active.**

The isolated configuration/protocol and three modules implement the finite declaration, individual source adapters, preserved aggregates, ten logit controls, twelve fixed blends, all 111 winner/control mappings, complete analytical tables and an independent grid auditor. Production execution gates, orchestration and bounded transfer/reporting are still pending. No execution declaration or selection has been frozen.

## Verified evidence

| Check | Observed result | Evidence |
| --- | --- | --- |
| Prospective declaration | 37 strategies; 55 source paths; 735 preserved paths; sixteen selection pairs | [Declaration](implementation-checks/declaration.json) |
| Historical preservation | All 3732 historical ignored files retain their original hashes; snapshot captured before implementation | `outputs/baf-bagged-anchor-fusion-v1/preimplementation_preservation.json` |
| Data and software | Actual Base SHA matches; Python 3.12.9 and all 27 pins match | [Data/software check](implementation-checks/data-software.json) |
| Full methodological regression suite | 1367 passed; 297 current-study tests; zero failures/skips in the passing invocation | [Closed log](implementation-checks/full-suite-1367.log), [JUnit](implementation-checks/full-suite-1367.xml), [receipt](implementation-checks/full-suite-receipt.json) |
| Synthetic analytical reconstruction | 120 physical scores, 111 selectable scores, 222 forward rows, 111 macro/pooled/monthly summaries, 947 contrasts | [Synthetic audit](implementation-checks/analytical/audit.json), [exported files](implementation-checks/analytical/files.json) |
| Additional synthetic diagnostics | Eighteen diversity rows, 36 decision-overlap rows, 39 clipping rows, sixty expression rows, eighteen bagged metadata blocks and fifteen reference metadata blocks | Same synthetic audit and exports |
| Current-study tiny cache fits | Eighteen total: six in each of three fitting invocations | [Development attempt ledger](implementation-checks/development-attempts.json) |

The six source-format tests cover balanced/unweighted bagged LightGBM, balanced/unweighted Newton spline, XGBoost and original LightGBM. Each tests genuine tiny fitting, persistence/inference auditing, audited reuse without refitting, origin copying and partial/corrupt rejection. Failed focused fixtures and the zero-fit test-collection failure remain recorded. Historical-module fits are separate; this checkpoint does not enumerate them as new study fits.

The synthetic fixture contains generated labels and scores and **no actual source-model fits**. Its high or low F2 values are not results for BAF and cannot establish a gain. The analytical auditor explicitly records `actual_source_models_audited=false`. Full-data preprocessing, numerical and inference auditing remains required before freeze.

Exact production/test/config/protocol source copies for the passing full suite are retained under `implementation-checks/full-suite-start/` with non-executable `.source` suffixes and a hash index. Earlier focused checks are development diagnostics rather than execution-gate receipts. The current receipt will become stale if these implementation sources change; retain it and rerun the full checks after such changes.

## Pending scientific work

Task 1.2 remains open until the source adapter and actual per-type work accounting are verified for the full required sources. Tasks 2.3 and 3.1 require the final execution checks/receipts, unchanged resource gate and preserved miniature ancestry verification. Tasks 3.2/3.3 require full development preparation and independent reconstruction of the actual grid and macro F2 winner.

Tasks 4.1/4.2 require the exact frozen winner/control DAG and bounded exploratory month-7 transfer. Tasks 5.1-5.4 require every uncertainty draw/interval, inspected figures, honest findings, isolated no-prior-cache reproduction, final preservation/identity auditing and OpenSpec closure. Do not archive or claim completion from this checkpoint.

No new full-size models or production miniature models were fitted, no full-data blends or new month-7 scores were evaluated, and all sixteen completed studies remain preserved. Month 7 is already consumed. Selection remains positive-class F2, beta=2, with no review quota or precision floor. Broader goal achievement still requires positive measured differences versus all five fixed references in both declared scopes.

Codex assisted with declaration, isolated implementation and verification. Pinned repository APIs were reused; no public notebook was copied. Settings remain analyst-selected and open to group review.
