## 1. Declare the separate study

- [x] 1.1 Add configs/calibrated-f2-policy.json and docs/calibrated-f2-policy-protocol.md with the three fixed score sources, beta=2, no quota or precision floor, chronological boundaries, 24 development calibration fits and 42 forward operating rows; verify the declared grid counts and that all five prior source/config/software/selection identities still match their frozen records.
- [x] 1.2 Implement the additive study identity and preservation guard, including data checksum, prior selection digests and new source/config/software hashes; verify changed prior sources or records are rejected and the new destination cannot overwrite a prior study.

## 2. Reproduce verified score sources

- [x] 2.1 Load and audit the nine fixed model/month score sources for months 4/5/6, including base-member and chronological combiner provenance; verify exact row IDs/labels, training boundaries and model/prediction hashes, and test rejection of a tampered cache.
- [x] 2.2 Implement missing-cache reproduction under the frozen recipes in new directories: up to 27 required base results and three fits of the one fixed stack; verify available dependencies are reused, missing ones are reproduced without invoking old validation/freeze commands, actual fit/reuse counts are recorded and prior files remain unchanged.

## 3. Implement chronological calibration and decisions

- [x] 3.1 Implement pooled/latest sigmoid-logit and isotonic calibrators with the declared clipping, training-only scaler, unweighted fit, monotonicity and convergence checks; verify empty/one-class behavior, clipping, saved-artifact reconstruction and that month-5/6 calibration uses only the permitted earlier prediction populations.
- [x] 3.2 Implement the label-free batch prefix rule and its k, predicted-positive, estimated-TP and surrogate exports; verify against brute-force tiny examples, all-zero/all-one probabilities, flat isotonic blocks, raw-score/ID ties and changed current labels that leave decisions identical.
- [x] 3.3 Retain the three empirical F2 controls with their frozen history, threshold ties and rank-fraction rounding; verify the balanced/unweighted pooled macro F2 references reproduce, and selection inputs exclude calibration surrogates and label-aware hindsight.

## 4. Measure and freeze development selection

- [x] 4.1 Run the finite development grid and export all 24 calibration records and 42 forward operating rows, with warnings/failures and exact calibration source months; independently reconstruct decisions and actual F2/F1/F3, precision/recall, confusion counts, workload, AP/ROC, MCC, balanced accuracy and raw/calibrated Brier/log-loss on identical cases.
- [x] 4.2 Export deterministic macro-F2 selection, monthly and pooled comparisons against both old baselines and each policy's same-model pooled control, plus ten equal-frequency calibration bins; verify all declared rows and unsuccessful contrasts are retained and baseline values match the prior studies.
- [x] 4.3 Freeze the selected policy, its final calibrator when applicable, all score-model dependencies and the new study identity before further month-7 access; if the fixed stack wins, fit its single final chronological combiner in the new destination before access, and verify every frozen artifact and dependency hash.
- [x] 4.4 Generate the six isolated label-aware development ranking-prefix diagnostics only after selection freeze; verify their diagnostic labels and timestamps, their exclusion from selection/calibration inputs, an unchanged selection digest and the absence of any month-7 hindsight optimum.

## 5. Transfer and assess measured performance

- [x] 5.1 Score only the frozen selected model/policy, its same-model raw pooled comparator and both old raw baseline policies on consumed month 7, with selected dependencies only; verify access follows freeze, losing calibrated policies are not swept, metrics reconstruct and findings label the transfer exploratory.
- [x] 5.2 Export 500 paired fixed-decision bootstrap draws with seed 1027 for selected-versus-both-reference and same-model differences, using within-month resampling for development macro F2; verify the draws against a small reference implementation, preserve realized batch decisions and state the conditional uncertainty exclusions.
- [x] 5.3 Produce checked findings, complete metric/calibration/contrast tables, provenance/preservation audit and calibration/operating figures under reports/baf-calibrated-f2-policy-v1/; inspect figures and verify claims use actual F2 on the named scope, retain failed transfer and leave the performance goal unachieved if improvement is not demonstrated.

## 6. Reproduction and completion checks

- [x] 6.1 Update README with dependencies, Base access, five frozen-record restoration steps, missing-cache reproduction and ordered validation/freeze/diagnostic/transfer/report commands; update AGENTS.md, project context and assistance acknowledgment, and verify every referenced path/command agrees with implemented behavior.
- [x] 6.2 Run the methodological suite, python -m compileall -q baf tests and openspec validate --all --strict --no-interactive; record actual outcomes, verify cache rejection and fresh-destination reproduction checks, and confirm all five prior frozen studies are preserved after execution.
- [x] 6.3 After the preceding outputs and checks pass, synchronize the implemented delta requirements to main specs; verify the synchronized requirements, checked completion evidence and final strict OpenSpec validation, without treating study completion as proof of a performance gain.

After the entire checklist is verified, archive through the standard OpenSpec workflow and verify the archived checklist and final strict validation. Archival is the final workflow action, outside its own prerequisite checklist.
