# Repository instructions

## Project context

BT4012 Fraud Analytics, NUS AY2026/2027 Semester 1, Group 8. The report and presentation are due **20 November 2026, 23:59 Asia/Singapore** and together account for 35% of the course grade.

Read the [revised proposal](BT4012_Proposal_BAF_Revised.pdf) and [feedback](feedback.md) for research intent. Feedback requires size-matched recent-versus-historical comparisons and model-specific feature/preprocessing justification. Its final sentence is truncated; do not reconstruct missing text.

Start with [README.md](README.md), the [code and evidence guide](docs/submission-code-guide.md) and [submission assets](submission/README.md). The canonical implementation is `baf/`. The original study uses `configs/study.json` and `docs/experiment-protocol.md`; improvements use `configs/improvements.json` and `docs/improvement-protocol.md`. Individual `reports/baf-*/findings.md` files are authoritative for measured results and limitations. `archive/` holds ignored historical planning/course material and is absent from fresh clones.

## Scientific preservation

- Month 7 has already been evaluated. Preserve the original selection and access records; follow-up evaluations are exploratory. A new directory or checkout does not restore an independent holdout.
- Preserve all nineteen completed predecessor studies, their source/config/protocol/software/model/score/selection/access identities, ancestor selection pairs, mandatory-path hashes and historical ignored-file indices. Do not change expected hashes, recipes or assertions to pass a verifier.
- Preserve separate Newton fit and corrected analysis histories, and the stopped L-BFGS attempt. The latter failed its declared convergence gate and is not a completed experiment; do not rank its partial grid as a complete search or archive it as successful.
- Current mutable README/AGENTS/OpenSpec context is distinct from exact historical snapshots. In `baf/context_ablation_study.py`, `context_snapshot` and `verify_parent_indices` resolve the three declared historical context paths through their preserved snapshots. Never edit those snapshots, parent indices or frozen source/model/result/log entries to accommodate a current documentation edit.
- Do not run a repository-wide formatter, refactor frozen modules, upgrade the scientific environment or overwrite existing scientific/report receipts during routine cleanup. Check path/hash dependencies before editing any scientific protocol or source, including Markdown.
- Full replay belongs in a disposable checkout with original LF bytes, the pinned environment and verified data. Report stages can overwrite tracked files. Use fresh receipt paths where supported; do not remove a working selection to bypass freeze guards. See [study recipes](docs/study-reproduction.md) and the [shallow CatBoost replay guide](reports/baf-shallow-catboost-v1/reproduction.md).
- Further modelling or policy tuning requires a separate study, not edits to a completed identity. Settings are analyst-selected, not assumed group-approved decisions.

## Current research state

The original controlled study and improvement study are complete. Completed F2 descendants, negative results, figures, reproduction checks and exact accounting remain in their report directories and archived OpenSpec tasks; do not duplicate that experiment ledger in this file.

The selected bagged-anchor/pooled strategy has development macro F2 0.330880962 and consumed-month F2 0.372239748, below the unweighted reference at 0.376884422. The later conditional-expert and shallow-CatBoost studies did not improve that selection. F2 has beta=2 with no quota or precision floor. Do not claim the broader performance goal has been achieved or retrospectively replace the development winner using month 7.

`test-context-feature-ablation-f2` remains in implementation at **3/14 verified tasks**. Read its [active tasks](openspec/changes/test-context-feature-ablation-f2/tasks.md), [configuration](configs/context-feature-ablation.json), [protocol](docs/context-feature-ablation-protocol.md) and `reports/baf-context-feature-ablation-v1/` before continuing it. Existing core adapters and synthetic audits are not completed BAF results; full checks, production gates and scientific execution remain pending. Its prospective declaration preserves nineteen predecessors, nineteen identical selection pairs, 2020 mandatory paths and 5432 indexed historical files plus one separately preserved compiled cache. The new scientific execution identity is not frozen. Submission consolidation excludes this unfinished work from measured claims.

## OpenSpec workflow

- Read `openspec/config.yaml`, relevant `openspec/specs/*/spec.md` and the active change before substantial work.
- Use `.agents/skills/openspec-*/` workflows for proposals, experiments, pipeline changes and evaluation behaviour. Routine editorial corrections need no new experiment.
- Keep proposed requirements in `openspec/changes/<change>/specs/`; main specs describe current requirements. Artifact existence is different from task completion. Mark a task complete only after actual outputs exist and have been checked; archive only after verification and spec synchronization.
- `define-baf-experiment-protocol` documents the protocol; `run-baf-eda-and-hypothesis-tests` tracks measured execution. `improve-baf-detection` tracks the completed improvement study and its adaptive logistic diagnostic. `consolidate-submission-deliverables` records the submission exporter and drafts.
- After editing OpenSpec artifacts run `openspec validate --all --strict --no-interactive`. Report an unavailable tool or failed validation honestly.
- Keep current README, project context, instructions and relevant specs consistent without rewriting archived historical documents.

## Research and implementation conventions

- Use chronological training/validation/evaluation on identical future applications. Random splitting is a labelled diagnostic, not evidence of drift by itself. Train preprocessing and learn thresholds only on eligible earlier data.
- Separate recency from sample size with matched-K controls. The original historical samples match exact fraud/nonfraud counts and may include recent observations; they are not old-only samples.
- Explain actual predictors, excluded fields, missing-value codes and categorical handling. Justify differences between logistic and tree feature processing rather than treating all models identically.
- Preserve hyperparameters, seeds, weight/prior corrections, clipping, blend arithmetic, rounding, tie-breaking and serialization. Distinguish fixed-capacity ranking from learned score thresholds; report actual workload as well as AP/F2, precision, recall, missed fraud and legitimate reviews.
- Keep the supplementary notebook timeline and monthly floor rounding separate from the canonical study's ceil capacity. Its embedded outputs and external Drive artifacts are qualified in the [pinned branch audit](docs/submission-branch-audit.md); do not imply independent inference reproduction or combine incompatible workflows into one leaderboard.
- Retain conditional-bootstrap limitations, failed experiments and negative findings. Do not invent results, financial savings or group approval.
- Keep datasets and bulky artifacts in ignored `data/` and `outputs/`. BAF Base v2 belongs at `data/raw/baf/Base.csv`; its expected SHA and exact Python/dependency setup are in the code guide. Tests differ in data/library/ancestor requirements and may fit small models; use an appropriate scope and distinguish current checks from historical receipts.
- Update concrete reproduction instructions alongside editable modelling work: dependencies, data, configurations, stage order, seeds and outputs.

## Submission deliverables

The report must have 8-12 pages excluding references/appendices and follow Problem Description -> Data Overview -> Methodology -> Experiments and Results -> Conclusion. The full course brief is locally at `archive/BT4012 Project Guidelines and Grading Criteria.pdf`. Presentation video must be no longer than 15 minutes. Group 8 filenames are `report08.pdf`, `presentation08.pptx` and `video08.mp4`.

The tracked Markdown/PDF and slide skeleton are drafts. The separately maintained `report08.docx` is not regenerated by the Markdown builder. Do not overwrite manual slides with the skeleton generator or claim final approval, page-count/video checks, repository access or Canvas submission without evidence. The [submission guide](submission/README.md) records asset sources and remaining checks.

Acknowledge generative AI, external code and public notebooks; retain study-specific attribution as substantive work is produced. Every group member must be able to explain and justify the submitted work.
