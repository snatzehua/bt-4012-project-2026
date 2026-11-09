## Context

The prior F2 search is complete and audited at reports/baf-recall-weighted-v1/: 30 new development fits, 15 reused fits and 108 forward model/policy evaluations. Balanced LightGBM remains selected at macro F2 0.3278015781. The best new small-tree/rank-fraction combination scores 0.3267080684. Longer boosting and weights up to 10 therefore failed to beat the existing balanced model; completion is not achievement of the performance goal. See proposal.md for motivation.

## Goals / Non-Goals

**Goals:** Seek a verified F2 gain with stronger class emphasis, shorter boosting, previously tested chronological training windows and scale-comparable ensembles. Preserve negative evidence and distinguish model, policy, training-window and ensembling effects.

**Non-Goals:** Change beta after seeing results, impose 1% capacity, optimize support-weighted F1, promise 90% recall, retune on month 7 or modify any completed study. The new search is adaptive exploratory evidence, not a fresh held-out test.

## Decisions

1. **Same data and timing.** Base v2 checksum and 29 eligible predictors remain unchanged. Train on months before 4/5/6; score identical applications. Operating evaluation on month 5 uses month 4 predictions, and month 6 uses 4/5. Numeric NaNs/native categoricals use training-only adapters; target/month/constant fields remain excluded. The original window models retain their frozen imputation/one-hot preprocessing, so differences from new native models are not attributed solely to training weights.

2. **New grid.** Twelve native LightGBM presets cross leaves {15,31}, positive weights {1,30,100}, and boosting rounds {200,600}; learning rate .05, min_child_samples=100, reg_lambda=10, colsample_bytree=.9, seed 2026, six threads. This yields 36 development fits. Matched weight-1 controls and crossed rounds/leaves isolate those factors within the new grid. No outer-fold early stopping. Retain all 15 previous F2 base presets via verified reuse or reproduction as references; their sources remain frozen.

3. **Original window reuse.** Add original unweighted LightGBM recent-K and historical-matched-K for K {100000,250000,500000}, historical seeds {17,29,43}: 12 models per fold, 36 existing predictions. Verify the frozen original config/source identity, exact training-row hashes using original sampling routines, class counts and row/label equality. Check recorded AP/ROC against scores and preserve prediction digests in new provenance. If missing, refit the exact original recipe into extension outputs with both old-recipe and new-study identities recorded. Never write old output directories. These comparisons can assess F2 at the existing matched controls; they do not establish causal recency effects.

4. **Seed and rank ensembles.** Average the three historical-matched raw-score members for each K, producing three fixed seed ensembles without fits. Add two earlier-selected ensemble families anchored on balanced LightGBM: best eligible window/seed-ensemble member and best eligible native LightGBM member, each with anchor weights {.25,.5,.75}. Select the other member by macro earlier-month F2 at its pooled earlier threshold, tie by candidate ID. Normalize each member's scores within the same unlabeled prediction month to average-tie percentiles `(rank_average(score)-.5)/n`, then blend. This avoids arbitrary mixing of weighted and unweighted raw score scales. It is a batch ranking rule, not an online per-application calibrated probability. No current-month labels enter normalization. Compare to both constituent policies and retain losing ensembles.

5. **Policies and selection.** Retain the prior pooled/latest maximum-F2 threshold and learned rank-fraction policies, with identical tie/rounding rules. Keep policy and member fitting earlier-only. Before fitting, enumerate the entire base/derived grid in config: 15 old base references + 12 new native presets + 12 original window models + 3 seed ensembles + 6 rank-blend strategies = 48 strategies, yielding 288 month-5/6 policy evaluations. Select by macro F2, then recall, precision and deterministic ID. Export pooled confusion-count F2 separately. Report both previous unweighted and balanced LightGBM policies, including the strongest observed consumed-period baseline; do not hide a worse comparator behind the selected development reference.

6. **Freeze and transfer.** Freeze model/policy/dependency identities and rules fitted from months 4/5/6 before any new month-7 scoring. Score only the selected strategy's dependencies and the two baseline comparators; reuse exact existing predictions when valid. Refit a selected training-window recipe if its month-7 cache is absent. Month 7 is already consumed and all transfer results remain exploratory. Preserve the new frozen selection hash in a separate access record.

7. **Audits and evidence.** Use the current F2 arithmetic and policy adapters read-only; add extension-specific provenance/rank/ensemble tests. Verify training hashes, class/size controls, identical cases, source eligibility, normalization without labels, deterministic ties, preserved artifacts and full metric recomputation. Report 500 paired row-bootstrap draws at fixed decisions (seed 1027), monthly variation and conditional uncertainty, explicitly excluding selection/training/temporal uncertainty. Improvement claims require a positive measured difference on the named scope and cannot be upgraded to clear confirmation if intervals include zero. No gain leaves the objective active.

## Risks / Trade-offs

- More adaptive comparisons can find noisy development winners → predeclare the finite grid, disclose search optimism and retain unsuccessful rows.
- Strong weights may worsen precision or ranking → matched weight-1 controls and explicit FP/FN/review volume.
- Rank normalization requires the whole application batch → state batch operation and compare identical month populations; do not present it as a calibrated probability model.
- Historical caches use an older metadata format → explicit adapters and training reconstruction; do not infer compatibility from names alone.
- Positive development changes may fail transfer → retain both prior baselines and describe consumed-month failure honestly.

## Migration Plan

Add configs/recall-weighted-extension.json, docs/recall-weighted-extension-protocol.md and new extension modules/tests. Preserve all existing frozen modules and manifests. Run the finite development search, audit, freeze, exploratory selected transfer and inspected report. Update README/AGENTS/context and synchronize requirements only after outputs exist. No new dependency is initially needed. New outputs may be removed without modifying historical studies.
