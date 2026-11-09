## Context

Use the existing verified BAF Base v2 CSV and the completed experiment protocol. The user authorized both analysis and implementation in this request. The protocol resolves execution settings for this first study, while group adoption remains reviewable.

## Goals / Non-Goals

**Goals:** Produce actual EDA and measured tests of recency, sample size, weighting and model choice; preserve a future holdout through selection; enable another member to reproduce the study.

**Non-Goals:** Exhaustive hyperparameter optimization, SHAP, resampling, alternative variants, monetary-loss claims or production deployment.

## Decisions

- Use script modules and pinned Python dependencies rather than a stateful notebook. Separate EDA, validation, selection freeze, final evaluation, diagnostic and reporting commands so test access is explicit.
- Implement the exact folds, K values, seeds, missing codes, model presets and operating metrics in the protocol/config. Fit preprocessing within each training subset.
- Cache each fitted-run result and predictions with config/source hashes; reject stale cache entries. Record selected training row hashes and month/class counts.
- Pick K per model from recent-strategy validation recall, and select an overall candidate only among the predeclared final comparator configurations. Save its validation-derived thresholds and hashes before reading final labels for performance.
- Use paired bootstrap weights to recompute ranked-capacity recall without repeatedly sorting duplicate rows. Report conditional intervals separately from seed and month variability.
- Export small summaries and inspected charts to `reports/baf-study-v1/`; leave bulk results in ignored `outputs/`.

## Risks / Trade-offs

- Runtime from 90 validation fits -> six-thread sequential execution and resumable checkpoints.
- Leakage or misleading recency claims -> boundary/sampling/metric tests, run audits, class-matched controls and explicit observational limitations.
- Single-test-month uncertainty -> three forward validation months, seed repetitions, conditional bootstrap intervals and restrained conclusions.
