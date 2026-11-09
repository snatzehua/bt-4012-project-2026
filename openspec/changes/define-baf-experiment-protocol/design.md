## Context

The revised proposal includes window design, imbalance handling, logistic regression, random forest, gradient boosting and a random-split benchmark. Feedback specifically requests matched-size training comparisons and clearer feature choices. The completed protocol reconciles this scope without silently adopting the archived narrower proposal; execution is tracked in the separate study change.

## Goals / Non-Goals

**Goals:** Specify comparisons that can distinguish recency from size, prevent temporal leakage, justify model-specific preprocessing and lead to reproducible evaluation. Produce `docs/experiment-protocol.md` with concrete settings or explicitly blocked decisions.

**Non-Goals:** Run experiments, promise a recency advantage, introduce novel algorithms, switch to IBM AML or assume synthetic results generalize to bank deployment.

## Decisions

1. **Document the protocol before implementing the pipeline.** Select the dataset variant using official metadata and later profiling; provide a variable table identifying target, time field, predictors and exclusions. Do not infer actual feature contents from the proposal alone.
2. **Use common future evaluation applications.** Define forward validation periods and a final untouched future test period. Choose models, preprocessing, K, imbalance settings and thresholds on eligible training/validation data. State label-availability assumptions.
3. **Control training size explicitly.** At each cutoff, compare the most recent K eligible applications with K sampled from the eligible historical pool for several feasible K values. Repeat historical sampling with recorded seeds. Keep expanding history as the practical reference. State whether historical samples can include recent rows; disjoint older-only controls change the estimand and need sufficient eligible rows.
4. **Handle monthly ties honestly.** BAF's month field cannot necessarily order applications within a month. Specify a deterministic, target-independent tie policy for a K boundary, or restrict K choices to complete recent months, and report actual counts. Do not claim within-month recency from file order.
5. **Separate class composition and preprocessing effects.** State whether matched-K controls also match fraud prevalence and why. Apply a consistent imbalance policy within each recency comparison. Document encoding, scaling and missing-value handling for logistic regression separately from trees; fit transformations only on the relevant training subset.
6. **Evaluate the operating policy explicitly.** Define precision/recall, F1, a precise PR-AUC or average-precision convention, confusion matrices and precision/recall at stated review capacities. Fix rounding and score-tie rules. Select thresholds on validation; top-budget ranking is a separate policy. Report missed fraud and legitimate applications flagged. A random-split benchmark is diagnostic and does not establish temporal drift by itself.

## Risks / Trade-offs

- A full cross-product of K, models, treatments, months and seeds may exceed the project budget. Define the core matched-K comparison first and justify smaller secondary ablations.
- Too few fraud examples can make matched controls infeasible or unstable. Record feasibility limits and uncertainty; never silently alter K or class proportions.
- Synthetic data, coarse monthly time and unknown reporting delays limit conclusions. Discuss these rather than making deployment claims.

## Resolved execution settings

The completed `docs/experiment-protocol.md` and `configs/study.json` select Base v2, validation months 4-6 and test month 7, K values 100,000/250,000/500,000, three historical seeds and explicit monthly tie handling. Logistic regression and LightGBM cover the recency controls; expanding-history weighting ablations include random forest. Review capacities, metrics, validation-derived thresholds, fixed presets and conditional paired-bootstrap rules are specified. Real-data feasibility and exact matching checks are recorded in `reports/baf-study-v1/protocol-checks.json`.
