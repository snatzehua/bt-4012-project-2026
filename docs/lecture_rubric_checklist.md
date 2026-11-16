# Lecture and rubric checklist

Where each grading criterion and each relevant lecture method is covered. File names refer to
`experiments/`, `results/` and `figures/` in this repository. "Report section" follows the
recommended structure in the project guidelines (1 Problem, 2 Data, 3 Methodology,
4 Experiments and Results, 5 Conclusion).

Status: ✅ done in code with results · 📝 evidence exists, still to be written into the report/slides · ➖ not applicable (reason given)

## 1. Final report rubric (20 marks)

| Criterion (marks) | What the marker looks for | Where it is covered | Status |
|---|---|---|---|
| Problem formulation and practical importance (2) | Fraud problem, why it matters, challenges | Proposal §1; README intro; challenges shown by data: 1.1% fraud (`e0_monthly.csv`), drift (`e0_covariate_drift.csv`), age bias (`e5_fairness.csv`) | 📝 Report §1 |
| Data understanding and preprocessing (3) | Source, size, variable types, tabular summary, cleaning, feature engineering | E0 `e0_variable_summary.csv` (variable table), E8 `e8_univariate.csv` / `e8_categorical.csv` (L2 statistics), cleaning log in `run_info.json`, `src/features.py` + E7 ablation | ✅ / 📝 Report §2–3 |
| Method selection and technical justification (4) | Why these models, supervised vs unsupervised, preprocessing per model | Dummy → logistic regression (L3) → RF / LightGBM (beyond) → MLP + autoencoder (L4); per-model preprocessing (`src/models.py`); E7 shows which model gains from engineering; E9 justifies the L2 penalty strength | ✅ / 📝 Report §3 |
| Experimental design and evaluation (4) | Sound splits, no leakage, right metrics, uncertainty, controlled comparisons | Out-of-time split with validation month (all experiments); random-vs-temporal check (E2); recency vs size confound removed (E3b, TA feedback); paired bootstrap CIs (E1, E7); 3 seeds; metrics PR-AUC, recall@5%FPR, confusion matrix, costs (E5) | ✅ |
| Results, insights and real-world applicability (3) | Interpretation, business integration | Capacity / cost / fairness (E5), threshold drift (E5), SHAP and error profiles (E6), adversarial robustness (E11), triage workflow with case briefs (E12, `docs/llm_case_summary_prompt.md`) | ✅ / 📝 Report §4 |
| Limitations and reproducibility (2) | Honest limits, README to reproduce | README "Limitations" and "How to reproduce"; `run_info.json`, `timings.json`, `full_run.log`; seeded; smoke test on fake data; NN gradient test | ✅ / 📝 Report §5 |
| Clarity, organisation and conciseness (2) | 8–12 pages, clear structure | `docs/RESULTS_SUMMARY.md` gives the storyline; figures in `figures/` | 📝 |
| Admin | `reportXX.pdf`, GitHub URL in report, AI declared | README "AI declaration" | 📝 add group number, names, URL |

## 2. Presentation rubric (15 marks)

| Criterion (marks) | Suggested evidence | Status |
|---|---|---|
| Background and problem statement (2) | Fraud rate by month figure (`e0_fraud_rate_by_month.png`) | 📝 |
| Models and technical justification (3) | Model ladder dummy → LR → trees → NN; `e9_regularisation.png`, `e10_mlp_tuning.png` | 📝 |
| Experiments, results and interpretation (3) | `e1_pr_curves.png`, `e3b_recency_effect.png`, `e5_confusion_matrices.png`, `e11_evasion.png` | 📝 |
| Coherence and flow (2) | Follow the storyline in `docs/RESULTS_SUMMARY.md` | 📝 |
| Slide quality (2), professionalism (2), time and balance (1) | Title + members shown ~8 s; ≤15 min; everyone speaks | 📝 |

## 3. Lecture methods (lectures 2–11)

| Lecture | Method taught | Where in the project | Status |
|---|---|---|---|
| **L2** Data visualisation & analysis | % missing, quantiles, IQR, mean vs median, MAD, skewness, kurtosis | E8 `e8_univariate.csv` | ✅ |
| L2 | Categorical cardinality, bar charts | E8 `e8_categorical.csv`, `e8_category_fraud_rate.png` | ✅ |
| L2 | Test chosen by data type: t-test (numeric vs fraud), chi-square (categorical vs fraud), information gain | E8 `e8_bivariate_tests.csv`, `e8_information_gain.png`; ANOVA / chi-square of each feature across months in E0 `e0_covariate_drift.csv` | ✅ |
| L2 | "Huge samples make tiny effects significant" | E8 and E0 rank by effect size (Cohen's d, Cramér's V, IG, PSI), not p-values | ✅ |
| L2 | Pearson correlation for redundancy | E8 `e8_correlation.csv`, `e8_high_correlation_pairs.csv`, `e8_correlation_heatmap.png` | ✅ |
| L2 | Outliers: boxplot rule, \|z\| > 3 | E8 `e8_outliers.csv` (with fraud rate among outliers), `e8_boxplots.png` | ✅ |
| L2 | Benford's law, chi-square goodness of fit, when it fails | E8 `e8_benford.csv`, `e8_benford.png` (capped / bucketed variables excluded, slide 85) | ✅ |
| L2 | Visualisation types (line, bar, box, histogram, heatmap) | `e0_fraud_rate_by_month.png`, `e8_*.png`, `e0_concept_drift.png` | ✅ |
| **L3** Predictive analytics | Logistic regression as interpretable baseline; start with a simple benchmark | E1 (dummy + LR), E6 `e6_logreg_coefficients.csv` | ✅ |
| L3 | L2 (ridge) and L1 (lasso) regularisation; L1 sparsity | E9 `e9_regularisation_path.csv`, `e9_lasso_path.png` | ✅ |
| L3 | Overfitting vs underfitting (training vs test error) | E9 train vs validation PR-AUC; E1 `train_pr_auc` column | ✅ |
| L3 | Train / validation / test; tune threshold on validation only | All experiments (months 0–4 / 5 / 6–7) | ✅ |
| L3 | Accuracy fails on imbalanced data | E5 `e5_confusion_matrices.csv`: 94–98% accuracy while recall is only 24–56%; accuracy is never used to choose anything | ✅ |
| L3 | Confusion matrix, precision, recall, PR curve | E5 `e5_confusion_matrices.png`, E1 `e1_pr_curves.png` | ✅ |
| L3 | Cost table decides the threshold | E5 `e5_lecture_cost_table.csv` (slides 46–48 style) and `e5_costs.csv` (cost-ratio sensitivity) | ✅ |
| L3 | Anomaly detection trained on normal data only, threshold on validation | E10 autoencoder (legit-only training, threshold from month-5 legit errors); finding: fraud in BAF is not unusual (ROC-AUC 0.51) | ✅ |
| **L4** Deep learning | Feedforward network: ReLU, sigmoid output, log loss, mini-batch, epochs | `src/nn.py`, E10 MLP; `tests/test_nn.py` gradient check | ✅ |
| L4 | Width / depth: biggest network is not best on new data | E10 `e10_mlp_tuning.csv`, `e10_mlp_tuning.png` | ✅ |
| L4 | Dropout, scaled by (1 − p) at inference | `src/nn.py` (`_forward`), tested in `tests/test_nn.py` | ✅ |
| L4 | Autoencoder for anomaly detection; high error = signal, not proof | E10 `e10_autoencoder_errors.png` | ✅ |
| L4 | ROC curve and AUC | E10 `e10_roc.png` | ✅ |
| **L5** LLM for case analysis | Structured (schema-constrained) output | E12 `e12_case_brief_schema.json`, briefs validated with `jsonschema` | ✅ |
| L5 | Prompting, temperature 0, prompt injection, PII | `docs/llm_case_summary_prompt.md` (design only, no LLM is run) | 📝 design |
| **L6** RAG | Ground answers in policy, check citations | `docs/llm_case_summary_prompt.md` (how policy clauses would be retrieved and cited) | 📝 design |
| **L7** CNN & adversarial | Evasion attack (non-targeted, dodging), black-box query attack | E11 `e11_evasion.csv`, `e11_evasion.png` | ✅ |
| L7 | Transferability via a proxy model | E11 (attack crafted on LR, applied to LightGBM) | ✅ |
| L7 | Proactive defences (adversarial training) | E11 `e11_defences.csv` (adversarial training; robust feature set) | ✅ |
| L7 | CNNs for images | ➖ BAF has no images | ➖ |
| **L8** Social network analysis | Centrality, PageRank, communities | ➖ BAF has no shared identifiers between applications, so no graph can be built; listed in E12 `not_checked` and README limitations | ➖ |
| **L9** Tool-using agents | Deterministic work (scores, thresholds) in tools, not the LLM | E12: score, band and reason codes all computed in code | ✅ |
| **L10** Agentic workflow | Escalation rule is code; human approval for irreversible actions | E12 `escalation_rule`, `requires_human_approval` | ✅ |
| **L11** Single vs multi-agent | Report contract with `not_checked`; multi-agent only if it pays | E12 `not_checked` in every brief; single pipeline, justified in `docs/llm_case_summary_prompt.md` | ✅ |

## 4. Proposal and TA feedback

| Item | Where | Status |
|---|---|---|
| RQ1 rolling vs expanding window | E3 | ✅ |
| RQ2 random vs chronological split | E2 | ✅ |
| RQ3 imbalance treatments | E4, and choice on validation in E1 | ✅ |
| RQ4 trees vs logistic regression, threshold choice | E1, E5 | ✅ |
| TA: specify features; feature engineering per model | E0/E8 variable tables; `src/features.py`; E7 | ✅ |
| TA: recency confounded with sample size → compare K recent vs K historical | E3b | ✅ |
| TA: justify every section in the report | `docs/RESULTS_SUMMARY.md` gives the reasons per section | 📝 |
