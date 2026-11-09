## Why

The user wants a better precision/recall score with recall weighted more heavily; the established objective is positive-class F2, beta=2, without a review quota or precision floor. The completed XGBoost study improved average precision for some models but failed to improve actual forward F2: its best new raw score was 0.326484 and its best percentile blend 0.325311, versus the balanced reference's 0.327802. This motivates a distinct question: can fixed raw-score or log-odds fusion exploit useful differences between the same frozen models that percentile fusion did not?

`BT4012_Proposal_BAF_Revised.pdf` motivates class-imbalance, model and threshold comparisons, while `feedback.md` requests controlled comparisons and model-specific justification. This is an exploratory extension of that scope, not a new approved group decision or a promise of improvement. It changes score composition while keeping model recipes, populations and chronological operating policies fixed.

## What Changes

- Declare a finite separate study of seven raw controls, six single-member clipped-logit controls and 48 fusions: four frozen XGBoost presets, two LightGBM anchors, three fixed weights and two composition methods. All 61 strategies use the unchanged three forward F2 policies.
- Reuse and audit existing base predictions and models; reproduce genuinely absent caches under exact frozen recipes only in the new destination. Do not add a package, change preprocessing, retune a base model or learn a combiner.
- Match each fusion to its members and its alternative composition method. Expose clipping and score-scale limitations; neither weighted scores nor their fusions are asserted to be calibrated fraud probabilities.
- Select on actual macro month-5/6 F2, preserve both original pooled references, freeze the winner and predetermined member/representation controls before limited exploratory transfer to already-consumed month 7.
- Export complete operating tables, matched contrasts, provenance, conditional paired uncertainty, checked figures, findings and reproducible commands. Preserve all nine completed studies and report negative findings if the method fails.

Core work includes the whole declared grid, controls and verification. Learned weights, more base hyperparameters, new features, calibration, extra datasets and alternative beta values are outside this change and require separate planning. Completing this study does not satisfy the performance goal unless measured evidence supports improvement.

## Capabilities

### New Capabilities

- `baf-xgboost-score-fusion-study`: A preserved chronological study of fixed raw-score and log-odds fusions with matched controls, actual-F2 selection and auditable limited exploratory transfer.

### Modified Capabilities

None. Existing study requirements and frozen implementations remain unchanged.

## Impact

Additive planned files: `configs/xgboost-score-fusion.json`, `docs/xgboost-score-fusion-protocol.md`, `baf/score_fusion.py`, `baf/score_fusion_study.py`, `baf/score_fusion_report.py` and focused methodological tests. Reuse unchanged `requirements-xgboost.txt` and its pinned environment. New artifacts go to ignored `outputs/baf-xgboost-score-fusion-v1/` and checked small exports in `reports/baf-xgboost-score-fusion-v1/`.

Implementation will update README reproduction commands, AGENTS, project context and assistance attribution to reflect verified state. It must preserve all previous source/config/software/selection identities and must not rewrite historical findings or present month 7 as fresh confirmation. This proposal creates no experiment result.
