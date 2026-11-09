# Supplementary assertion correction

The production `validate` command completed successfully and independently audited all 84 saved source models, all compositions and tables before selection freeze. A subsequent inline supplementary reader failed on this assertion:

```python
assert a['compositions'] == {'new_full_compositions':48,'member_subsets':54}
```

The actual composition subrecord also contains `actual_source_models_audited: false`, correctly scoped to that score-only subroutine. The separate enclosing model audit sets `actual_source_models_audited: true` and `saved_model_checks: 84`, alongside independent preprocessing/inference/cost/membership checks. The extra scoped field does not contradict the enclosing model audit. The failed supplementary assertion stopped before checking task 3.3 complete. The frozen selection command independently required the successful enclosing 84-model audit and completed without modifying it.

The saved corrected reader checks each actual field at its declared scope. No source/config/model/prediction/audit/selection artifact was edited, no fit was repeated and no numerical requirement was relaxed. This is a reader assertion correction, not a changed experiment or model-audit bypass.
