"""Read-only accounting check after the declared development process exits."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from baf import conditional_expert_study as run
from baf.common import read_config, write_json, digest, load_data

c, fp = run.initialize()
run.execution_gate(c, fp)
out, dest = Path(c['output_dir']), Path(c['report_dir'])
prep = read_config(out/'preparation.json')
assert prep['fingerprint'] == fp
expected = dict(base_results=84, reference_base_results=24, reference_reuses=24,
    reproduced_bagged_fits=0, reproduced_spline_fits=0, reproduced_tree_fits=0,
    local_fits=54, capacity_fits=6, derived_results=63, reference_derived_results=15,
    reference_derived_reuses=15, new_compositions=48, invocation_new_conditional_fits=60,
    invocation_exact_reference_fits=0, model_artifacts=84, model_files=87, prediction_artifacts=147)
assert all(prep[k] == v for k, v in expected.items())
run.verify_hashes({str(out/n): h for n,h in prep['work_files'].items()})
frame = load_data(c)
frame = frame[frame.month < 7]
assert run.exact_artifact_grid(c, frame, 'validation') == {k: prep[k] for k in ['model_artifacts','model_files','prediction_artifacts']}
assert not (out/'selection.json').exists() and not (out/'exploratory_access.json').exists()
records = [read_config(p) for p in (out/'runs').glob('validation-*.json')]
new = [r for r in records if r['candidate'].startswith(('expert_model_', 'capacity400_'))]
assert len(new) == 60 and len(records) == 147
eligibility = []
for month in [4,5,6]:
    whole = frame[frame.month < month]
    for partition, levels in {'payment_type':['AA','AB','AC','AD','AE'],
            'housing_status':['BA','BB','BC','BD','BE','BF','BG']}.items():
        for level in levels:
            local = whole[whole[partition].eq(level)]
            n, fraud = len(local), int(local.fraud_bool.sum())
            eligible = n >= 10000 and fraud >= 100 and n-fraud >= 100
            for weighting in ['balanced','unweighted']:
                eligibility.append(dict(eval_month=month, partition=partition, level=level,
                    weighting=weighting, rows=n, fraud=fraud, eligible=eligible))
assert len(eligibility) == 72 and sum(not r['eligible'] for r in eligibility) == 18
assert sum(r['eligible'] for r in eligibility) == 54
for r in new:
    spec = r['spec']
    assert r['fingerprint'] == fp and r['warnings'] == [] and not r['reused']
    assert r['requested_model_params']['class_weight'] is None
    whole = frame[frame.month < r['eval_month']]
    local = whole if spec['kind'] == 'capacity' else whole[whole[spec['partition']].eq(spec['level'])]
    assert r['global_train_rows'] == len(whole) and r['global_train_fraud'] == int(whole.fraud_bool.sum())
    assert r['local_train_rows'] == len(local) and r['local_train_fraud'] == int(local.fraud_bool.sum())
    costs = {'0': 1., '1': 1.} if spec['weighting'] == 'unweighted' else {
        str(y): len(whole)/(2*int((whole.fraud_bool==y).sum())) for y in [0,1]}
    assert r['class_costs'] == costs
    assert r['subset'] == (spec['kind'] == 'local')
    assert 1 <= r['fitted_tree_count'] <= (200 if spec['kind'] == 'local' else 400)
    assert r['train_row_rounds'] == len(local)*r['fitted_tree_count']
    model = out/'models'/f"validation-m{r['eval_month']}-{r['candidate']}.joblib"
    pred = out/'predictions'/f"validation-m{r['eval_month']}-{r['candidate']}.npz"
    assert digest(model) == r['model_sha256'] and digest(pred) == r['prediction_sha256']
assert sum(r['train_row_rounds'] for r in new) == prep['conditional_train_row_rounds']
assert sum(r['fitted_tree_count'] for r in new) == prep['conditional_stored_trees'] <= 13200
assert np.isclose(sum(r['fit_seconds'] for r in new), prep['conditional_original_fit_seconds'], atol=1e-10, rtol=0)
assert len(pd.read_csv(out/'conditional_model_work.csv')) == 60
assert len(pd.read_csv(out/'conditional_inference_work.csv')) == 96
run.guard(c)
result = dict(status='passed', fingerprint=fp, preparation_sha256=digest(out/'preparation.json'),
    completed_process_log_sha256=digest(dest/'development-preparation.log'),
    counts=expected, eligibility=eligibility, eligibility_records=72, fallback_records=18,
    new_production_development_fits=60, new_reference_reproduction_fits=0,
    model_inference_audit_pending=True, month7_access=False, created_at_utc=run.now())
write_json(dest/'development-preparation-verification.json', result)
print('Verified 60 new development fits, 24 base/15 derived reuses, 84 models/87 files, 93 full scores/54 subsets and raw eligibility; independent inference audit follows.')
