"""Independently inspect actual saved sixteen-fit miniature artifacts before full fits."""
from pathlib import Path
import copy
import hashlib
import inspect
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from baf import conditional_expert_study as run
from baf import conditional_expert_report as audit
from baf.common import read_config,write_json,digest,make_model

c,fp=run.initialize();run.execution_gate(c,fp);out=Path(c['output_dir']);dest=Path(c['report_dir'])
gate=read_config(out/'miniature_gate.json');decl=read_config(out/'declaration.json')
resource=read_config(out/'resource_preflight.json');source=read_config(out/'source_miniature_gate.json')
assert resource['predictor_only'] and resource['training_rows']==903157 and resource['max_dimension']==157
assert resource['workspace_upper_bytes']==11130259216 and resource['workspace_upper_bytes']<=.7*resource['host_ram_bytes']
assert source['preserved_sampling_fits']==9 and source['preserved_older_fits']==8 and source['new_production_miniature_fits']==0
assert gate['fit_count']==16 and len(gate['results'])==len(gate['attempts'])==16
assert all(r['status']=='completed' and decl['created_at_utc']<r['started_at_utc'] for r in gate['attempts'])
columns=pd.read_csv(c['data_path'],nrows=0).drop(columns=['fraud_bool','month','device_fraud_count']).columns
frame,base=run.miniature_data(columns);whole=frame.index[frame.month<2];future=frame.index[frame.month==2]
assert len(whole)==24000 and len(future)==12000 and list(base)==gate['predictor_columns']
results={r['tag']:r for r in gate['results']};assert len(results)==16
getter_corrections=[]
values={};shared=None;actual_files={};types=dict(original=0,explicit=0,capacity=0,local=0,repeat=0)
original_config=dict(read_config(c['original_base_config']),threads=c['threads'])
for tag,r in results.items():
    spec=r['spec'];weighting=r['weighting'];train=whole;ev=future
    if spec is not None:
        p,level=spec['partition'],spec['level'];train=whole[frame.loc[whole,p].eq(level)];ev=future[frame.loc[future,p].eq(level)]
        assert len(train)==12000 and int(frame.loc[train,'fraud_bool'].sum())>=100
        assert int((1-frame.loc[train,'fraud_bool']).sum())>=100
    costs=audit.direct_class_costs(frame.loc[whole,'fraud_bool'],weighting=='balanced')
    assert r['class_costs']==costs and r['local_rows']==len(train) and r['local_fraud']==int(frame.loc[train,'fraud_bool'].sum())
    weights=np.asarray([costs[str(int(y))] for y in frame.loc[train,'fraud_bool']],dtype='<f8')
    assert r['local_weight_sha256']==hashlib.sha256(weights.tobytes()).hexdigest()
    assert r['warnings']==[]
    model_path,pred_path=list(r['files']);actual_files.update(r['files'])
    assert digest(model_path)==r['files'][model_path] and digest(pred_path)==r['files'][pred_path]
    pipe=joblib.load(model_path);model=pipe.named_steps['model'];meta=run.scores.bagged.audit_preprocessing(pipe,base.loc[whole])
    assert r['preprocessing']==meta
    if shared is None:shared=meta
    assert meta==shared
    expected=(make_model('lightgbm',weighting=='balanced',original_config,base.columns).named_steps['model']
              if r['original_class_weight'] else run.scores.estimator(c,200 if spec is not None or '200_' in tag else 400))
    actual_params=run.scores.signature(model);declared_params=run.scores.signature(expected)
    assert actual_params==r['requested_model_params']
    if actual_params!=declared_params:
        # Preserve and explain the frozen original-miniature getter artifact.
        # This is not permission to alter any effective training parameter.
        assert tag=='original200_balanced'
        assert {k for k in declared_params if actual_params[k]!=declared_params[k]}=={'class_weight'}
        assert actual_params['class_weight'] is None and declared_params['class_weight']=='balanced'
        assert model.class_weight==model._class_weight=='balanced' and model._other_params['class_weight'] is None
        probe=make_model('lightgbm',True,original_config,base.columns).named_steps['model']
        assert probe.get_params()['class_weight']=='balanced'
        shallow=copy.copy(probe);assert shallow._other_params is probe._other_params
        shallow.set_params(class_weight=None)
        assert probe.class_weight=='balanced' and probe.get_params()['class_weight'] is None
        from lightgbm import LGBMModel
        assert 'self._class_weight = self.class_weight' in inspect.getsource(LGBMModel.fit)
        assert 'class_sample_weight = _LGBMComputeSampleWeight(self._class_weight, y)' in inspect.getsource(LGBMModel.fit)
        assert 'params.pop("class_weight", None)' in inspect.getsource(LGBMModel._process_params)
        getter_corrections.append(dict(tag=tag,raw_postinspection_getter=actual_params,
            original_requested_recipe=declared_params,fitted_class_weight_attribute=model.class_weight,
            retained_fit_class_weight=model._class_weight,getter_only_metadata_defect=True,
            installed_lightgbm_source_sha256=digest(inspect.getsourcefile(LGBMModel)),
            miniature_source_sha256=fp['sources']['baf/conditional_expert_study.py'],
            production_source_and_artifacts_changed=False,new_fits=0))
    else:
        assert model.class_weight==declared_params['class_weight']
    assert model.booster_.params==r['booster_params'] and model.booster_.current_iteration()==r['fitted_tree_count']
    assert 1<=r['fitted_tree_count']<=expected.n_estimators
    with np.load(pred_path) as saved:
        d={k:saved[k] for k in ['row_ids','labels','scores']}
    assert np.array_equal(d['row_ids'],ev.to_numpy()) and np.array_equal(d['labels'],frame.loc[ev,'fraud_bool'].to_numpy())
    assert d['scores'].ndim==1 and np.isfinite(d['scores']).all() and ((d['scores']>=0)&(d['scores']<=1)).all()
    with threadpool_limits(limits=c['threads']):
        actual=pipe.predict_proba(base.loc[ev])[:,1]
        reverse=pipe.predict_proba(base.loc[ev[::-1]])[:,1][::-1]
    assert np.allclose(actual,d['scores'],atol=1e-12,rtol=0) and np.allclose(reverse,d['scores'],atol=1e-12,rtol=0)
    values[tag]=d
    kind=('repeat' if tag.startswith('repeat_') else 'original' if tag.startswith('original200_') else
          'explicit' if tag.startswith('explicit200_') else 'local' if spec is not None else 'capacity')
    types[kind]+=1
assert types==dict(original=2,explicit=2,capacity=2,local=8,repeat=2)
assert actual_files==gate['artifacts'] and len(actual_files)==32
assert {str(p) for p in (out/'miniature').glob('*')}==set(actual_files)
for w in ['balanced','unweighted']:
    assert np.allclose(values['original200_'+w]['scores'],values['explicit200_'+w]['scores'],atol=1e-12,rtol=0)
repeated=results['repeat_expert']['spec']['id']
assert np.array_equal(values['repeat_expert']['scores'],values[repeated]['scores'])
assert np.array_equal(values['repeat_capacity']['scores'],values['capacity400_balanced']['scores'])
assert len(getter_corrections)==1
write_json(dest/'miniature-parameter-metadata-correction.json',dict(
    fingerprint=fp,corrections=getter_corrections,original_artifacts_preserved=True,
    exact_original_explicit_prediction_equivalence=True,numerical_recipe_relaxed=False,
    new_model_fits=0,created_at_utc=run.now()))
direct_routes=[]
for p in audit.PARTITIONS:
    for w in ['balanced','unweighted']:
        local={r['spec']['level']:values[tag] for tag,r in results.items() if r['spec'] is not None and
               r['spec']['partition']==p and r['weighting']==w and not tag.startswith('repeat_')}
        d,routes=run.scores.stitch(frame,2,p,values['original200_'+w],local,list(local))
        expected=values['original200_'+w]['scores'].copy();b=audit.buckets(frame,future.to_numpy(),p)
        for level,v in local.items():expected[b==level]=v['scores']
        assert np.array_equal(d['scores'],expected)
        for route in routes:
            assert route['rows']==int(np.sum(b==route['level']))
            direct_routes.append(dict(route,weighting=w))
assert direct_routes==gate['routing']
for folder in ['models','runs','predictions']:
    assert not any((out/folder).glob('*'))
result=dict(status='passed',fingerprint=fp,declaration_sha256=digest(out/'declaration.json'),
    resource_preflight_sha256=digest(out/'resource_preflight.json'),miniature_sha256=digest(out/'miniature_gate.json'),
    source_gate_sha256=digest(out/'source_miniature_gate.json'),fit_types=types,new_miniature_fits=16,
    preserved_old_fits=17,new_old_gate_fits=0,saved_models_inference_checked=16,
    shared_global_preprocessing=True,whole_training_costs=True,original_explicit_equivalence=True,
    deterministic_repeats=True,all_routes_reconstructed=True,files=actual_files,
    getter_metadata_correction_sha256=digest(dest/'miniature-parameter-metadata-correction.json'),
    original_balanced_fit_state_verified=True,numerical_recipe_relaxed=False,
    production_full_data_fits_started=False,month7_access=False,created_at_utc=run.now())
write_json(dest/'production-gates-verification.json',result)
print('Verified resource gate, preserved 9/8-fit evidence and all sixteen saved miniature models; no full-data fit or new month-7 access yet.')
