"""Independent output arithmetic and matrix checks; never fits or selects models."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import confusion_matrix,fbeta_score,precision_score,recall_score,average_precision_score,roc_auc_score
from baf.common import read_config,features,load_data,digest
from baf import newton_spline_study as study
from baf.improvements import prediction,paths
from baf.recall_weighted_metrics import decisions

c=read_config(study.CONFIG);study.guard(c);root=Path(c['output_dir'])
frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
solver=pd.read_csv(root/'solver.csv');coeff=pd.read_csv(root/'coefficients.csv');numerical=pd.read_csv(root/'numerical.csv')
assert len(solver)==36 and len(numerical)==36 and len(pd.read_csv(root/'feature_diagnostics.csv'))==2088
worst=0.;actual_models=[]
for m in [4,5,6]:
    ids=frame.index[frame.month<m];y=frame.loc[ids,'fraud_bool'].to_numpy()
    for spec in c['candidates']:
        a=joblib.load(study.model_path(c,'validation',m,spec['id']));model=a['model'];inp=a['inputs'];x=inp.transform(base.loc[ids]);d=prediction(c,'validation',m,spec['id'])
        z=np.asarray(x@model.coef_[0]).ravel()+model.intercept_[0]
        w=np.ones(len(y)) if spec['class_weight'] is None else len(y)/(2*np.bincount(y)[y])
        loss=np.average(np.logaddexp(0,z)-y*z,weights=w)+np.square(model.coef_[0]).sum()/(2*spec['C']*sum(w))
        residual=w*(expit(z)-y)
        g=np.r_[np.asarray(x.T@residual).ravel()/sum(w)+model.coef_[0]/(spec['C']*sum(w)),sum(residual)/sum(w)]
        worst=max(worst,float(abs(g).max()));assert abs(g).max()<=1.1e-6
        row=numerical[(numerical.candidate==spec['id'])&(numerical.eval_month==m)].iloc[0]
        assert np.isclose(row.objective,loss,atol=1e-12,rtol=0)
        assert row.fallback_count==0
        np.testing.assert_allclose(json.loads(row.gradient),g,atol=1e-12,rtol=0)
        r=solver[(solver.candidate==spec['id'])&(solver.eval_month==m)].iloc[0]
        assert r.transformed_dimension==x.shape[1] and r.matrix_nnz==x.nnz and r.iterations==model.n_iter_.max()
        assert np.isclose(r.matrix_density,x.nnz/(x.shape[0]*x.shape[1]),atol=1e-12,rtol=0)
        exported=coeff[(coeff.candidate==spec['id'])&(coeff.eval_month==m)]
        assert list(exported.feature)==[*inp.names,'__INTERCEPT__']
        np.testing.assert_allclose(exported.coefficient,np.r_[model.coef_[0],model.intercept_[0]],atol=1e-12,rtol=0)
        future=inp.transform(base.loc[d['row_ids']]);np.testing.assert_allclose(d['scores'],expit(np.asarray(future@model.coef_[0]).ravel()+model.intercept_[0]),atol=1e-12,rtol=1e-12)
        actual_models.append((m,spec['id']))
cf=pd.read_csv(root/'crossfit.csv')
for row in cf.to_dict('records'):
    d=prediction(c,'validation',row['eval_month'],row['candidate']);mask=decisions(d,json.loads(row['parameters']));y=d['labels']
    assert np.array_equal(d['row_ids'],frame.index[frame.month==row['eval_month']])
    tn,fp,fn,tp=confusion_matrix(y,mask,labels=[0,1]).ravel()
    for name,value in dict(tn=tn,fp=fp,fn=fn,tp=tp,n=len(y),precision=precision_score(y,mask,zero_division=0),recall=recall_score(y,mask,zero_division=0),f1=fbeta_score(y,mask,beta=1,zero_division=0),f2=fbeta_score(y,mask,beta=2,zero_division=0),f3=fbeta_score(y,mask,beta=3,zero_division=0),average_precision=average_precision_score(y,d['scores']),roc_auc=roc_auc_score(y,d['scores']),flagged_fraction=mask.mean()).items():
        # The project's AUROC field may use its full name.
        if name=='roc_auc':name='roc_auc' if name in row else 'roc_auc_score'
        if name in row:assert np.isclose(row[name],value,atol=1e-12,rtol=0),(name,row[name],value)
assert len(actual_models)==36 and len(cf)==90
assert len(list((root/'predictions').glob('validation-*.npz')))==48
assert len(list((root/'runs').glob('validation-*.json')))==48
assert not (root/'selection.json').exists() and not (root/'exploratory_access.json').exists()
comp=pd.read_csv(root/'old_solver_comparisons.csv');assert len(comp)==12
report=dict(checked_new_models=36,checked_forward_rows=90,checked_score_artifacts=48,maximum_independent_gradient=worst,optional_old_solver_models=int(comp.available.sum()),numerical_and_score_reconstruction=True,source_sha256=digest('/tmp/audit_newton_outputs.py'))
(root/'independent_development_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
