import copy
import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from baf.common import read_config,write_json,digest,CATEGORICAL
from baf.focal_loss import terms,objective,estimator,scores,integration_gate
from baf.focal_study import (guard,declared_presets,fit_loss,audit_loss,model_paths,control_names,
    paths,reference_config,prepare_references)
from baf.recall_weighted_metrics import learn


class FocalLossTests(unittest.TestCase):
    def test_true_derivatives_against_central_finite_differences(self):
        y=np.tile([0,1],5);z=np.repeat([-4.,-1.,0.,1.,4.],2);step=1e-4
        for gamma in [0,1,2]:
            for weight in [1,3]:
                l,g,h=terms(y,z,gamma,weight);plus=terms(y,z+step,gamma,weight);minus=terms(y,z-step,gamma,weight)
                np.testing.assert_allclose(g,(plus[0]-minus[0])/(2*step),atol=1e-8,rtol=2e-6)
                np.testing.assert_allclose(h,(plus[1]-minus[1])/(2*step),atol=1e-8,rtol=2e-6)
                np.testing.assert_allclose(h,(plus[0]-2*l+minus[0])/step**2,atol=5e-7,rtol=5e-5)

    def test_gamma_zero_is_weighted_logistic(self):
        y=np.array([0,1,0,1]);z=np.array([-8.,-2.,2.,8.]);p=expit(z);w=np.where(y==1,3.,1.)
        l,g,h=terms(y,z,0,3)
        np.testing.assert_allclose(l,w*(np.logaddexp(0,z)-y*z),atol=1e-14)
        np.testing.assert_allclose(g,w*(p-y),atol=1e-14)
        np.testing.assert_allclose(h,w*p*(1-p),atol=1e-14)

    def test_extreme_margins_and_invalid_inputs(self):
        for gamma in [0,1,2]:
            outputs=terms([0,1,0,1],[-1000,-1000,1000,1000],gamma,3)
            self.assertTrue(all(np.isfinite(a).all() for a in outputs));self.assertTrue((outputs[0]>=0).all())
        for args in [([],[],2,1),([2],[0],2,1),([1],[np.nan],2,1),([1],[0],-1,1),([1],[0],2,0)]:
            with self.assertRaises(ValueError):terms(*args)

    def test_curvature_floor_is_explicit_and_does_not_change_gradient(self):
        y=np.array([1,0,1,0]);z=np.array([-5.,5.,0.,0.]);l,g,h=terms(y,z,2,3)
        self.assertTrue((h<0).any());trace=[];grad,supplied=objective(2,3,1e-6,trace)(y,z)
        np.testing.assert_equal(grad,g);np.testing.assert_equal(supplied,np.maximum(h,1e-6))
        self.assertEqual(trace[0]['replaced_count'],int((h<1e-6).sum()));self.assertEqual(trace[0]['iteration'],1)
        self.assertEqual(trace[0]['approximation'],'positive diagonal curvature floor')
        trace=[];g,h=objective(0,1,1e-6,trace)(np.array([1]),np.array([100.]))
        self.assertLess(h[0],1e-6);self.assertEqual(trace[0]['replaced_count'],0)
        with self.assertRaises(ValueError):objective(2,1,0,[])

    def test_declared_grid_matches_controls_and_six_preserved_identities(self):
        c=read_config('configs/focal-loss.json');guard(c)
        self.assertEqual(c['presets'],declared_presets());self.assertEqual(len(c['presets'])*3,24)
        self.assertEqual(len(c['references'])*3,9);self.assertEqual((len(c['presets'])+len(c['references']))*3*2,66)
        for spec in c['presets']:
            names=control_names(c,spec['id'])
            if spec['gamma']>0:self.assertEqual(names,[f"custom_g0_w{spec['positive_weight']}",f"builtin_w{spec['positive_weight']}"])
            else:self.assertEqual(len(names),1)
        self.assertEqual(control_names(c,'baseline_balanced'),[])
        c['source_fingerprint']={}
        with self.assertRaisesRegex(ValueError,'Sixth'):guard(c)

    def test_overwrite_and_altered_reference_rejected(self):
        c=read_config('configs/focal-loss.json');c['output_dir']='outputs/baf-calibrated-f2-policy-v1'
        with self.assertRaisesRegex(ValueError,'overlaps'):guard(c)
        c=read_config('configs/focal-loss.json');c['references'][2]['params']['num_leaves']=4
        with self.assertRaisesRegex(ValueError,'Reference recipe'):guard(c)

    def test_custom_builtin_integration_and_no_double_class_weights(self):
        c=read_config('configs/focal-loss.json');result=integration_gate(c)
        self.assertEqual(sum(r['miniature_fits'] for r in result),4)
        self.assertTrue(all(r['max_probability_difference']<=1e-6 for r in result))
        model=estimator(c['presets'][2],c,[])
        self.assertIsNone(model.class_weight);self.assertNotIn('scale_pos_weight',model.get_params())
        self.assertFalse(model.get_params()['boost_from_average'])

    def test_raw_predictions_receive_exactly_one_sigmoid(self):
        class Fixture:
            def predict(self,x,raw_score=False):
                if not raw_score:raise AssertionError('Raw flag required')
                return np.array([-3.,0.,3.])
        raw,p=scores(Fixture(),None);np.testing.assert_equal(p,expit(raw));self.assertEqual(p[1],.5)

    def test_real_frozen_recipe_cache_reconstruction_chronology_and_tampering(self):
        rng=np.random.default_rng(33);n=600
        frame=pd.DataFrame(dict(month=np.repeat([0,1,2],200),fraud_bool=np.tile([0,1,0,0],150)))
        base=pd.DataFrame(dict(x=frame.fraud_bool.to_numpy()+rng.normal(size=n)))
        for cat in CATEGORICAL:base[cat]=np.where(frame.month==2,'future',np.where(rng.random(n)<.5,'a','b'))
        with tempfile.TemporaryDirectory() as tmp:
            c=read_config('configs/focal-loss.json');c['output_dir']=tmp;c['tree_params'].update(n_estimators=8,min_child_samples=5)
            spec=c['presets'][2];record=fit_loss(c,{},frame,base,spec,2)
            self.assertEqual(record['train_rows'],400);self.assertEqual(record['train_month_max'],1)
            self.assertEqual(record['feature_names'],list(base.columns));self.assertEqual(fit_loss(c,{},frame,base,spec,2),record)
            mp,tp=model_paths(c,'validation',2,spec['id']);artifact=joblib.load(mp)
            self.assertNotIn('future',artifact['adapter'].levels[CATEGORICAL[0]])
            trace=read_config(tp);self.assertEqual(len(trace),8)
            self.assertEqual(trace[0]['n'],400)
            with self.assertRaisesRegex(ValueError,'identity'):audit_loss(c,{'changed':True},frame,base,spec,2)
            mp.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'artifact changed'):audit_loss(c,{},frame,base,spec,2)

    def test_missing_reference_reproduces_in_new_destination(self):
        from unittest.mock import patch
        c=read_config('configs/focal-loss.json');frame=pd.DataFrame(dict(month=[0,1],fraud_bool=[0,1]));base=pd.DataFrame(dict(x=[1,2]))
        with tempfile.TemporaryDirectory() as tmp:
            c['output_dir']=str(Path(tmp)/'new')
            with patch('baf.focal_study.ensure_base',return_value={'reused':False}) as reproduce, \
                 patch('baf.focal_study.engineer',return_value=base):
                result=prepare_references(c,{},frame,base,specs=c['references'][:1])
                self.assertEqual(len(result),3)
                for call in reproduce.call_args_list:self.assertEqual(call.args[-2]['output_dir'],c['output_dir'])
                self.assertNotEqual(reference_config(c)['output_dir'],read_config(c['source_config'])['output_dir'])

    def test_policies_refuse_current_month_labels(self):
        d=dict(row_ids=np.arange(4),labels=np.array([1,0,1,0]),scores=np.array([.9,.7,.3,.1]))
        with self.assertRaises(ValueError):learn({5:d},'pooled',5)
        self.assertEqual(learn({4:d},'pooled',5)['source_months'],[4])

    def test_development_bootstrap_is_paired_within_month_then_macro_averaged(self):
        from baf.focal_report import development
        from baf.calibrated_policy_metrics import evaluate
        h={name:{} for name in ['winner','control']};rows=[];rng=np.random.default_rng(17)
        rule=dict(kind='pooled',threshold=.5,beta=2)
        for month,n in [(5,7),(6,19)]:
            y=np.arange(n)%3==0
            for name in h:
                d=dict(row_ids=np.arange(n)+month*100,labels=y.astype(int),scores=rng.random(n))
                h[name][month]=d
                rows.append(dict(candidate=name,policy='pooled',eval_month=month,parameters=json.dumps(rule)))
        c=dict(bootstrap_seed=1027,bootstrap_replicates=50)
        s=dict(choices=dict(selected=['winner','pooled'],reference=['control','pooled']))
        table,draws=development(c,s,pd.DataFrame(rows),h)
        manual_rng=np.random.default_rng(1027);manual=[];point=[]
        for month,n in [(5,7),(6,19)]:
            data={alias:h[name][month] for alias,(name,_) in s['choices'].items()}
            masks={alias:d['scores']>=.5 for alias,d in data.items()};values=[]
            point.append(evaluate(data['selected'],masks['selected'])['f2']-evaluate(data['reference'],masks['reference'])['f2'])
            for _ in range(50):
                idx=manual_rng.integers(0,n,n);f2={}
                for alias,d in data.items():
                    y=d['labels'][idx];pred=masks[alias][idx]
                    tp=int((y*pred).sum());fp=int(((1-y)*pred).sum());fn=int(y.sum()-tp)
                    den=5*tp+4*fn+fp;f2[alias]=5*tp/den if den else 0.
                values.append(f2['selected']-f2['reference'])
            manual.append(np.array(values))
            np.testing.assert_allclose(draws[draws.scope==str(month)].reference,values,atol=1e-14)
        macro=table[table.scope=='macro'].iloc[0];expected=(manual[0]+manual[1])/2
        self.assertAlmostEqual(macro.difference,np.mean(point))
        self.assertAlmostEqual(macro.ci_low,np.quantile(expected,.025))
        self.assertAlmostEqual(macro.ci_high,np.quantile(expected,.975))


if __name__=='__main__':unittest.main()
