import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from baf.common import read_config,write_json,digest
from baf.improvements import paths,save_prediction
from baf.calibrated_policy import guard,ensure_base,apply_rule
from baf.calibrated_policy_metrics import (batch_decisions,calibrated,fit_calibrator,logit,
    calibration_population,hindsight_prefix,paired_draws,evaluate,reliability)
from baf.recall_weighted import ids_digest


def data(scores=(.9,.7,.3,.1),labels=(1,0,1,0),month=1):
    return dict(scores=np.array(scores,dtype=float),labels=np.array(labels),row_ids=np.arange(month*4,month*4+len(scores)))


class CalibratedPolicyTests(unittest.TestCase):
    def test_grid_and_preserved_source_guard(self):
        c=read_config('configs/calibrated-f2-policy.json');guard(c)
        self.assertEqual(len(c['score_sources'])*len(c['calibration_policies'])*2,24)
        self.assertEqual(len(c['score_sources'])*(len(c['calibration_policies'])+len(c['policies']))*2,42)
        c['stack_fingerprint']={}
        with self.assertRaisesRegex(ValueError,'stacking identity'):guard(c)

    def test_overwrite_destination_rejected(self):
        c=read_config('configs/calibrated-f2-policy.json');c['output_dir']='outputs/baf-chronological-stacking-v1'
        with self.assertRaisesRegex(ValueError,'overlaps'):guard(c)

    def test_surrogate_matches_tiny_brute_force_and_smallest_tie(self):
        from itertools import combinations
        p=np.array([.8,.6,.2,.1]);raw=p.copy();ids=np.arange(4)
        mask,info=batch_decisions(p,raw,ids)
        options=[(0.,0)]
        for k in range(1,5):
            best=max(5*p[list(s)].sum()/(4*p.sum()+k) for s in combinations(range(4),k))
            options.append((best,k))
        best=min(options,key=lambda x:(-x[0],x[1]))
        self.assertEqual(info['k'],best[1]);self.assertAlmostEqual(info['surrogate'],best[0])
        np.testing.assert_equal(mask,np.arange(4)<best[1])

    def test_zero_one_probabilities_and_flat_blocks(self):
        p=np.zeros(4);mask,info=batch_decisions(p,np.array([.1,.9,.8,.4]),np.arange(4))
        self.assertFalse(mask.any());self.assertEqual(info['k'],0)
        mask,info=batch_decisions(np.ones(4),np.arange(4)/4,np.arange(4))
        self.assertTrue(mask.all());self.assertEqual(info['surrogate'],1.)
        p=np.array([.7,.7,0,0]);mask,_=batch_decisions(p,np.array([.3,.8,.1,.1]),np.array([4,3,2,1]))
        self.assertTrue(mask[1]);self.assertFalse(mask[2])
        with self.assertRaises(ValueError):batch_decisions([],[],[])
        with self.assertRaises(ValueError):batch_decisions([.1,.2],[.1,.2],[1,1])

    def test_tie_rule_and_current_labels_never_enter_batch_api(self):
        a=data(labels=(1,0,1,0));b=data(labels=(0,1,0,1))
        ma,ia=batch_decisions(a['scores'],a['scores'],a['row_ids'])
        mb,ib=batch_decisions(b['scores'],b['scores'],b['row_ids'])
        np.testing.assert_equal(ma,mb);self.assertEqual(ia,ib)
        with self.assertRaises(TypeError):batch_decisions(a['scores'],a['scores'],a['row_ids'],labels=a['labels'])

    def test_clipping_scaling_and_calibration_chronology(self):
        d=data(scores=(0,.3,.7,1),labels=(0,0,1,1));model=fit_calibrator(d,'sigmoid',threads=1)
        self.assertTrue(np.isfinite(logit(d['scores'])).all())
        np.testing.assert_allclose(model['model'].named_steps['standardscaler'].mean_,logit(d['scores']).mean(axis=0))
        p=calibrated(model,d['scores']);self.assertTrue((np.diff(p)>=0).all())
        history={4:data(month=4),5:data(month=5)}
        pop,months=calibration_population(history,6,'pooled');self.assertEqual(months,[4,5]);self.assertEqual(len(pop['labels']),8)
        _,months=calibration_population(history,6,'latest');self.assertEqual(months,[5])
        with self.assertRaises(ValueError):calibration_population(history,5,'pooled')
        with self.assertRaises(ValueError):calibration_population({},5,'latest')

    def test_one_class_empty_negative_slope_and_isotonic_clipping(self):
        for label in [0,1]:
            d=data(labels=(label,)*4);m=fit_calibrator(d,'sigmoid');np.testing.assert_equal(calibrated(m,d['scores']),np.full(4,label))
        with self.assertRaises(ValueError):fit_calibrator(dict(scores=np.array([]),labels=np.array([]),row_ids=np.array([])),'sigmoid')
        with self.assertRaisesRegex(ValueError,'Negative sigmoid'):fit_calibrator(data(labels=(0,0,1,1)),'sigmoid',threads=1)
        m=fit_calibrator(data(scores=(.2,.4,.6,.8),labels=(0,0,1,1)),'isotonic')
        np.testing.assert_equal(calibrated(m,np.array([0.,1.])),[0,1])
        self.assertTrue(np.all(np.diff(calibrated(m,np.linspace(0,1,20)))>=0))

    def test_cache_adapter_checks_training_and_prediction_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=dict(output_dir=str(root/'old'));write_json(root/'old.json',old)
            c=dict(output_dir=str(root/'new'),stack_config=str(root/'old.json'),stack_fingerprint={'old':True})
            spec=dict(id='baseline_lgbm',family='baseline');frame=pd.DataFrame(dict(month=[0,0,1,1],fraud_bool=[1,0,1,0]))
            rp,pp=paths(old,'validation',1,spec['id']);save_prediction(pp,np.array([2,3]),np.array([1,0]),np.array([.9,.1]))
            raw=dict(spec=spec,fingerprint=c['stack_fingerprint'],eval_month=1,train_month_max=0,
                     train_ids_sha256=ids_digest([0,1]),train_rows=2,train_fraud=1,evaluation_ids_sha256=ids_digest([2,3]),
                     prediction_sha256=digest(pp),average_precision=1.,roc_auc=1.,warnings=[])
            write_json(rp,raw);before={str(p):digest(p) for p in [rp,pp]}
            record=ensure_base(frame,None,None,spec,1,'validation',c,{'new':True})
            self.assertTrue(record['reused']);self.assertEqual(record['fingerprint'],{'new':True})
            ownpp=paths(c,'validation',1,spec['id'])[1];save_prediction(ownpp,np.array([2,3]),np.array([1,0]),np.array([.8,.1]))
            with self.assertRaisesRegex(ValueError,'hash'):ensure_base(frame,None,None,spec,1,'validation',c,{'new':True})
            self.assertEqual(before,{str(p):digest(p) for p in [rp,pp]})

    def test_missing_score_route_uses_new_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=dict(output_dir=str(root/'old'));write_json(root/'old.json',old)
            c=dict(output_dir=str(root/'new'),stack_config=str(root/'old.json'))
            with patch('baf.calibrated_policy.run_base',return_value={'reused':False}) as fit:
                result=ensure_base(None,None,None,dict(id='x'),1,'validation',c,{})
                self.assertEqual(fit.call_args.args[-2]['output_dir'],str(root/'new'));self.assertFalse(result['reused'])
                self.assertFalse((root/'old').exists())

    def test_actual_frozen_base_recipe_refits_absent_caches_in_new_folder(self):
        from baf.common import CATEGORICAL
        rng=np.random.default_rng(2026);n=1200
        frame=pd.DataFrame(dict(month=np.repeat([0,1,2],400),fraud_bool=np.tile([1,0,0,0],300)))
        base=pd.DataFrame({'x':frame.fraud_bool.to_numpy()+rng.normal(size=n)})
        for col in CATEGORICAL:base[col]=np.where(rng.random(n)<.5,'a','b')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=read_config('configs/calibrated-f2-policy.json');spec=c['candidates'][0]
            old=read_config(c['stack_config']);old['output_dir']=str(root/'old-stack');write_json(root/'old-stack.json',old)
            prior=dict(output_dir=str(root/'old-prior'),candidates=[]);write_json(root/'old-prior.json',prior)
            c.update(output_dir=str(root/'new-study'),stack_config=str(root/'old-stack.json'),
                     previous_f2_config=str(root/'old-prior.json'),prior_config=str(root/'old-prior.json'))
            result=ensure_base(frame,base,base,spec,2,'validation',c,{'fixture':True})
            self.assertFalse(result['reused']);self.assertEqual(result['train_rows'],800)
            self.assertEqual(result['train_month_max'],1);self.assertTrue(paths(c,'validation',2,spec['id'])[1].exists())
            self.assertFalse((root/'old-stack').exists());self.assertFalse((root/'old-prior').exists())
            again=ensure_base(frame,base,base,spec,2,'validation',c,{'fixture':True})
            self.assertEqual(result,again)

    def test_independent_metrics_reliability_and_hindsight(self):
        d=data();mask=np.array([True,False,False,False]);m=evaluate(d,mask)
        self.assertAlmostEqual(m['f2'],5/9);self.assertEqual(m['tp'],1);self.assertEqual(m['fn'],1)
        bins=reliability(d,d['scores'],2);self.assertEqual(sum(b['n'] for b in bins),4)
        opt=hindsight_prefix(d);self.assertGreaterEqual(opt['f2'],m['f2']);self.assertIn('not deployable',opt['diagnostic'])

    def test_diagnostic_is_after_freeze_and_cannot_change_selection(self):
        from baf.calibrated_policy import diagnostics
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(output_dir=tmp,score_sources=['a','b','c']);selection=dict(created_at_utc='2000-01-01T00:00:00+00:00')
            sp=Path(tmp)/'selection.json';write_json(sp,selection);before=digest(sp)
            h={n:{m:data(month=m) for m in [5,6]} for n in c['score_sources']}
            with patch('baf.calibrated_policy.load_selection',return_value=(c,selection)),patch('baf.calibrated_policy.histories',return_value=(h,{})):
                diagnostics();self.assertEqual(digest(sp),before)
                table=pd.read_csv(Path(tmp)/'hindsight.csv');self.assertEqual(len(table),6);self.assertEqual(set(table.eval_month),{5,6})
                rec=read_config(Path(tmp)/'diagnostic_provenance.json');self.assertTrue(rec['excluded_from_selection'])
                rec['created_at_utc']='1999-01-01T00:00:00+00:00';write_json(Path(tmp)/'diagnostic_provenance.json',rec)
                with self.assertRaisesRegex(ValueError,'boundary'):diagnostics()

    def test_saved_calibrator_provenance_and_tamper_rejection(self):
        from baf.calibrated_policy import fit_calibration,cal_paths
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(output_dir=tmp,validation_months=[4,5,6],clip=1e-6,seed=2026,threads=1)
            h={'baseline_lgbm':{4:data(month=4),5:data(month=5)}};records={}
            for m in [4,5]:
                rec=dict(eval_month=m,train_month_max=m-1);records[('baseline_lgbm',m)]=rec
                rp,pp=paths(c,'validation',m,'baseline_lgbm');write_json(rp,rec)
                d=h['baseline_lgbm'][m];save_prediction(pp,d['row_ids'],d['labels'],d['scores'])
            model,rec=fit_calibration(c,{},h,records,'baseline_lgbm','sigmoid_pooled',5)
            self.assertEqual(rec['source_months'],[4]);self.assertEqual(rec['train_rows'],4)
            model2,rec2=fit_calibration(c,{},h,records,'baseline_lgbm','sigmoid_pooled',5,False);self.assertEqual(rec,rec2)
            np.testing.assert_equal(calibrated(model,h['baseline_lgbm'][5]['scores']),calibrated(model2,h['baseline_lgbm'][5]['scores']))
            rp,mp=cal_paths(c,'baseline_lgbm','sigmoid_pooled',5);mp.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'artifact changed'):fit_calibration(c,{},h,records,'baseline_lgbm','sigmoid_pooled',5,False)

    def test_bootstrap_matches_shared_explicit_resampling(self):
        y=np.array([1,0,1,0]);masks=dict(selected=np.array([1,1,1,0]),ref=np.array([0,1,1,0]))
        actual=paired_draws(y,masks,np.random.default_rng(42),10);rng=np.random.default_rng(42)
        for row in actual:
            idx=rng.integers(0,4,4)
            for name,mask in masks.items():
                tp=int((y[idx]*mask[idx]).sum());fp=int(((1-y[idx])*mask[idx]).sum());fn=int(y[idx].sum()-tp)
                expected=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0
                self.assertAlmostEqual(row[name],expected)
        same=paired_draws(y,dict(selected=masks['selected'],ref=masks['selected']),np.random.default_rng(42),10)
        self.assertTrue(all(r['selected']==r['ref'] for r in same))


if __name__=='__main__':unittest.main()
