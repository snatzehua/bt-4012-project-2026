import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from baf.common import read_config,write_json
from baf.improvements import paths,save_prediction
from baf.stacking import base_pairs,fit_meta,guard,meta_paths
from baf.stacking_features import matched_control,policy_history,score_matrix,training_matrix,verify_source


def data(month=1,labels=(1,0,1,0),scores=(.8,.7,.3,.1)):
    return dict(row_ids=np.arange(month*4,(month+1)*4),labels=np.array(labels),scores=np.array(scores))


class StackingTests(unittest.TestCase):
    def test_declared_grid_and_matched_controls(self):
        c=read_config('configs/chronological-stacking.json')
        self.assertEqual(len(base_pairs(c)),27);self.assertEqual(len(c['combiners']),24)
        for spec in c['combiners']:
            ctrl=matched_control(spec,c['combiners'])
            if spec['group']=='stack':
                self.assertEqual(ctrl['members'],['baseline_balanced'])
                for key in ['C','representation','pos_weight']:self.assertEqual(ctrl[key],spec[key])
            else:self.assertIsNone(ctrl)

    def test_clipping_and_fixed_member_column_order(self):
        a=data(scores=(0,1,.5,.8));b=data(scores=(.2,.3,.4,.5))
        x,cols,_=score_matrix({'a':a,'b':b},['b','a'],'logit')
        self.assertEqual(cols,['b__logit','a__logit']);self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(x[0,1],np.log(1e-6/(1-1e-6)))
        with self.assertRaises(ValueError):score_matrix({'a':a},['a'],'unknown')

    def test_percentile_features_do_not_use_labels_and_average_ties(self):
        a=data(scores=(.1,.5,.5,.9));b=data(labels=(0,1,0,1),scores=(.1,.5,.5,.9))
        x,_,_=score_matrix({'a':a},['a'],'logit_percentile')
        y,_,_=score_matrix({'a':b},['a'],'logit_percentile')
        np.testing.assert_equal(x,y);np.testing.assert_allclose(x[:,1],[.125,.5,.5,.875])

    def test_misaligned_ids_labels_and_invalid_scores_rejected(self):
        a=data();b=data();b['row_ids']=b['row_ids'][::-1]
        with self.assertRaises(ValueError):score_matrix({'a':a,'b':b},['a','b'],'logit')
        b=data(labels=(0,1,0,1))
        with self.assertRaises(ValueError):score_matrix({'a':a,'b':b},['a','b'],'logit')
        with self.assertRaises(ValueError):score_matrix({'a':data(scores=(-1,.5,.3,.2))},['a'],'logit')

    def test_nested_training_uses_only_earlier_prediction_rows(self):
        history={'a':{m:data(m) for m in [1,2,3,4]}}
        records={('a',m):dict(eval_month=m,train_month_max=m-1) for m in [1,2,3,4]}
        spec=dict(members=['a'],representation='logit')
        x,cols,pop,months=training_matrix(history,records,spec,4)
        self.assertEqual(months,[1,2,3]);self.assertEqual(len(x),12)
        self.assertFalse(np.intersect1d(pop['row_ids'],history['a'][4]['row_ids']).size)
        records[('a',2)]['train_month_max']=2
        with self.assertRaises(ValueError):training_matrix(history,records,spec,4)

    def test_policy_inputs_must_be_earlier_and_out_of_time(self):
        h={'meta':{m:data(m) for m in [4,5,6]}}
        r={('meta',m):dict(eval_month=m,train_month_max=m-1) for m in [4,5,6]}
        self.assertEqual(list(policy_history(h,r,'meta',5,[4,5,6])),[4])
        self.assertEqual(list(policy_history(h,r,'meta',6,[4,5,6])),[4,5])
        r[('meta',4)]['in_sample']=True
        with self.assertRaises(ValueError):policy_history(h,r,'meta',5,[4,5,6])
        with self.assertRaises(ValueError):verify_source(dict(eval_month=4,train_month_max=4),4)

    def test_preserved_extension_identity_mismatch_rejected(self):
        c=read_config('configs/chronological-stacking.json');c['extension_fingerprint']={'bad':'identity'}
        with self.assertRaisesRegex(ValueError,'extension identity'):guard(c)

    def test_real_combiner_scaling_cache_and_training_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(output_dir=tmp,clip=1e-6,threads=1,seed=2026)
            spec=dict(id='fixture',members=['a'],representation='logit',C=.1,pos_weight=1,group='single')
            h={'a':{m:data(m) for m in [1,2,3,4]}};r={}
            h['a'][4]=data(4,scores=(.99,.99,.98,.98))
            for m in [1,2,3,4]:
                r[('a',m)]=dict(eval_month=m,train_month_max=m-1)
                rp,pp=paths(c,'validation',m,'a');write_json(rp,r[('a',m)])
                d=h['a'][m];save_prediction(pp,d['row_ids'],d['labels'],d['scores'])
            record=fit_meta(c,spec,4,{},h,r)
            self.assertEqual(record['train_months'],[1,2,3]);self.assertEqual(record['train_rows'],12)
            self.assertEqual(fit_meta(c,spec,4,{},h,r),record)
            with self.assertRaisesRegex(ValueError,'provenance'):fit_meta(c,spec,4,{'changed':True},h,r)
            pp=paths(c,'validation',4,'fixture')[1]
            save_prediction(pp,h['a'][4]['row_ids'],h['a'][4]['labels'],np.zeros(4))
            with self.assertRaisesRegex(ValueError,'predictions changed'):fit_meta(c,spec,4,{},h,r)

    def test_missing_base_cache_refits_in_new_destination(self):
        import pandas as pd
        from baf.recall_extension import run_base
        from baf.recall_weighted import ids_digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old.json';spec=dict(id='base',family='lightgbm',engineered=False)
            write_json(old,dict(output_dir=str(root/'old-output'),candidates=[spec]))
            c=dict(output_dir=str(root/'new-output'),previous_f2_config=str(old),previous_f2_fingerprint={},prior_config=str(old),prior_fingerprint={})
            frame=pd.DataFrame(dict(month=[0,0,1,1,2,2],fraud_bool=[1,0,1,0,1,0]))
            def fake_fit(f,b,e,s,m,phase,cfg,identity):
                self.assertEqual(cfg['output_dir'],c['output_dir'])
                train=np.arange(4);ids=np.array([4,5]);save_prediction(paths(cfg,phase,m,s['id'])[1],ids,np.array([1,0]),np.array([.9,.1]))
                return dict(spec=s,fingerprint=identity,eval_month=m,train_month_max=1,train_ids_sha256=ids_digest(train),train_rows=4,train_fraud=2,
                            evaluation_ids_sha256=ids_digest(ids),average_precision=1.,roc_auc=1.,warnings=[])
            with patch('baf.recall_extension.fit_run',side_effect=fake_fit):result=run_base(frame,frame,frame,spec,2,'validation',c,{})
            self.assertFalse(result['reused']);self.assertFalse((root/'old-output').exists())

    def test_paired_uncertainty_recomputes_f2_on_shared_resampled_cases(self):
        from baf.stacking_report import paired_draws
        a=dict(row_ids=np.array([1,2]),labels=np.array([1,0]),scores=np.array([.9,.9]))
        b={**a,'scores':np.array([.1,.1])}
        rules={n:dict(kind='pooled',threshold=.5) for n in ['selected','reference']}
        table,draws=paired_draws(dict(selected=a,reference=b),rules,['reference'],np.random.default_rng(123),20)
        self.assertAlmostEqual(table.iloc[0].difference,5/6)
        self.assertTrue(all(any(np.isclose(v,x) for x in [0,5/6,1]) for v in draws[:,0]))
        _,same=paired_draws(dict(selected=a,reference=a),rules,['reference'],np.random.default_rng(123),20)
        np.testing.assert_equal(same,np.zeros((20,1)))


if __name__=='__main__':unittest.main()
