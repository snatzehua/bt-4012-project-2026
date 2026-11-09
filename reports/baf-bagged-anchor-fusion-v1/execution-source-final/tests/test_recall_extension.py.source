import copy
import unittest
import numpy as np

from baf.recall_extension_metrics import choose,leaves,mean_scores,percentile_scores,rank_blend
from baf.recall_extension import normalize_original,verify_training


def data(y=(1,0,1,0),s=(.8,.7,.3,.1),ids=(1,2,3,4)):
    return dict(labels=np.array(y),scores=np.array(s),row_ids=np.array(ids))


class ExtensionTests(unittest.TestCase):
    def test_percentile_average_ties_and_scale_invariance(self):
        s=np.array([.1,.5,.5,.9])
        np.testing.assert_allclose(percentile_scores(s),[.125,.5,.5,.875])
        np.testing.assert_equal(percentile_scores(s),percentile_scores(100*s+4))

    def test_normalization_never_uses_labels(self):
        a=data();b=data(s=(.9,.4,.2,.1));r=rank_blend(a,b,.5)
        a2=data(y=(0,1,0,1));b2=data(y=(0,1,0,1),s=(.9,.4,.2,.1))
        np.testing.assert_equal(r['scores'],rank_blend(a2,b2,.5)['scores'])

    def test_rank_blend_and_seed_mean_alignment(self):
        a=data();b=data(s=(.1,.2,.7,.8))
        expected=.25*percentile_scores(a['scores'])+.75*percentile_scores(b['scores'])
        np.testing.assert_equal(rank_blend(a,b,.25)['scores'],expected)
        np.testing.assert_allclose(mean_scores([a,b])['scores'],(a['scores']+b['scores'])/2)
        with self.assertRaises(ValueError): mean_scores([a,data(ids=(2,1,3,4))])
        with self.assertRaises(ValueError): rank_blend(a,b,1.1)

    def test_choose_rejects_current_month(self):
        self.assertEqual(choose({'a':{4:data()}},['a'],5),'a')
        with self.assertRaises(ValueError): choose({'a':{5:data()}},['a'],5)

    def test_recursive_seed_and_rank_dependencies(self):
        config={'seed_means':[dict(id='mean',members=['a','b','c'])]}
        recipes={'rank':dict(kind='rank_blend',anchor='base',member='mean')}
        self.assertEqual(leaves('rank',config,recipes),{'a','b','c','base'})
        self.assertEqual(leaves('a',config,recipes),{'a'})

    def test_original_metadata_must_match_recipe(self):
        spec=dict(id='window',original_spec=dict(model='lightgbm',strategy='recent',k=2,seed=17,weighted=False))
        raw={**spec['original_spec'], 'train_row_ids_sha256':'a','train_months':{'3':2}}
        result=normalize_original(raw,spec,{'config':'new'})
        self.assertEqual(result['train_month_max'],3)
        with self.assertRaises(ValueError): normalize_original({**raw,'seed':29},spec,{})

    def test_training_hash_count_and_time_checks(self):
        import pandas as pd
        from baf.recall_weighted import ids_digest
        frame=pd.DataFrame(dict(month=[0,1,2],fraud_bool=[1,0,1]))
        chosen=np.array([0,1]);r=dict(train_ids_sha256=ids_digest(chosen),train_rows=2,train_fraud=1,train_month_max=1)
        verify_training(r,chosen,frame,2)
        for key,value in [('train_fraud',2),('train_rows',3),('train_ids_sha256','bad'),('train_month_max',2)]:
            with self.assertRaises(ValueError): verify_training({**r,key:value},chosen,frame,2)

    def test_percentile_invalid_inputs(self):
        for s in [[],[1,np.nan],[[1,2]]]:
            with self.assertRaises(ValueError): percentile_scores(s)

    def test_guard_rejects_software_and_original_source_drift(self):
        from unittest.mock import patch
        from baf.recall_extension import guard
        cfg=dict(previous_f2_config='prior',previous_f2_fingerprint={'software':'old'},original_config='original',original_identity={'source':'old'})
        with patch('baf.recall_extension.prior_guard'),patch('baf.recall_extension.previous_fingerprint',return_value={'software':'new'}):
            with self.assertRaisesRegex(ValueError,'software changed'):guard(cfg)
        with patch('baf.recall_extension.prior_guard'),patch('baf.recall_extension.previous_fingerprint',return_value={'software':'old'}),patch('baf.recall_extension.original_identity',return_value={'source':'new'}):
            with self.assertRaisesRegex(ValueError,'recipe changed'):guard(cfg)

    def test_missing_window_cache_refits_only_in_new_directory(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        from baf.common import read_config,write_json
        from baf.improvements import save_prediction
        from baf.study import identity,run_name,train_indices
        from baf.recall_extension import run_base
        from baf.recall_weighted import ids_digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=read_config('configs/study.json');old['output_dir']=str(root/'old-output')
            old_path=root/'original.json';write_json(old_path,old)
            config=dict(output_dir=str(root/'new-output'),original_config=str(old_path),original_identity=identity(old_path))
            frame=pd.DataFrame(dict(month=[0,0,1,1,2,2],fraud_bool=[1,0,1,0,1,0]))
            spec=dict(id='fixture_window',family='window',engineered=False,original_spec=dict(model='lightgbm',strategy='recent',k=2,seed=17,weighted=False))
            def fake_fit(f,x,eligible,eval_ids,recipe,month,phase,fit_config,frozen):
                self.assertNotEqual(fit_config['output_dir'],old['output_dir'])
                chosen=train_indices(f,eligible,recipe,fit_config);stem=run_name(phase,month,recipe)
                rec=dict(**recipe,eval_month=month,phase=phase,train_rows=len(chosen),train_fraud=1,train_months={'1':2},
                         train_row_ids_sha256=ids_digest(chosen),evaluation_ids_sha256=ids_digest(eval_ids),fingerprint=frozen,
                         warnings=[],average_precision=1.,roc_auc=1.)
                write_json(Path(fit_config['output_dir'])/'runs'/(stem+'.json'),rec)
                save_prediction(Path(fit_config['output_dir'])/'predictions'/(stem+'.npz'),eval_ids,np.array([1,0]),np.array([.9,.1]))
            with patch('baf.recall_extension.original_fit',side_effect=fake_fit):
                result=run_base(frame,frame,frame,spec,2,'validation',config,{'config':'new'})
            self.assertFalse(result['reused'])
            self.assertFalse((root/'old-output').exists())
            self.assertIn('original-refits',result['provenance']['record'])


if __name__=='__main__':unittest.main()
