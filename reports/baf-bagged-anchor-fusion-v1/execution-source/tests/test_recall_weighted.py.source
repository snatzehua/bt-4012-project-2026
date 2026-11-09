"""Methodological checks for recall emphasis and chronological policy transfer."""
import copy
import unittest

import numpy as np
from sklearn.metrics import fbeta_score

from baf.recall_weighted import verify_record
from baf.recall_weighted_metrics import (aligned, blend, choose_member, decisions, fbeta,
                                         fit_policy, learn, measure, paired_f2)


def data(y=(1,0,1,0), s=(.8,.7,.3,.1), ids=(10,11,12,13)):
    return dict(labels=np.array(y), scores=np.array(s), row_ids=np.array(ids))


class RecallWeightedTests(unittest.TestCase):
    def test_fbeta_arithmetic(self):
        d = data()
        for beta in [1,2,3]:
            pred = np.array([1,1,1,0])
            self.assertAlmostEqual(float(fbeta(2,1,0,beta)), fbeta_score(d['labels'],pred,beta=beta))
        self.assertEqual(float(fbeta(0,0,0)),0)
        for beta in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError): fbeta(1,1,1,beta)

    def test_policy_finds_bruteforce_optimum(self):
        d=data()
        p=fit_policy(d)
        scores=[measure(d,dict(kind='pooled',threshold=t))['f2'] for t in d['scores']]
        self.assertAlmostEqual(measure(d,p)['f2'],max(scores))
        self.assertEqual(p['threshold'],.3)

    def test_threshold_ties_cannot_split_equal_scores(self):
        d=data(s=(.8,.8,.3,.1))
        pred=decisions(d,fit_policy(d))
        self.assertEqual(pred[0],pred[1])

    def test_rank_ties_use_row_ids(self):
        d=data(y=(0,1,0,0),s=(.5,.5,.1,.1),ids=(20,10,30,40))
        p=fit_policy(d,'rank_fraction')
        self.assertEqual(p['numerator'],1)
        np.testing.assert_equal(decisions(d,p),[False,True,False,False])

    def test_exact_fraction_ceiling(self):
        d=data()
        pred=decisions(d,dict(kind='rank_fraction',numerator=1,denominator=3))
        self.assertEqual(pred.sum(),2)
        pred=decisions(d,dict(kind='rank_fraction',numerator=0,denominator=3))
        self.assertEqual(pred.sum(),0)

    def test_degenerate_labels(self):
        for kind in ['pooled','latest','rank_fraction']:
            d=data(y=(0,0,0,0))
            self.assertEqual(decisions(d,fit_policy(d,kind)).sum(),0)
            d=data(y=(1,1,1,1))
            self.assertEqual(measure(d,fit_policy(d,kind))['recall'],1)

    def test_temporal_policy_sources(self):
        a=data(); b=data(ids=(20,21,22,23))
        self.assertEqual(learn({4:a},'pooled',5)['source_months'],[4])
        self.assertEqual(learn({4:a,5:b},'latest',6)['source_months'],[5])
        self.assertEqual(learn({4:a,5:b},'pooled',6)['source_months'],[4,5])
        with self.assertRaises(ValueError): learn({4:a,5:b},'pooled',5)
        with self.assertRaises(ValueError): learn({},'pooled',5)

    def test_member_selection_rejects_future(self):
        specs=[dict(id='a',family='lightgbm')]
        self.assertEqual(choose_member({'a':{4:data()}},specs,'lightgbm',5),'a')
        with self.assertRaises(ValueError): choose_member({'a':{5:data()}},specs,'lightgbm',5)

    def test_population_alignment_and_blend(self):
        a=data(); b=data(s=(.5,.4,.2,.1))
        np.testing.assert_allclose(blend(a,b,.25)['scores'],.25*a['scores']+.75*b['scores'])
        with self.assertRaises(ValueError): aligned(a,data(ids=(11,10,12,13)))
        with self.assertRaises(ValueError): aligned(a,data(y=(0,0,1,0)))
        with self.assertRaises(ValueError): fit_policy(data(s=(np.nan,.1,.2,.3)))

    def test_identity_and_temporal_record_guards(self):
        from baf.recall_weighted import ids_digest
        d=data(); ident={'config':'a'}; spec={'id':'a'}
        record=dict(fingerprint=ident,spec=spec,evaluation_ids_sha256=ids_digest(d['row_ids']),train_month_max=3,eval_month=4)
        verify_record(record,ident,spec,d,d)
        for key,value in [('fingerprint',{'config':'b'}),('spec',{'id':'b'}),('train_month_max',4)]:
            wrong={**record,key:value}
            with self.assertRaises(ValueError): verify_record(wrong,ident,spec,d,d)

    def test_preservation_guard(self):
        import tempfile
        from pathlib import Path
        from baf.recall_weighted import guard
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source';p.write_text('changed')
            cfg=dict(beta=2,policies=['pooled','latest','rank_fraction'],preserve_sha256={str(p):'wrong'})
            with self.assertRaisesRegex(ValueError,'Preserved artifact'): guard(cfg)

    def test_paired_identical_predictions_zero_interval(self):
        d=data(); p=fit_policy(d)
        row=paired_f2({'a':d,'b':copy.deepcopy(d)},{'a':p,'b':p},'a',replicates=30)[0]
        self.assertEqual((row['difference'],row['ci_low'],row['ci_high']),(0,0,0))


if __name__=='__main__': unittest.main()
