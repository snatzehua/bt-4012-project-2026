import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from baf.common import read_config,CATEGORICAL
from baf.direct_grouping import grouping,describe,integration_gate,raw_scores,estimator,fit_checked
from baf.direct_ranking import guard,declared_presets,control_names,fit_model,audit_model,model_paths,paths,prepare_references
from baf.recall_weighted_metrics import fit_policy,decisions,measure,learn
from baf.direct_ranking_report import development


class DirectRankingTests(unittest.TestCase):
    def test_complete_deterministic_grouping_and_singleton_tail(self):
        ids=np.arange(12);months=np.r_[np.zeros(9,dtype=int),np.ones(3,dtype=int)]
        order,g,gm=grouping(ids,months,4)
        np.testing.assert_equal(np.sort(order),ids);np.testing.assert_equal(g,[4,5,3]);np.testing.assert_equal(gm,[0,0,1])
        for a,b in zip((order,g,gm),grouping(ids[::-1],months[::-1],4)):np.testing.assert_equal(a,b)
        offset=0
        for size,m in zip(g,gm):
            self.assertTrue(np.all(months[order[offset:offset+size]]==m));offset+=size
        for invalid in [(np.array([0]),np.array([0]),4), (np.array([1,1]),np.array([0,0]),4),(ids,months,1)]:
            with self.assertRaises(ValueError):grouping(*invalid)

    def test_grouping_ignores_labels_and_size_does_not_change_order(self):
        ids=np.arange(1030);months=np.repeat([0,1],515);y=np.tile([0,0,0,0,1],206)
        a,ga,ma=grouping(ids,months,256);b,gb,mb=grouping(ids,months,1024)
        np.testing.assert_equal(a,b)
        first=describe(a,ga,ma,y[a]);second=describe(a,ga,ma,y[::-1][a])
        for key in ['ordered_ids_sha256','groups_sha256','group_months_sha256','groups']:self.assertEqual(first[key],second[key])
        self.assertEqual(sum(r['n'] for r in first['query_rows']),len(ids));self.assertEqual(sum(r['fraud'] for r in first['query_rows']),int(y.sum()))

    def test_zero_positive_queries_are_retained(self):
        order,g,gm=grouping(np.arange(20),np.zeros(20,dtype=int),4);y=np.zeros(20,dtype=int);y[order[0]]=1
        meta=describe(order,g,gm,y[order]);self.assertEqual(meta['queries'],5);self.assertEqual(meta['zero_positive_fraction'],.8)
        self.assertEqual(len(meta['query_rows']),5);self.assertEqual(sum(r['n'] for r in meta['query_rows']),20)
        with self.assertRaises(ValueError):describe(order,g,gm,np.full(20,2))

    def test_finite_grid_and_matched_control_inclusion(self):
        c=read_config('configs/direct-ranking.json');guard(c);self.assertEqual(c['presets'],declared_presets())
        self.assertEqual(len(c['presets'])*3,18);self.assertEqual((len(c['presets'])+len(c['references']))*3*2,54)
        for spec in c['presets']:
            names=control_names(c,spec['id']);self.assertEqual(len(names),1 if spec['kind']=='rank' else 2)
            for name in names:self.assertEqual(next(x for x in c['presets'] if x['id']==name)['reg_lambda'],spec['reg_lambda'])
        self.assertEqual(control_names(c,'baseline_balanced'),[])

    def test_preserved_identity_destination_and_data_are_guarded(self):
        for field,value in [('source_fingerprint',{}),('output_dir','outputs/baf-focal-loss-v1'),('data_sha256','bad'),('group_seed',17)]:
            c=read_config('configs/direct-ranking.json');c[field]=value
            with self.assertRaises(ValueError):guard(c)
        c=read_config('configs/direct-ranking.json');c['references'][0]['id']='changed'
        with self.assertRaisesRegex(ValueError,'Reference recipe'):guard(c)

    def test_installed_group_contract_repeatability_and_initialization(self):
        c=read_config('configs/direct-ranking.json');rows=integration_gate(c)
        self.assertEqual(sum(r['miniature_fits'] for r in rows),4)
        self.assertTrue(all(r['initial_margin_max']==r['repeat_max_difference']==0 for r in rows))
        model=estimator(c['presets'][0],c);x=pd.DataFrame(dict(x=np.arange(4)))
        for y,g in [([0,0,1,1],[2]),([0,0,1,1],[1,3]),([0,0,1,2],[2,2])]:
            with self.assertRaises(ValueError):fit_checked(model,x,y,g)
        self.assertIsNone(model.class_weight);self.assertNotIn('scale_pos_weight',model.get_params())

    def test_raw_scores_are_never_probability_transformed(self):
        class Fixture:
            def predict(self,x,raw_score=False):
                if not raw_score:raise AssertionError('Raw flag missing')
                return np.array([-100.,0.,100.])
        np.testing.assert_equal(raw_scores(Fixture(),None),[-100.,0.,100.])
        class Invalid:
            def predict(self,x,raw_score=False):return np.array([np.nan])
        with self.assertRaises(ValueError):raw_scores(Invalid(),None)

    def test_raw_policies_ties_and_monotone_ranking(self):
        d=dict(row_ids=np.array([4,1,3,2]),labels=np.array([0,1,0,1]),scores=np.array([-4.,2.,2.,1.]))
        transformed={**d,'scores':np.exp(d['scores']/5)}
        for kind in ['pooled','latest','rank_fraction']:
            rule=fit_policy(d,kind);other=fit_policy(transformed,kind)
            np.testing.assert_equal(decisions(d,rule),decisions(transformed,other))
            self.assertAlmostEqual(measure(d,rule)['f2'],10/11)
        with self.assertRaises(ValueError):learn({5:d},'pooled',5)
        self.assertEqual(learn({4:d},'pooled',5)['source_months'],[4])

    def test_real_saved_model_grouping_preprocessing_and_cache_audit(self):
        rng=np.random.default_rng(33);n=600
        frame=pd.DataFrame(dict(month=np.repeat([0,1,2],200),fraud_bool=np.tile([0,1,0,0],150)))
        base=pd.DataFrame(dict(x=frame.fraud_bool.to_numpy()+rng.normal(size=n)))
        for cat in CATEGORICAL:base[cat]=np.where(frame.month==2,'future',np.where(rng.random(n)<.5,'a','b'))
        with tempfile.TemporaryDirectory() as tmp:
            c=read_config('configs/direct-ranking.json');c['output_dir']=tmp;c['tree_params'].update(n_estimators=8,min_child_samples=5)
            spec=c['presets'][0];record=fit_model(c,{},frame,base,spec,2)
            self.assertEqual(record['train_rows'],400);self.assertEqual(record['train_month_max'],1)
            self.assertEqual(fit_model(c,{},frame,base,spec,2),record)
            mp,gp=model_paths(c,'validation',2,spec['id']);a=joblib.load(mp)['adapter'];self.assertNotIn('future',a.levels[CATEGORICAL[0]])
            meta=read_config(gp);self.assertEqual(meta['rows'],400);self.assertEqual(sum(meta['groups']),400)
            with self.assertRaisesRegex(ValueError,'identity'):audit_model(c,{'changed':True},frame,base,spec,2)
            gp.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'artifact changed'):audit_model(c,{},frame,base,spec,2)
            paths(c,'validation',2,spec['id'])[1].unlink()
            with self.assertRaisesRegex(ValueError,'Partial'):fit_model(c,{},frame,base,spec,2)

    def test_missing_references_route_to_new_destination_and_frozen_recipe(self):
        c=read_config('configs/direct-ranking.json')
        with tempfile.TemporaryDirectory() as tmp:
            c['output_dir']=str(Path(tmp)/'new')
            with patch('baf.focal_study.prepare_references',return_value=[{'reused':False}]) as fit:
                result=prepare_references(c,{},None,None)
                self.assertFalse(result[0]['reused']);adapted=fit.call_args.args[0]
                self.assertEqual(adapted['output_dir'],c['output_dir']);self.assertEqual(adapted['source_config'],c['reference_source_config'])

    def test_unequal_month_paired_macro_bootstrap_matches_manual_rows(self):
        h={name:{} for name in ['winner','control']};rows=[];rng=np.random.default_rng(17);rule=dict(kind='pooled',threshold=0.,beta=2)
        for month,n in [(5,7),(6,19)]:
            y=np.arange(n)%3==0
            for name in h:
                h[name][month]=dict(row_ids=np.arange(n)+month*100,labels=y.astype(int),scores=rng.normal(size=n))
                rows.append(dict(candidate=name,policy='pooled',eval_month=month,parameters=json.dumps(rule)))
        c=dict(bootstrap_seed=1027,bootstrap_replicates=50);s=dict(choices=dict(selected=['winner','pooled'],reference=['control','pooled']))
        table,draws=development(c,s,pd.DataFrame(rows),h);r=np.random.default_rng(1027);manual=[];points=[]
        for month,n in [(5,7),(6,19)]:
            values=[];data={alias:h[name][month] for alias,(name,_) in s['choices'].items()};mask={a:d['scores']>=0 for a,d in data.items()}
            points.append(measure(data['selected'],rule)['f2']-measure(data['reference'],rule)['f2'])
            for _ in range(50):
                idx=r.integers(0,n,n);f={}
                for a,d in data.items():
                    y=d['labels'][idx];p=mask[a][idx];tp=int((y*p).sum());fp=int(((1-y)*p).sum());fn=int(y.sum()-tp);den=5*tp+4*fn+fp
                    f[a]=5*tp/den if den else 0.
                values.append(f['selected']-f['reference'])
            manual.append(np.array(values));np.testing.assert_allclose(draws[draws.scope==str(month)].reference,values,atol=1e-14)
        macro=table[table.scope=='macro'].iloc[0];self.assertAlmostEqual(macro.difference,np.mean(points))
        np.testing.assert_allclose([macro.ci_low,macro.ci_high],np.quantile((manual[0]+manual[1])/2,[.025,.975]),atol=1e-14)


if __name__=='__main__':unittest.main()
