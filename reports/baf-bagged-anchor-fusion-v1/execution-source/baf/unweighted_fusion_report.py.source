"""Independent grid reconstruction and honest conditional fusion reporting."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import resource
import shutil
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .common import CATEGORICAL, digest, features, load_data, read_config, write_json
from .interaction_report import close, check_values, independent_pair_diagnostics
from .spline_fusion_report import independent_rule, independent_mask, independent_metrics
from .spline_report import direct_draw_audit, markdown
from .direct_ranking_report import development, intervals
from .score_fusion_report import audit_intervals
from . import unweighted_fusion_study as run




def independent_id(anchor,added,weight):
    if (anchor,added,weight)==('tree','balanced',.5):
        return 'logit_mean_xgb_d4_w1_baseline_balanced_a25'
    return f'lgbm_blend_{anchor}_{added}_l{int(weight*100)}'


def independent_declarations():
    anchors=dict(tree='logit_mean_xgb_d4_w1_baseline_balanced_a50',
        recent='margin_blend_spline_c0p1_unweighted_a25',
        latest='interaction_blend_spline_c1p0_balanced_a10',
        interaction='interaction_blend_interaction_c1p0_balanced_a10')
    members=dict(balanced='baseline_balanced',unweighted='baseline_lgbm')
    names=list(anchors.values())+list(members.values())+['logit_'+n for n in members.values()]
    names += [independent_id(a,b,w) for a,b,w in itertools.product(anchors,members,[.1,.25,.5,.75])]
    return anchors,members,names


def expected_factors():
    anchors,members,names=independent_declarations();policies=['pooled','latest','rank_fraction']
    rows=[]
    references=[(anchors['latest'],'latest'),(anchors['recent'],'latest'),(anchors['tree'],'pooled'),
        ('baseline_balanced','pooled'),('baseline_lgbm','pooled')]
    for name,k in itertools.product(names,policies):
        rows += [('fixed operating reference',name,k,r,rk) for r,rk in references]
        if k!='pooled':rows.append(('policy minus own pooled',name,k,name,'pooled'))
    for a,b,w,k in itertools.product(anchors,members,[.1,.25,.5,.75],policies):
        name=independent_id(a,b,w)
        rows += [(label,name,k,r,k) for label,r in [('addition minus anchor',anchors[a]),
            ('addition minus native',members[b]),('addition minus transformed','logit_'+members[b])]]
    for a,w,k in itertools.product(anchors,[.1,.25,.5,.75],policies):
        rows.append(('unweighted minus balanced',independent_id(a,'unweighted',w),k,
                     independent_id(a,'balanced',w),k))
    for prefix,k in itertools.product(['','logit_'],policies):
        rows.append(('unweighted minus balanced',prefix+'baseline_lgbm',k,prefix+'baseline_balanced',k))
    for b,w,k in itertools.product(members,[.1,.25,.5,.75],policies):
        rows.append(('interaction minus additive anchor',independent_id('interaction',b,w),k,
                     independent_id('latest',b,w),k))
    rows += [('interaction minus additive anchor',anchors['interaction'],k,anchors['latest'],k) for k in policies]
    for a,b,(lo,hi),k in itertools.product(anchors,members,list(itertools.combinations([.1,.25,.5,.75],2)),policies):
        rows.append(('larger lambda minus smaller lambda',independent_id(a,b,hi),k,independent_id(a,b,lo),k))
    rows += [('transformed minus native','logit_'+n,k,n,k) for n,k in itertools.product(members.values(),policies)]
    if len(rows)!=1199 or len(set(rows))!=1199:raise ValueError('Independent factor union differs')
    return rows


def check_keys(table, cols, keys):
    actual=list(table[cols].itertuples(index=False,name=None))
    if len(actual)!=len(keys) or set(actual)!=set(keys):raise ValueError('Complete output key set differs: '+str(cols))

def independent_logit(p):
    p=np.asarray(p)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Independent probability domain invalid')
    p=np.maximum(1e-6,np.minimum(1-1e-6,p))
    return np.log(p)-np.log(1-p)

def independent_leaf_coefficients(name):
    anchors,members,_=independent_declarations()
    leaves={
        anchors['tree']:{'xgb_d4_w1':.5,'baseline_balanced':.5},
        anchors['recent']:{'xgb_d4_w1':.375,'baseline_balanced':.375,'spline_c0p1_unweighted':.25},
        anchors['latest']:{'xgb_d4_w1':.45,'baseline_balanced':.45,'spline_c1p0_balanced':.1},
        anchors['interaction']:{'xgb_d4_w1':.45,'baseline_balanced':.45,'interaction_c1p0_balanced':.1},
        'logit_mean_xgb_d4_w1_baseline_balanced_a25':{'xgb_d4_w1':.25,'baseline_balanced':.75}}
    if name in leaves:return leaves[name]
    for member in members.values():
        if name in [member,'logit_'+member]:return {member:1.}
    for a,b,w in itertools.product(anchors,members,[.1,.25,.5,.75]):
        if independent_id(a,b,w)==name:
            result={n:(1-w)*v for n,v in leaves[anchors[a]].items()}
            result[members[b]]=result.get(members[b],0.)+w
            return result
    raise ValueError('Independent leaf identity unknown')


def audit_compositions(c,fp,months,phase,names):
    dag=run.dependencies(names)
    anchors,members,_=independent_declarations()
    for month in months:
        for name in dag['derived']:
            spec=next(s for s in c['strategies'] if s['id']==name)
            actual=run.prediction(c,phase,month,name)
            ds=[run.prediction(c,phase,month,n) for n in spec['members']]
            for d in ds:run.aligned(d,actual)
            u=independent_logit(ds[0]['scores'])
            if name=='logit_mean_xgb_d4_w1_baseline_balanced_a25':
                v=independent_logit(ds[1]['scores']);expected=.25*u+.75*v
                nested=.5*v+.5*(.5*u+.5*v);close(expected,nested,inference=True)
            elif name==anchors['tree']:
                expected=.5*u+.5*independent_logit(ds[1]['scores'])
            elif spec['kind']=='anchor':
                weight=.25 if name==anchors['recent'] else .1
                expected=weight*u+(1-weight)*ds[1]['scores']
            elif spec['kind']=='logit':expected=u
            else:expected=spec['weight']*u+(1-spec['weight'])*ds[1]['scores']
            close(actual['scores'],expected,inference=True)
            rp,pp=run.paths(c,phase,month,name);r=read_config(rp)
            parents={n:{str(p):digest(p) for p in run.paths(c,phase,month,n)} for n in spec['members']}
            if (r['dependencies']!=parents or r['fingerprint']!=fp or r['spec']!=spec
                    or r['epsilon']!=1e-6 or r['prediction_sha256']!=digest(pp)
                    or r['leaf_coefficients']!=independent_leaf_coefficients(name)
                    or r['domains']!=['probability-like' if n in run.BASES else 'finite margin' for n in spec['members']]
                    or r['arithmetic']!=spec.get('arithmetic','fixed multiply then add')):
                raise ValueError('Independent composition provenance differs')
            run.verify_hashes(r['origin_artifacts'])
            if bool(r['origin_artifacts'])!=r['reused']:raise ValueError('Derived reuse status differs')
            for source,identity in run.derived_origins(c,spec):
                sr,sp=run.paths(source,phase,month,name)
                if not run.tree.cache_state([sr,sp]):continue
                origin=read_config(sr);saved=run.prediction(source,phase,month,name)
                original_parents={}
                for member in spec['members']:
                    original_record,original_score=run.paths(source,phase,month,member)
                    original_parents[member]=(
                        {'record':digest(original_record),'prediction':digest(original_score)}
                        if name in [anchors['tree'],'logit_mean_xgb_d4_w1_baseline_balanced_a25']
                        else {str(original_record):digest(original_record),str(original_score):digest(original_score)})
                if (origin['fingerprint']!=identity or origin['prediction_sha256']!=digest(sp)
                        or origin['dependencies']!=original_parents
                        or r['origin_artifacts'].get(str(sr))!=digest(sr)
                        or r['origin_artifacts'].get(str(sp))!=digest(sp)):
                    raise ValueError('Derived original provenance differs')
                run.aligned(saved,actual)
                if not np.array_equal(saved['scores'],actual['scores']):
                    raise ValueError('Historical direct arithmetic changed')
                break


def audit_tables(c, frame, base, fp=None):
    out=Path(c['output_dir']);h=run.history(c);fp=run.fingerprint() if fp is None else fp
    expected_names=set(independent_declarations()[2])
    cf=pd.read_csv(out/'crossfit.csv')
    check_keys(cf,['candidate','policy','eval_month'],{(n,k,m) for n in h for k in c['policies'] for m in [5,6]})
    for r in cf.to_dict('records'):
        n,k,m=r['candidate'],r['policy'],r['eval_month'];rule=independent_rule(h[n],k,m)
        if rule!=json.loads(r['parameters']) or str(r['source_months'])!=','.join(map(str,rule['source_months'])):
            raise ValueError('Independent earlier-only rule differs')
        check_values(r,independent_metrics(h[n][m],independent_mask(h[n][m],rule)))
    summary=pd.read_csv(out/'summary.csv');index=summary.set_index(['candidate','policy'])
    check_keys(summary,['candidate','policy'],{(n,k) for n in h for k in c['policies']})
    for (n,k),g in cf.groupby(['candidate','policy']):
        check_values(index.loc[(n,k)],{v:float(g[v].mean()) for v in
            ['f2','f1','f3','precision','recall','flagged_fraction','average_precision','roc_auc'] if v in index.columns})
    ordered=summary.sort_values(['f2','recall','precision','candidate','policy'],ascending=[False,False,False,True,True])
    if list(ordered.index)!=list(range(len(summary))):raise ValueError('Actual macro winner/order differs')
    contrasts=pd.read_csv(out/'contrasts.csv')
    check_keys(contrasts,['contrast','candidate','policy','reference','reference_policy'],expected_factors())
    for r in contrasts.to_dict('records'):
        a,b=index.loc[(r['candidate'],r['policy'])],index.loc[(r['reference'],r['reference_policy'])]
        check_values(r,{v+'_difference':a[v]-b[v] for v in ['f2','recall','precision','flagged_fraction','average_precision']})
    pooled=pd.read_csv(out/'pooled.csv');check_keys(pooled,['candidate','policy'],set(index.index))
    for r in pooled.to_dict('records'):
        g=cf[(cf.candidate==r['candidate'])&(cf.policy==r['policy'])];tn,fp_,fn,tp=[int(g[v].sum()) for v in ['tn','fp','fn','tp']]
        metrics=dict(tn=tn,fp=fp_,fn=fn,tp=tp,precision=tp/(tp+fp_) if tp+fp_ else 0.,
            recall=tp/(tp+fn) if tp+fn else 0.,flagged_fraction=(tp+fp_)/g.n.sum())
        for beta in [1,2]:
            den=(1+beta**2)*tp+beta**2*fn+fp_;metrics['f'+str(beta)]=(1+beta**2)*tp/den if den else 0.
        check_values(r,metrics)
    scores=pd.read_csv(out/'score_diagnostics.csv')
    check_keys(scores,['candidate','eval_month'],{(n,m) for n in h for m in [4,5,6]})
    for r in scores.to_dict('records'):
        p=h[r['candidate']][r['eval_month']]['scores']
        check_values(r,dict(n=len(p),score_min=p.min(),score_max=p.max(),score_mean=p.mean(),
            unique_scores=len(np.unique(p)),tied_fraction=1-len(np.unique(p))/len(p)))
    clips=pd.read_csv(out/'clipping.csv')
    check_keys(clips,['candidate','eval_month'],{(n,m) for n in run.BASES for m in [4,5,6]})
    for r in clips.to_dict('records'):
        p=run.prediction(c,'validation',r['eval_month'],r['candidate'])['scores'];independent_logit(p)
        check_values(r,dict(n=len(p),epsilon=1e-6,input_min=p.min(),input_max=p.max(),
            below_epsilon=int((p<1e-6).sum()),above_one_minus_epsilon=int((p>1-1e-6).sum()),
            changed_count=int((p!=np.maximum(1e-6,np.minimum(1-1e-6,p))).sum())))
    div=pd.read_csv(out/'diversity.csv')
    check_keys(div,['anchor','added','policy','eval_month'],
        {(a,b,k,m) for a in run.ANCHORS for b in run.ADDED for k in c['policies'] for m in [5,6]})
    for r in div.to_dict('records'):
        n,k,m=r['candidate'],r['policy'],r['eval_month'];anchor=run.ANCHORS[r['anchor']]
        if n!=run.ADDED[r['added']]:raise ValueError('Diversity member identity differs')
        a,b=h['logit_'+n][m],h[anchor][m]
        pa=independent_mask(a,independent_rule(h['logit_'+n],k,m))
        pb=independent_mask(b,independent_rule(h[anchor],k,m))
        constant=np.ptp(a['scores'])==0 or np.ptp(b['scores'])==0
        if r['correlation_reason']!=('constant member' if constant else 'defined'):raise ValueError('Correlation reason differs')
        if constant:
            if not pd.isna(r['pearson']) or not pd.isna(r['spearman']):raise ValueError('Undefined correlation fabricated')
        else:check_values(r,dict(pearson=np.corrcoef(a['scores'],b['scores'])[0,1],
            spearman=np.corrcoef(rankdata(a['scores']),rankdata(b['scores']))[0,1]))
        masks={'both':pa&pb,'member_only':pa&~pb,'tree_only':~pa&pb,'neither':~pa&~pb}
        vals={'both_flagged':int(masks['both'].sum()),'member_only':int(masks['member_only'].sum()),
              'tree_only':int(masks['tree_only'].sum()),'neither_flagged':int(masks['neither'].sum())}
        vals.update({label+'_'+key:int(((a['labels']==y)&mask).sum()) for y,label in [(1,'fraud'),(0,'legitimate')] for key,mask in masks.items()})
        check_values(r,vals)
    audit_compositions(c,fp,[4,5,6],'validation',h)
    wanted={f'validation-m{m}-{n}' for n in expected_names|set(run.BASES) for m in [4,5,6]}
    for folder,suffix in [('runs','.json'),('predictions','.npz')]:
        if {p.stem for p in (out/folder).glob('validation-*'+suffix)}!=wanted:raise ValueError('Physical artifact set differs')
    audit_feature_exports(c,frame,base)
    return dict(scores=132,selectable_scores=120,operating_rows=240,macro_rows=120,contrasts=1199,
        clipping_rows=18,diversity_rows=48,feature_rows=522,pair_rows=18,
        coefficient_rows=len(pd.read_csv(out/'coefficients.csv')),winner=str(summary.iloc[0].candidate),
        policy=str(summary.iloc[0].policy),development_f2=float(summary.iloc[0].f2))

def audit_feature_exports(c,frame,base):
    """Compatible feature-only reconstruction; no predecessor grid assumptions."""
    out=Path(c['output_dir'])
    main=read_config(out/'feature_blocks.json');pairs=read_config(out/'interaction_blocks.json')
    wanted_main={f'{m}:{s["id"]}' for m in [4,5,6] for s in c['candidates']}
    wanted_pairs={f'{m}:{s["id"]}' for m in [4,5,6] for s in c['candidates'] if s['representation']=='interaction'}
    if set(main)!=wanted_main or set(pairs)!=wanted_pairs:raise ValueError('Feature metadata key sets differ')
    fd=pd.read_csv(out/'feature_diagnostics.csv');pdg=pd.read_csv(out/'pair_diagnostics.csv')
    coefficients=pd.read_csv(out/'coefficients.csv');solver=pd.read_csv(out/'solver.csv');numbers=pd.read_csv(out/'numerical.csv')
    resources=pd.read_csv(out/'resources.csv')
    expected_fd=set();expected_pd=set();expected_coef=set()
    for m,spec in itertools.product([4,5,6],c['candidates']):
        n=spec['id'];a=joblib.load(run.model_path(c,'validation',m,n));inp,model=a['inputs'],a['model']
        r=read_config(run.paths(c,'validation',m,n)[0]);key=f'{m}:{n}';interaction=spec['representation']=='interaction'
        if main[key]!=dict(reused=bool(r.get('reused')),metadata=inp.additive.metadata() if interaction else inp.metadata()):
            raise ValueError('Main metadata differs')
        if interaction and pairs[key]!=inp.metadata():raise ValueError('Interaction metadata differs')
        for t in [solver,numbers,resources]:
            if len(t[(t.candidate==n)&(t.eval_month==m)])!=1:raise ValueError('Numerical/resource key set differs')
        sr=solver[(solver.candidate==n)&(solver.eval_month==m)].iloc[0]
        check_values(sr,{k:r[k] for k in ['iterations','transformed_dimension','matrix_nnz','matrix_density','fit_seconds','peak_process_rss_bytes']})
        if json.loads(sr.solver_params)!=r['solver_params'] or json.loads(sr.warnings)!=r['warnings']:
            raise ValueError('Solver metadata differs')
        nr=numbers[(numbers.candidate==n)&(numbers.eval_month==m)].iloc[0]
        for k,v in r['numerical'].items():
            if k in ['gradient','warnings']:
                if json.loads(nr[k])!=v:raise ValueError('Numerical vector differs')
            else:check_values(nr,{k:v})
        for population,mask in [('training',frame.month<m),('evaluation',frame.month==m)]:
            x=base.loc[frame.index[mask]];add=inp.additive if interaction else inp
            cats=add.categories(x)
            for col in add.columns:
                expected_fd.add((n,m,population,col))
                row=fd[(fd.candidate==n)&(fd.eval_month==m)&(fd.population==population)&(fd.feature==col)]
                if len(row)!=1:raise ValueError('Feature diagnostic identity differs')
                if col in CATEGORICAL:
                    levels=add.encoder.categories_[CATEGORICAL.index(col)]
                    values=dict(n=len(x),missing=int(x[col].isna().sum()),below_training=0,above_training=0,
                        unseen_category=int((~cats[col].isin(levels)).sum()),all_missing_training=False)
                else:
                    block=next(b for b in add.blocks if b['feature']==col);v=x[col].dropna()
                    values=dict(n=len(x),missing=int(x[col].isna().sum()),
                        below_training=int((v<block['minimum']).sum()) if block['minimum'] is not None else 0,
                        above_training=int((v>block['maximum']).sum()) if block['maximum'] is not None else 0,
                        unseen_category=0,all_missing_training=block['minimum'] is None)
                check_values(row.iloc[0],values)
            if interaction:
                for b in inp.pair_blocks:
                    expected_pd.add((n,m,population,b['left'],b['right']))
                    row=pdg[(pdg.candidate==n)&(pdg.eval_month==m)&(pdg.population==population)&
                            (pdg.left==b['left'])&(pdg.right==b['right'])]
                    if len(row)!=1:raise ValueError('Pair diagnostic identity differs')
                    check_values(row.iloc[0],independent_pair_diagnostics(inp,x,b))
        for name,value in list(zip(inp.names,model.coef_[0]))+[('__intercept__',model.intercept_[0])]:
            expected_coef.add((n,m,name))
            row=coefficients[(coefficients.candidate==n)&(coefficients.eval_month==m)&(coefficients.feature==name)]
            if len(row)!=1:raise ValueError('Coefficient key differs')
            close(row.coefficient.to_numpy(),[value],inference=True)
        rr=resources[(resources.candidate==n)&(resources.eval_month==m)].iloc[0]
        check_values(rr,dict(rows=r['train_rows'],dimension=r['transformed_dimension'],nnz=r['matrix_nnz'],
            estimated_hessian_bytes=8*(r['transformed_dimension']+1)**2,fit_seconds=r['fit_seconds'],
            peak_process_rss_bytes=r['peak_process_rss_bytes']))
        if interaction:check_values(rr,r['resources'])
    for table,cols,expected_keys,count in [(fd,['candidate','eval_month','population','feature'],expected_fd,522),
         (pdg,['candidate','eval_month','population','left','right'],expected_pd,18),
         (coefficients,['candidate','eval_month','feature'],expected_coef,len(expected_coef))]:
        if len(table)!=count or set(table[cols].itertuples(index=False,name=None))!=expected_keys:
            raise ValueError('Complete diagnostic/coefficient keys differ')
    if any(len(t)!=9 for t in [solver,numbers,resources]):raise ValueError('Numerical/resource count differs')

def audit_development(config_path=run.CONFIG, *, write=True):
    c,fp=run.initialize(config_path);out=Path(c['output_dir'])
    if write and (out/'selection.json').exists():raise ValueError('Frozen independent audit cannot be overwritten')
    done=read_config(out/'validation_complete.json')
    if done['fingerprint']!=fp:raise ValueError('Development identity differs')
    run.verify_hashes({str(out/p):h for p,h in done['files'].items()})
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    records=run.prepare(c,fp,frame,base,allow_missing=False)
    preparation=read_config(out/'preparation.json')
    if (len(records)!=18 or preparation['base_reuses']!=sum(bool(r.get('reused')) for r in records)
            or preparation['new_fits']!=sum(not r.get('reused',False) for r in records)):
        raise ValueError('Actual development fit/reuse evidence differs')
    result=audit_tables(c,frame,base,fp)
    audit_reference_values(pd.read_csv(out/'summary.csv').set_index(['candidate','policy']))
    run.guard(c)
    if write:
        write_json(out/'independent_development_audit.json',dict(fingerprint=fp,
            validation_sha256=digest(out/'validation_complete.json'),saved_model_checks=len(records),
            independent_statistics_inference_and_gradient=True,independent_policies_metrics_factors=True,
            elapsed_seconds=time.perf_counter()-start,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            created_at_utc=run.now(),**result))
    return records

def audit_reference_values(index):
    values=[('interaction_blend_spline_c1p0_balanced_a10','latest',.3307932536724773),
        ('margin_blend_spline_c0p1_unweighted_a25','latest',.3290099447918403),
        ('logit_mean_xgb_d4_w1_baseline_balanced_a50','pooled',.3289065701775199),
        ('baseline_balanced','pooled',.32780157809618937),('baseline_lgbm','pooled',.3226536849212962)]
    for n,k,value in values:
        close([index.loc[(n,k),'f2']],[value],inference=True)


def operating_figure(c,summary,dest):
    index=summary.set_index(['candidate','policy']);fig,axes=plt.subplots(2,3,figsize=(18,10))
    for kind,label in [('native','Native LightGBM'),('logit','Logit LightGBM'),('anchor','Preserved anchor'),('addition','Fixed addition')]:
        ids=[s['id'] for s in c['strategies'] if s['kind']==kind];t=summary[summary.candidate.isin(ids)]
        axes[0,0].scatter(t.recall,t.precision,label=label,s=22,alpha=.7)
    axes[0,0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 120 strategy/policy results')
    axes[0,0].legend(fontsize=8)
    for tag,ax in zip(['tree','recent','latest','interaction'],[axes[0,1],axes[0,2],axes[1,0],axes[1,1]]):
        for added in ['balanced','unweighted']:
            names=[run.ANCHORS[tag]]+[independent_id(tag,added,w) for w in [.1,.25,.5,.75]]+['logit_'+run.ADDED[added]]
            vals=[float(index.loc[(n,'pooled'),'f2']) for n in names]
            ax.plot([0,.1,.25,.5,.75,1],vals,marker='o',label=added)
        ax.set(xlabel='Added LightGBM weight (lambda)',ylabel='Actual macro F2',title=tag+' anchor / pooled policy')
        ax.legend(fontsize=8)
    for tag,added in itertools.product(['latest','interaction'],['balanced','unweighted']):
        vals=[float(index.loc[(independent_id(tag,added,w),'pooled'),'f2']) for w in [.1,.25,.5,.75]]
        axes[1,2].plot([.1,.25,.5,.75],vals,marker='o',label=tag+' / '+added)
    axes[1,2].set(xlabel='Added LightGBM weight (lambda)',ylabel='Actual macro F2',title='Matched additive / interaction anchors')
    axes[1,2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'fusion_operating.png',dpi=160);plt.close(fig)


def report(config_path=run.CONFIG):
    c,s=run.load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'report_hashes.json').exists():raise ValueError('Frozen report cannot be overwritten')
    records=audit_development(config_path,write=False)
    frame=load_data(c);base=features(frame);names={n for n,k in s['choices'].values()}
    final_records=run.prepare(c,s['fingerprint'],frame,base,'exploratory',names,False)
    audit_compositions(c,s['fingerprint'],[7],'exploratory',names)
    access=read_config(out/'exploratory_access.json');done=read_config(out/'transfer_complete.json');sha=digest(out/'selection.json')
    if (access['selection_sha256']!=sha or done['selection_sha256']!=sha
            or not s['created_at_utc']<access['access_at_utc'] or done['table_sha256']!=digest(out/'exploratory.csv')):
        raise ValueError('Transfer access/table evidence differs')
    data=run.final_data(c,s);final=pd.read_csv(out/'exploratory.csv');masks={};h=run.history(c)
    check_keys(final,['role'],{(a,) for a in s['choices']})
    for a,d in data.items():
        run.aligned(d,data['selected']);n,k=s['choices'][a]
        if s['rules'][a]!=independent_rule(h[n],k,7):raise ValueError('Final earlier-only rule differs')
        masks[a]=independent_mask(d,s['rules'][a]);r=final[final.role==a].iloc[0]
        if r.candidate!=n or r.policy!=k or json.loads(r.parameters)!=s['rules'][a]:raise ValueError('Transfer role/rule changed')
        check_values(r,independent_metrics(d,masks[a]))
    if done['base_results']!=len(final_records) or done['new_fits']!=sum(not r.get('reused',False) for r in final_records):
        raise ValueError('Transfer fit counts differ')
    cf=pd.read_csv(out/'crossfit.csv');summary=pd.read_csv(out/'summary.csv')
    dint,ddraws=development(c,s,cf,h);fint,fdraws=intervals(data,masks,np.random.default_rng(c['bootstrap_seed']),500,'7_consumed')
    audit_intervals(dint,ddraws,fint,fdraws,s['choices'],500)
    rng=np.random.default_rng(c['bootstrap_seed'])
    for m in [5,6]:
        d={a:h[n][m] for a,(n,k) in s['choices'].items()}
        p={a:independent_mask(d[a],independent_rule(h[n],k,m)) for a,(n,k) in s['choices'].items()}
        direct_draw_audit(d,p,rng,ddraws[ddraws.scope==str(m)].reset_index(drop=True))
    direct_draw_audit(data,masks,np.random.default_rng(c['bootstrap_seed']),fdraws)
    for n,t in [('development_intervals',dint),('development_draws',ddraws),('consumed_intervals',fint),
                ('consumed_draws',fdraws.assign(replicate=np.arange(len(fdraws))))]:t.to_csv(dest/(n+'.csv'),index=False)
    for n in run.TABLES+['validation_complete.json','independent_development_audit.json','selection.json',
                        'exploratory_access.json','transfer_complete.json','exploratory.csv']:
        shutil.copyfile(out/n,dest/n)
    shutil.copyfile(config_path,dest/'config.json');shutil.copyfile(c['protocol_path'],dest/'protocol.md')
    write_json(dest/'run_provenance.json',records);write_json(dest/'transfer_provenance.json',final_records)
    winner=summary.iloc[0];index=summary.set_index(['candidate','policy'])
    reference_roles={a:tuple(nk) for a,nk in run.REFERENCE_ROLES.items()}
    gain={a:float(winner.f2-index.loc[nk,'f2']) for a,nk in reference_roles.items()}
    fg={r.reference:float(r.difference) for r in fint.itertuples()}
    measured=all(gain[a]>0 and fg[a]>0 for a in reference_roles)
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,development_gain=gain,consumed_gain=fg,
        measured_gain_on_both_scopes=measured,direct_draws_checked=1500,saved_development_models_checked=len(records),
        conditional_clear_development_gain=all(float(dint[(dint.scope=='macro')&(dint.reference==a)].iloc[0].ci_low)>0 for a in gain),
        conditional_clear_consumed_gain=all(float(fint[fint.reference==a].iloc[0].ci_low)>0 for a in gain),preservation_checked=True)
    write_json(dest/'audit.json',audit);operating_figure(c,summary,dest)
    comp=pd.read_csv(out/'contrasts.csv');counts=read_config(out/'preparation.json');receipt=read_config(out/'checks.json')
    text=f'''# Unweighted-reference fusion findings

Selected **{winner.candidate} / {winner.policy}**, macro development F2 **{winner.f2:.6f}**. Positive measured differences against all five fixed references on both development and consumed transfer: **{measured}**. Finite-search completion does not itself establish the performance goal.

## Design and actual work

Four preserved margins receive balanced or unweighted LightGBM at fixed lambdas 0.10/0.25/0.50/0.75. Four anchors, two native and two logit controls plus 32 additions form 40 canonical strategies. The half-balanced addition to the tree margin retains its historical quarter-XGBoost/three-quarter-balanced identity and direct floating-point association; it is not a renamed gain. Only probability-like scores receive logit clipping (epsilon 1e-6); anchors are finite margins. Neither scores nor analyst-selected weights are asserted calibrated probabilities or bank costs.

Development used {counts['base_reuses']} independently checked base reuses and {counts['new_fits']} new full fits. Its 114 derived results include {counts['derived_reuses']} checked historical reuses and {counts['new_derived_computations']} new arithmetic calculations. There are 132 physical scores, 120 selectable scores, 240 forward rows, 120 macro/pooled/monthly-score summaries and 1199 contrasts. Four new tiny cache-test fits per full invocation are separate from historical test-module fits; the recorded suite passed {receipt['tests_passed']} tests. The preserved eight-fit miniature gate required no new production miniature fits. Predictor-only preflight covers 903157 eligible rows, bounded 8192-row construction and the unchanged 70%-of-host-RAM gate.

Models/preprocessing use earlier training months. Month-5 policies learn month 4; month-6 policies learn months 4/5 or latest month 5. Actual macro F2 beta=2 determines selection, then recall/precision/deterministic IDs, without a quota or precision floor. AP, pooled F2, correlations and joint errors are descriptive.

## Complete chronological results

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

## Matched effects and all five references

All fixed-reference, own-policy, anchor/native/logit member, added-weighting, matched latest/interaction-anchor and addition-weight contrasts retain negative effects and ties. Recent versus latest anchors change C, weighting and spline fraction together, so their contrast does not isolate one factor. Equal C across representations does not mean equal complexity. Clipping retains 18 rows and diversity 48, including explicit null reasons for undefined correlations. Nine logistic numerical/resource/main-feature blocks, three interaction blocks, 522 original-feature rows, 18 pair rows and all actual saved coefficient/intercept names were verified. Historical fit resources are distinguished from current verification/composition resources.

{markdown(comp[comp.candidate==winner.candidate][['contrast','policy','reference','reference_policy','f2_difference']])}

## Conditional development differences

{markdown(dint)}

## Already-consumed month 7

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

Only {len(s['choices'])} frozen roles transferred with {done['base_reuses']} base reuses and {done['new_fits']} new fits. Exact DAG, original cache hashes and tighter per-type fit bounds were frozen before further access. No losing-grid evaluation, hindsight cutoff or replacement of the development winner occurred.

{markdown(fint)}

## Limitations and reproduction

Every one of 1500 paired fixed-decision draws and its quantiles was independently reconstructed, with seed 1027 and 500 draws per month. Intervals condition on fixed models/policies and exclude adaptive source/anchor/lambda choices, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity adjustment. Clear-gain language additionally requires a positive conditional lower bound on the named scope. Month 7 was already consumed and provides no independent confirmation. The synthetic data and eight monthly periods limit generalization.

Fourteen completed studies, separate Newton fit/analysis identities and the stopped attempt remain preserved. README provides pinned environment/data identity, fourteen exact selection restores and ordered reproduction. Existing frozen audit/selection/report evidence must not be overwritten. Settings remain analyst-selected and open to group review. Codex assisted with implementation, execution, independent audits, diagnostics and reporting, using preserved repository interfaces and pinned scikit-learn/SciPy/XGBoost/LightGBM APIs. No public notebook was copied; the group must acknowledge assistance and understand submitted work.

![All operating results and matched additions](fusion_operating.png)
'''
    (dest/'findings.md').write_text(text)
    run.guard(c)
    write_json(dest/'report_hashes.json',{str(p):digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='report_hashes.json'})
    print('Report complete; measured goal acceptance:',measured,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit_development','report'])
    parser.add_argument('--config',default=run.CONFIG);args=parser.parse_args();globals()[args.command](args.config)
