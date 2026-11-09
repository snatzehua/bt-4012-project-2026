"""Independent declarations and score checks for the bagged-anchor study."""
from itertools import product
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import json
import argparse
import resource
import shutil
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import read_config, digest, write_json, load_data, features
from .direct_ranking_report import development, intervals
from .score_fusion_report import audit_intervals
from .spline_report import direct_draw_audit, markdown
from .recall_weighted_metrics import aligned
from .interaction_report import check_values, close
from .spline_fusion_report import independent_rule, independent_mask, independent_metrics
from . import bagged_anchor_study as run

H = 'interaction_blend_spline_c1p0_balanced_a10'
T = 'logit_mean_xgb_d4_w1_baseline_balanced_a50'
R = 'margin_blend_spline_c0p1_unweighted_a25'
REFS = {'latest_fusion':[H,'latest'], 'recent_fusion':[R,'latest'], 'fusion':[T,'pooled'],
        'balanced':['baseline_balanced','pooled'], 'unweighted':['baseline_lgbm','pooled']}
WEIGHTS = ['balanced','unweighted']
POLICIES = ['pooled','latest','rank_fraction']
BASELINE = {'balanced':'baseline_balanced','unweighted':'baseline_lgbm'}
SEEDS = [f'bagged_{w}_s{s}' for w,s in product(WEIGHTS,[17,29,43])]
MEANS = [f'bagged_{w}_{f}_mean' for w,f in product(WEIGHTS,['prob','logit'])]
LOGITS = ['logit_'+p for p in SEEDS+[f'bagged_{w}_prob_mean' for w in WEIGHTS]+list(BASELINE.values())]
BLENDS = [f'bagged_anchor_{w}_{f}_a{a}' for w,f,a in product(WEIGHTS,['prob','logit'],['10','25','50'])]
NAMES = [v[0] for v in REFS.values()]+SEEDS+MEANS+LOGITS+BLENDS
OLD = {H:['spline_c1p0_balanced',T], R:['spline_c0p1_unweighted',T], T:['xgb_d4_w1','baseline_balanced']}
OLD_BASES = {'spline_c1p0_balanced','spline_c0p1_unweighted','xgb_d4_w1','baseline_balanced','baseline_lgbm'}


def independent_spec(n):
    if n in OLD:
        return dict(id=n,kind='anchor',anchor={H:'latest',R:'recent',T:'tree'}[n],
                    members=OLD[n],alpha={H:.1,R:.25,T:.5}[n])
    if n in MEANS:
        w='balanced' if n.startswith('bagged_balanced_') else 'unweighted'
        return dict(id=n,kind='aggregate',weighting=w,form='prob' if '_prob_' in n else 'logit',
                    members=[f'bagged_{w}_s{s}' for s in [17,29,43]])
    if n in LOGITS:
        return dict(id=n,kind='logit',members=[n[6:]])
    m=re.fullmatch(r'bagged_anchor_(balanced|unweighted)_(prob|logit)_a(10|25|50)',n)
    if m is None:
        raise ValueError('Undeclared derived score')
    w,f,a=m.groups()
    g=f'logit_bagged_{w}_prob_mean' if f=='prob' else f'bagged_{w}_logit_mean'
    return dict(id=n,kind='blend',weighting=w,form=f,alpha=int(a)/100,members=[H,g])


def graph(names):
    found,bases=set(),set()
    def visit(n):
        if n in found:
            return
        found.add(n)
        if n in OLD_BASES or n in SEEDS:
            bases.add(n)
        else:
            for member in independent_spec(n)['members']:
                visit(member)
    for n in names:
        visit(n)
    return dict(bases=sorted(bases),scores=sorted(found),derived=sorted(found-bases),
                bagged_bases=sorted(bases&set(SEEDS)))


def check_keys(table,columns,keys):
    actual=list(table[columns].itertuples(index=False,name=None))
    if len(actual)!=len(set(actual)) or set(actual)!=set(keys):
        raise ValueError('Exact output key set differs: '+','.join(columns))


def logit(values):
    p=np.asarray(values,dtype=np.float64)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probability domain')
    q=np.maximum(1e-6,np.minimum(1-1e-6,p))
    return np.log(q)-np.log1p(-q)


def expected_values(spec,items):
    arrays=[np.asarray(items[n]['scores'],dtype=np.float64) for n in spec['members']]
    for n in spec['members']:
        aligned(items[spec['members'][0]],items[n])
    if spec['kind']=='aggregate':
        if spec['form']=='logit':
            arrays=[logit(p) for p in arrays]
        else:
            for p in arrays:
                logit(p)  # Check probability domain without using transformed values.
        return (arrays[0]+arrays[1]+arrays[2])/3.
    if spec['kind']=='logit':
        return logit(arrays[0])
    if spec['kind']=='anchor':
        a=spec['alpha']
        return a*logit(arrays[0])+(1-a)*(logit(arrays[1]) if spec['id']==T else arrays[1])
    a=spec['alpha']
    return (1-a)*arrays[0]+a*arrays[1]


def independent_expressions():
    rows=[]
    for n in BLENDS:
        s=independent_spec(n);a=s['alpha'];w=s['weighting']
        atoms={'logit(spline_c1p0_balanced)':(1-a)*.1,
               'logit(xgb_d4_w1)':(1-a)*.9/2,
               'logit(baseline_balanced)':(1-a)*.9/2}
        if s['form']=='prob':
            atoms[f'logit(bagged_{w}_prob_mean)']=a
        else:
            atoms.update({f'logit(bagged_{w}_s{seed})':a/3 for seed in [17,29,43]})
        for atom,coefficient in atoms.items():
            rows.append(dict(candidate=n,atom=atom,coefficient=coefficient,
                atom_kind='nonlinear probability mean' if 'prob_mean)' in atom else 'source logit',
                evaluation_arithmetic='(1-alpha)*anchor + alpha*margin'))
    return rows


def audit_compositions(c,fp,months,phase,names):
    dag=graph(names)
    if dag!=run.scores.dependencies(names):
        raise ValueError('Exact source graph differs')
    expressions=independent_expressions()
    for m in months:
        data={n:run.prediction(c,phase,m,n) for n in dag['scores']}
        for n in dag['derived']:
            s=independent_spec(n);rp,pp=run.paths(c,phase,m,n);r=read_config(rp);d=data[n]
            arithmetic=('preserved reference' if s['kind']=='anchor' else 'ordered float64 sum / 3'
                if s['kind']=='aggregate' else 'log(p)-log1p(-p) after clip' if s['kind']=='logit'
                else '(1-alpha)*anchor + alpha*margin')
            identity=dict(candidate=n,phase=phase,eval_month=m,fingerprint=fp,spec=s,
                epsilon=1e-6,arithmetic=arithmetic,prediction_sha256=digest(pp),
                dependencies={p:{str(path):digest(path) for path in run.paths(c,phase,m,p)} for p in s['members']})
            if any(r.get(k)!=v for k,v in identity.items()):
                raise ValueError('Derived provenance identity differs')
            run.verify_hashes(r['origin_artifacts'])
            for p in s['members']:
                aligned(d,data[p])
            expected=expected_values(s,data)
            if not np.array_equal(expected,d['scores']):
                raise ValueError('Declared score association differs')
            if s['kind']=='blend':
                terms=[row['coefficient']*logit(data[row['atom'][6:-1]]['scores'])
                       for row in expressions if row['candidate']==n]
                if not np.allclose(d['scores'],sum(terms),atol=1e-12,rtol=0):
                    raise ValueError('Diagnostic expression differs from associated evaluation')


def diagnostic_pairs():
    pairs=[]
    for w,f in product(WEIGHTS,['prob','logit']):
        pairs.append(('anchor versus '+w+' '+f,H,f'logit_bagged_{w}_prob_mean' if f=='prob' else f'bagged_{w}_logit_mean'))
    for w in WEIGHTS:
        pairs.append(('aggregation forms '+w,f'logit_bagged_{w}_prob_mean',f'bagged_{w}_logit_mean'))
    return pairs


def expected_factors():
    facts=set()
    expected=set(NAMES);policies=POLICIES;weights=WEIGHTS;refs=REFS
    blends=BLENDS;logits=LOGITS;baseline=BASELINE
    def factor(label,n,k,p,pk=None):facts.add((label,n,k,p,k if pk is None else pk))
    for n,k in product(expected,policies):
     for label,(p,pk) in refs.items():factor('fixed reference '+label,n,k,p,pk)
     if k!='pooled':factor('policy minus own pooled',n,k,n,'pooled')
    for n,k in product(blends,policies):
     w,f,a=re.fullmatch(r'bagged_anchor_(balanced|unweighted)_(prob|logit)_a(10|25|50)',n).groups()
     g=f'logit_bagged_{w}_prob_mean' if f=='prob' else f'bagged_{w}_logit_mean'
     factor('blend minus anchor',n,k,H);factor('blend minus added margin',n,k,g)
    for w,f,k in product(weights,['prob','logit'],policies):
     for low,high in [('10','25'),('10','50'),('25','50')]:factor('higher minus lower blend weight',f'bagged_anchor_{w}_{f}_a{high}',k,f'bagged_anchor_{w}_{f}_a{low}')
    for f,a,k in product(['prob','logit'],['10','25','50'],policies):factor('blend balanced minus unweighted',f'bagged_anchor_balanced_{f}_a{a}',k,f'bagged_anchor_unweighted_{f}_a{a}')
    for w,a,k in product(weights,['10','25','50'],policies):factor('blend mean-logit minus logit-mean-probability',f'bagged_anchor_{w}_logit_a{a}',k,f'bagged_anchor_{w}_prob_a{a}')
    for n,k in product(logits,policies):factor('transformed minus native',n,k,n[6:])
    for w,k in product(weights,policies):factor('added mean-logit minus logit-mean-probability',f'bagged_{w}_logit_mean',k,f'logit_bagged_{w}_prob_mean')
    for f,k in product(['s17','s29','s43','prob_mean','logit_mean'],policies):factor('native balanced minus unweighted',f'bagged_balanced_{f}',k,f'bagged_unweighted_{f}')
    for source_b,source_u in [(f'bagged_balanced_s{seed}',f'bagged_unweighted_s{seed}') for seed in [17,29,43]]+[('bagged_balanced_prob_mean','bagged_unweighted_prob_mean'),('baseline_balanced','baseline_lgbm')]:
     for k in policies:factor('logit balanced minus unweighted','logit_'+source_b,k,'logit_'+source_u)
    for w,seed,k in product(weights,[17,29,43],policies):
     factor('probability aggregate minus native member',f'bagged_{w}_prob_mean',k,f'bagged_{w}_s{seed}')
     factor('mean-logit aggregate minus logit member',f'bagged_{w}_logit_mean',k,f'logit_bagged_{w}_s{seed}')
     factor('logit-probability aggregate minus logit member',f'logit_bagged_{w}_prob_mean',k,f'logit_bagged_{w}_s{seed}')
    controls=expected-set(blends)-{v[0] for v in refs.values()}-{'logit_baseline_balanced','logit_baseline_lgbm'};assert len(controls)==18
    for n,k in product(controls,policies):
     w='balanced' if '_balanced_' in n else 'unweighted';margin=n.startswith('logit_') or n.endswith('_logit_mean')
     factor('randomized minus matched unbagged',n,k,('logit_' if margin else '')+baseline[w])
    if len(facts)!=947:
        raise ValueError("Independent factor union differs")
    return facts


def audit_tables(c,fp,frame):
    """Reconstruct every grid value; actual source-model audits are a separate gate."""
    out=Path(c['output_dir']);h=run.history(c);names=NAMES;policies=POLICIES
    if set(h)!=set(names):
        raise ValueError('Selectable strategy union differs')
    cf=pd.read_csv(out/'crossfit.csv')
    check_keys(cf,['candidate','policy','eval_month'],product(names,policies,[5,6]))
    for row in cf.to_dict('records'):
        n,k,m=row['candidate'],row['policy'],row['eval_month'];rule=independent_rule(h[n],k,m)
        if rule!=json.loads(row['parameters']) or str(row['source_months'])!=','.join(map(str,rule['source_months'])):
            raise ValueError('Earlier-only rule differs')
        check_values(row,independent_metrics(h[n][m],independent_mask(h[n][m],rule)))
    summary=pd.read_csv(out/'summary.csv');index=summary.set_index(['candidate','policy'])
    check_keys(summary,['candidate','policy'],product(names,policies))
    for (n,k),g in cf.groupby(['candidate','policy']):
        check_values(index.loc[(n,k)],{v:float(g[v].mean()) for v in
            ['f1','f2','f3','precision','recall','flagged_fraction','average_precision','roc_auc']})
    ordered=summary.sort_values(['f2','recall','precision','candidate','policy'],ascending=[False,False,False,True,True])
    if list(ordered.index)!=list(range(len(summary))):
        raise ValueError('Actual macro winner/order differs')
    contrasts=pd.read_csv(out/'contrasts.csv')
    check_keys(contrasts,['contrast','candidate','policy','reference','reference_policy'],expected_factors())
    for row in contrasts.to_dict('records'):
        a,b=index.loc[(row['candidate'],row['policy'])],index.loc[(row['reference'],row['reference_policy'])]
        check_values(row,{v+'_difference':a[v]-b[v] for v in ['f2','recall','precision','flagged_fraction','average_precision']})
    pooled=pd.read_csv(out/'pooled.csv');check_keys(pooled,['candidate','policy'],product(names,policies))
    for row in pooled.to_dict('records'):
        g=cf[(cf.candidate==row['candidate'])&(cf.policy==row['policy'])]
        tn,fp_,fn,tp=[int(g[v].sum()) for v in ['tn','fp','fn','tp']]
        vals=dict(tn=tn,fp=fp_,fn=fn,tp=tp,precision=tp/(tp+fp_) if tp+fp_ else 0.,
                  recall=tp/(tp+fn) if tp+fn else 0.,flagged_fraction=(tp+fp_)/g.n.sum())
        for beta in [1,2]:
            den=(1+beta**2)*tp+beta**2*fn+fp_;vals['f'+str(beta)]=(1+beta**2)*tp/den if den else 0.
        check_values(row,vals)
    monthly=pd.read_csv(out/'score_diagnostics.csv');check_keys(monthly,['candidate','eval_month'],product(names,[4,5,6]))
    for row in monthly.to_dict('records'):
        p=h[row['candidate']][row['eval_month']]['scores']
        check_values(row,dict(n=len(p),score_min=p.min(),score_max=p.max(),score_mean=p.mean(),
            unique_scores=len(np.unique(p)),tied_fraction=1-len(np.unique(p))/len(p)))
    probabilities=OLD_BASES|set(SEEDS)|{f'bagged_{w}_prob_mean' for w in WEIGHTS}
    clips=pd.read_csv(out/'clipping.csv');check_keys(clips,['candidate','eval_month'],product(probabilities,[4,5,6]))
    for row in clips.to_dict('records'):
        p=run.prediction(c,'validation',row['eval_month'],row['candidate'])['scores'];logit(p)
        check_values(row,dict(n=len(p),epsilon=1e-6,input_min=p.min(),input_max=p.max(),
            below_epsilon=int((p<1e-6).sum()),above_one_minus_epsilon=int((p>1-1e-6).sum()),
            changed_count=int(((p<1e-6)|(p>1-1e-6)).sum())))
    div=pd.read_csv(out/'diversity.csv');pairs=diagnostic_pairs()
    check_keys(div,['pair','first','second','eval_month'],[(a,n,r,m) for (a,n,r),m in product(pairs,[4,5,6])])
    for row in div.to_dict('records'):
        m=row['eval_month'];a,b=h[row['first']][m]['scores'],h[row['second']][m]['scores']
        constant=np.ptp(a)==0 or np.ptp(b)==0
        if row['correlation_reason']!=('constant member' if constant else 'defined'):
            raise ValueError('Correlation reason differs')
        if constant:
            if not pd.isna(row['pearson']) or not pd.isna(row['spearman']):
                raise ValueError('Undefined correlation fabricated')
        else:
            check_values(row,dict(pearson=np.corrcoef(a,b)[0,1],spearman=np.corrcoef(rankdata(a),rankdata(b))[0,1]))
        check_values(row,dict(mean_absolute_difference=np.abs(a-b).mean(),unequal_scores=int((a!=b).sum())))
    joint=pd.read_csv(out/'decision_overlap.csv')
    check_keys(joint,['pair','first','second','eval_month','policy'],
               [(a,n,r,m,k) for (a,n,r),m,k in product(pairs,[5,6],policies)])
    for row in joint.to_dict('records'):
        n,r,m,k=[row[x] for x in ['first','second','eval_month','policy']];a,b=h[n][m],h[r][m]
        pa=independent_mask(a,independent_rule(h[n],k,m));pb=independent_mask(b,independent_rule(h[r],k,m))
        masks=dict(both=pa&pb,first_only=pa&~pb,second_only=~pa&pb,neither=~pa&~pb)
        vals={n:int(v.sum()) for n,v in masks.items()}
        vals.update({label+'_'+n:int(((a['labels']==y)&v).sum()) for y,label in [(1,'fraud'),(0,'legitimate')] for n,v in masks.items()})
        check_values(row,vals)
    model_table=pd.read_csv(out/'model_diagnostics.csv');check_keys(model_table,['candidate','eval_month'],product(SEEDS,[4,5,6]))
    pre=read_config(out/'preprocessing.json');refmeta=read_config(out/'reference_diagnostics.json')
    if set(pre)!={f'{m}:{n}' for n,m in product(SEEDS,[4,5,6])} or set(refmeta)!={f'{m}:{n}' for n,m in product(OLD_BASES,[4,5,6])}:
        raise ValueError('Preprocessing/reference key set differs')
    for row in model_table.to_dict('records'):
        n,m=row['candidate'],row['eval_month'];r=read_config(run.paths(c,'validation',m,n)[0])
        match=re.fullmatch(r'bagged_(balanced|unweighted)_s(17|29|43)',n);w,s=match.groups()
        if pre[f'{m}:{n}']!=r['preprocessing']:
            raise ValueError('Exported preprocessing differs')
        for key in ['class_weights','warnings','requested_model_params','booster_params']:
            if json.loads(row[key])!=r[key]:
                raise ValueError('Exported source parameter evidence differs')
        check_values(row,dict(weighting=w,bagging_seed=int(s),
            **{key:r[key] for key in ['fitted_tree_count','fit_seconds','peak_process_rss_bytes','train_rows','train_fraud']},
            transformed_dimension=r['preprocessing']['transformed_dimension']))
    for m,n in product([4,5,6],OLD_BASES):
        if refmeta[f'{m}:{n}']!=read_config(run.paths(c,'validation',m,n)[0]):
            raise ValueError('Exported reference evidence differs')
    expressions=pd.read_csv(out/'expressions.csv');expected=independent_expressions()
    check_keys(expressions,['candidate','atom'],[(r['candidate'],r['atom']) for r in expected])
    eindex=expressions.set_index(['candidate','atom'])
    for r in expected:
        check_values(eindex.loc[(r['candidate'],r['atom'])],{k:v for k,v in r.items() if k not in ['candidate','atom']})
    audit_compositions(c,fp,[4,5,6],'validation',names)
    dag=graph(names);wanted={f'validation-m{m}-{n}' for n,m in product(dag['scores'],[4,5,6])}
    for folder,suffix in [('runs','.json'),('predictions','.npz')]:
        if {p.stem for p in (out/folder).glob('validation-*'+suffix)}!=wanted:
            raise ValueError('Physical artifact key set differs')
    for m,n in product([4,5,6],dag['scores']):
        aligned(run.prediction(c,'validation',m,n),run.expected_population(frame,m))
    return dict(scores=120,selectable_scores=111,operating_rows=222,macro_rows=111,pooled_rows=111,
        contrasts=947,diversity_rows=18,overlap_rows=36,clipping_rows=39,expression_leaf_rows=60,
        bagged_model_rows=18,reference_results=15,winner=str(summary.iloc[0].candidate),
        policy=str(summary.iloc[0].policy),development_f2=float(summary.iloc[0].f2),
        actual_source_models_audited=False)


def audit_development(config_path=run.CONFIG,*,write=True):
    c,fp=run.initialize(config_path);out=Path(c['output_dir'])
    if write and (out/'selection.json').exists():raise ValueError('Frozen independent audit cannot be overwritten')
    done=read_config(out/'validation_complete.json')
    if done['fingerprint']!=fp:raise ValueError('Development fingerprint differs')
    run.verify_hashes({str(out/p):h for p,h in done['files'].items()})
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    records,ds,work=run.prepare(c,fp,frame,base,allow_missing=False)
    if len(records)!=33 or len(ds)!=87:raise ValueError('Complete saved source DAG required')
    prep=read_config(out/'preparation.json');actual=run.work_accounting(c,records,ds,work)
    for key in ['base_results','bagged_reuses','new_bagged_fits','reference_reuses','reproduced_additive_fits',
                'reproduced_tree_fits','derived_results','derived_reuses','new_derived_computations']:
        if prep[key]!=actual[key]:raise ValueError('Actual preparation accounting differs: '+key)
    result=audit_tables(c,fp,frame);result['actual_source_models_audited']=True
    from .unweighted_fusion_report import audit_reference_values
    audit_reference_values(pd.read_csv(out/'summary.csv').set_index(['candidate','policy']))
    run.guard(c)
    if write:
        write_json(out/'independent_development_audit.json',dict(fingerprint=fp,
            validation_sha256=digest(out/'validation_complete.json'),saved_model_checks=len(records),
            independent_preprocessing_inference_and_parameters=True,independent_policies_metrics_factors=True,
            elapsed_seconds=time.perf_counter()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            created_at_utc=run.now(),**result))
    return records


def uncertainty(c,s,cf,h,data,masks):
    dint,ddraws=development(c,s,cf,h)
    fint,fdraws=intervals(data,masks,np.random.default_rng(c['bootstrap_seed']),c['bootstrap_replicates'],'7_consumed')
    refs=[n for n in s['choices'] if n!='selected'];b=c['bootstrap_replicates']
    check_keys(dint,['scope','reference'],product(['5','6','macro'],refs))
    check_keys(fint,['scope','reference'],[('7_consumed',r) for r in refs])
    check_keys(ddraws,['scope','replicate'],product(['5','6'],range(b)))
    if set(fdraws)!=set(refs):raise ValueError('Exact consumed draw columns differ')
    audit_intervals(dint,ddraws,fint,fdraws,s['choices'],b)
    rng=np.random.default_rng(c['bootstrap_seed'])
    for m in [5,6]:
        dm={a:h[n][m] for a,(n,k) in s['choices'].items()}
        pm={a:independent_mask(dm[a],independent_rule(h[n],k,m)) for a,(n,k) in s['choices'].items()}
        direct_draw_audit(dm,pm,rng,ddraws[ddraws.scope==str(m)].reset_index(drop=True))
    direct_draw_audit(data,masks,np.random.default_rng(c['bootstrap_seed']),fdraws)
    return dint,ddraws,fint,fdraws.assign(replicate=np.arange(b))


def operating_figure(summary,dest):
    index=summary.set_index(['candidate','policy']);fig,axes=plt.subplots(2,2,figsize=(14,10))
    points=summary[['candidate','policy','f2','precision','recall']].copy();curves=[];legends={}
    for k,marker in [('pooled','o'),('latest','^'),('rank_fraction','s')]:
        g=summary[summary.policy==k]
        artist=axes[0,0].scatter(g.recall,g.precision,label=k,marker=marker,s=20,alpha=.65)
        if not np.allclose(artist.get_offsets(),g[['recall','precision']].to_numpy(),atol=1e-12,rtol=0):
            raise ValueError('Rendered operating coordinates differ')
    axes[0,0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 111 strategy / policy operating points')
    axes[0,0].legend();legends['operating']=axes[0,0].get_legend_handles_labels()[1]
    for ax,k in zip([axes[0,1],axes[1,0],axes[1,1]],POLICIES):
        for w,f in product(WEIGHTS,['prob','logit']):
            controls=run.scores.curve_controls(w,f);x=[a for a,n in controls]
            y=[float(index.loc[(n,k),'f2']) for a,n in controls]
            label=w+' / '+('logit(probability mean)' if f=='prob' else 'mean(logits)')
            line,=ax.plot(x,y,marker='o',linestyle='-' if f=='prob' else '--',
                          color='tab:blue' if w=='balanced' else 'tab:orange',label=label)
            if not np.array_equal(line.get_xdata(),x) or not np.array_equal(line.get_ydata(),y):
                raise ValueError('Rendered curve differs')
            curves += [dict(weighting=w,form=f,policy=k,alpha=a,candidate=n,f2=float(index.loc[(n,k),'f2'])) for a,n in controls]
        ax.set(xlabel='Addition weight (0 = anchor; 1 = added margin)',ylabel='Macro development F2',title=k+' policy')
        ax.set_xticks([0,.1,.25,.5,1]);ax.legend(fontsize=8)
        legends[k]=ax.get_legend_handles_labels()[1]
    fig.tight_layout();fig.savefig(dest/'bagged_anchor_operating.png',dpi=160);plt.close(fig)
    points.to_csv(dest/'plotted_operating_points.csv',index=False)
    pd.DataFrame(curves).to_csv(dest/'plotted_alpha_curves.csv',index=False)
    write_json(dest/'figure_legends.json',legends)
    audit_figure(summary,dest)


def audit_figure(summary,dest):
    points=pd.read_csv(dest/'plotted_operating_points.csv');curves=pd.read_csv(dest/'plotted_alpha_curves.csv')
    index=summary.set_index(['candidate','policy'])
    check_keys(points,['candidate','policy'],product(NAMES,POLICIES))
    for r in points.to_dict('records'):
        check_values(r,index.loc[(r['candidate'],r['policy'])][['f2','precision','recall']].to_dict())
    check_keys(curves,['weighting','form','policy','alpha'],product(WEIGHTS,['prob','logit'],POLICIES,[0.,.1,.25,.5,1.]))
    for r in curves.to_dict('records'):
        w,f,k,a=[r[v] for v in ['weighting','form','policy','alpha']]
        expected=H if a==0 else (f'logit_bagged_{w}_prob_mean' if f=='prob' else f'bagged_{w}_logit_mean') if a==1 else f'bagged_anchor_{w}_{f}_a{round(a*100):02d}'
        if r['candidate']!=expected:raise ValueError('Canonical figure endpoint differs')
        check_values(r,dict(f2=float(index.loc[(expected,k),'f2'])))
    legends=read_config(dest/'figure_legends.json')
    expected=[w+' / '+f for w,f in product(WEIGHTS,['logit(probability mean)','mean(logits)'])]
    if legends!={'operating':POLICIES,**{k:expected for k in POLICIES}}:
        raise ValueError('Figure legend identities differ')


def audit_final(c,s,frame,base):
    out=Path(c['output_dir']);names={n for n,k in s['choices'].values()};sha=digest(out/'selection.json')
    records,ds,work=run.prepare(c,s['fingerprint'],frame,base,'exploratory',names,False)
    audit_compositions(c,s['fingerprint'],[7],'exploratory',names)
    access=read_config(out/'exploratory_access.json');done=read_config(out/'transfer_complete.json')
    if access['selection_sha256']!=sha or done['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc'] or done['table_sha256']!=digest(out/'exploratory.csv'):
        raise ValueError('Transfer evidence differs')
    actual=run.work_accounting(c,records,ds,work)
    for key,value in actual.items():
        if key!='invocation_work' and done[key]!=value:raise ValueError('Actual transfer accounting differs')
    dag=graph(names);wanted={f'exploratory-m7-{n}' for n in dag['scores']}
    for folder,suffix in [('runs','.json'),('predictions','.npz')]:
        if {p.stem for p in (out/folder).glob('exploratory-*'+suffix)}!=wanted:
            raise ValueError('Exact bounded transfer artifact set differs')
    data=run.final_data(c,s);h=run.history(c);final=pd.read_csv(out/'exploratory.csv');masks={}
    check_keys(final,['role'],[(a,) for a in s['choices']])
    for a,d in data.items():
        aligned(d,data['selected']);n,k=s['choices'][a]
        if s['rules'][a]!=independent_rule(h[n],k,7):raise ValueError('Final rule chronology differs')
        masks[a]=independent_mask(d,s['rules'][a]);row=final[final.role==a].iloc[0]
        if row.candidate!=n or row.policy!=k or json.loads(row.parameters)!=s['rules'][a]:raise ValueError('Final role/rule differs')
        check_values(row,independent_metrics(d,masks[a]))
    return records,data,masks,final


def report(config_path=run.CONFIG):
    c,s=run.load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir'])
    if any((dest/n).exists() for n in ['report_hashes.json','generated_report_hashes.json']):
        raise ValueError('Completed scientific report cannot be overwritten')
    records=audit_development(config_path,write=False)
    frame=load_data(c);base=features(frame);final_records,data,masks,final=audit_final(c,s,frame,base)
    cf=pd.read_csv(out/'crossfit.csv');summary=pd.read_csv(out/'summary.csv');h=run.history(c)
    dint,ddraws,fint,fdraws=uncertainty(c,s,cf,h,data,masks)
    for n,t in [('development_intervals',dint),('development_draws',ddraws),('consumed_intervals',fint),('consumed_draws',fdraws)]:
        t.to_csv(dest/(n+'.csv'),index=False)
    for name in run.TABLES+['validation_complete.json','independent_development_audit.json','selection.json',
        'exploratory_access.json','transfer_complete.json','exploratory.csv','declaration.json','checks.json',
        'preparation.json','resource_preflight.json','source_miniature_gate.json']:
        shutil.copyfile(out/name,dest/name)
    shutil.copyfile(config_path,dest/'config.json');shutil.copyfile(c['protocol_path'],dest/'protocol.md')
    write_json(dest/'run_provenance.json',records);write_json(dest/'transfer_provenance.json',final_records)
    winner=summary.iloc[0];index=summary.set_index(['candidate','policy']);sha=digest(out/'selection.json')
    gains={a:float(winner.f2-index.loc[tuple(nk),'f2']) for a,nk in REFS.items()}
    fg={r.reference:float(r.difference) for r in fint.itertuples()}
    measured=all(gains[a]>0 and fg[a]>0 for a in gains)
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,development_gain=gains,consumed_gain=fg,
        measured_gain_on_both_scopes=measured,direct_draws_checked=1500,saved_development_models_checked=33,
        saved_final_models_checked=len(final_records),
        conditional_clear_development_gain=all(float(dint[(dint.scope=='macro')&(dint.reference==a)].iloc[0].ci_low)>0 for a in gains),
        conditional_clear_consumed_gain=all(float(fint[fint.reference==a].iloc[0].ci_low)>0 for a in gains),
        preservation_checked=True,development_intervals=len(dint),consumed_intervals=len(fint),visible_roles=len(s['choices']))
    write_json(dest/'audit.json',audit);operating_figure(summary,dest)
    contrasts=pd.read_csv(out/'contrasts.csv');effects=contrasts.groupby('contrast',as_index=False).agg(
        rows=('f2_difference','size'),minimum=('f2_difference','min'),maximum=('f2_difference','max'),
        mean_difference=('f2_difference','mean'),positive=('f2_difference',lambda x:int((x>0).sum())),
        negative=('f2_difference',lambda x:int((x<0).sum())),tied=('f2_difference',lambda x:int((x==0).sum())))
    effects.to_csv(dest/'matched_effect_summary.csv',index=False)
    checks=read_config(out/'checks.json');prep=read_config(out/'preparation.json');done=read_config(out/'transfer_complete.json')
    ref_rows=[]
    for a,(n,k) in REFS.items():
        dev=index.loc[(n,k)];f=final[final.role==a].iloc[0]
        ref_rows.append(dict(role=a,development_f2=dev.f2,development_difference=gains[a],consumed_f2=f.f2,consumed_difference=fg[a]))
    blends=summary[summary.candidate.isin(BLENDS)]
    text=f'''# Fixed bagged-anchor fusion findings

Selected **{winner.candidate} / {winner.policy}**, actual macro chronological development F2 **{winner.f2:.6f}**. Positive measured differences against all five fixed references in both scopes: **{measured}**. Completing this finite experiment does not itself establish the broader improvement goal.

## Fixed design and actual work

The existing margin H retains its original arithmetic. Balanced/unweighted three-seed bagged ensembles supply either logit of the probability mean or mean of individually clipped logits. Each is added as `(1-alpha)*H+alpha*G` at weights 0.1/0.25/0.5. The two forms have different scaling and are not interchangeable; no second logit is applied to a margin. Sixty expression-leaf diagnostics were checked without replacing the declared scoring association. Original models, whole-training class weights, train-only preprocessing and sampling settings remain unchanged; internal bags are not exactly class matched. Probability-like scores are not claimed calibrated.

Development used **{prep['new_bagged_fits']} new bagged fits**, **{prep['bagged_reuses']} bagged reuses**, **{prep['reference_reuses']} old-reference reuses**, **{prep['reproduced_additive_fits']} additive reproductions** and **{prep['reproduced_tree_fits']} tree reproductions**. The 33 bases, 21 preserved derived results and 66 new calculations produced 120 physical/111 selectable scores, 222 forward operating rows and 947 contrasts. Actual derived reuses/calculations were {prep['derived_reuses']}/{prep['new_derived_computations']}. Predictor-only allocation, preserved nine-fit sampling evidence and older ancestry passed with zero new production miniature fits. The full methodological suite passed {checks['tests_passed']} tests; its actual fit ledger separates six current-study tiny fits from {checks['actual_historical_test_fits']} completed historical-module fits in that invocation ({checks['actual_historical_test_fit_attempts']} attempts, {checks['actual_historical_test_fit_failures']} expected failures). Earlier focused checks/retries are retained separately.

Models use earlier months only. Month 5 rules use month 4; month 6 rules use months 4/5 or latest month 5. Selection maximizes actual macro F2 beta=2, then recall/precision and deterministic IDs, with no review quota or precision floor. AP, pooled F2 and diagnostics do not select alternatives. Sources, configurations, software, development files and exact role/DAG/cache bounds were frozen before further consumed-month access.

## Five required reference comparisons

{markdown(pd.DataFrame(ref_rows))}

## All development operating points

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision','tp','fp','fn']])}

## All twelve blends and complete matched effects

{markdown(blends[['candidate','policy','f2','precision','recall','flagged_fraction']])}

Every fixed-reference, own-policy, blend-versus-anchor/member, higher/lower weight, matched weighting/form, logit/native, aggregate/constituent and randomized/unbagged factor is retained in `contrasts.csv`. The table below covers all factor groups and preserves ties and negative effects. These are otherwise-matched finite comparisons, not universal or causal claims. Eighteen diversity rows, 36 decision overlaps, 39 clipping rows and all original preprocessing/numerical evidence accompany results. High correlation or different errors alone do not establish a benefit.

{markdown(effects)}

## Conditional development uncertainty

{markdown(dint)}

## Already-consumed month 7: frozen exploratory transfer

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

Exactly {len(s['choices'])} visible roles transferred, including aliases, with {done['base_results']} unique base results and {done['total_scores']} physical scores. Actual new bagged fits/reuses were {done['new_bagged_fits']}/{done['bagged_reuses']}; reference reuses/additive reproductions/tree reproductions were {done['reference_reuses']}/{done['reproduced_additive_fits']}/{done['reproduced_tree_fits']}. Only the frozen deduplicated DAG was evaluated; no losing-grid sweep, hindsight cutoff or winner replacement occurred.

{markdown(fint)}

## Limits and reproduction

All 1500 paired fixed-decision draws and every interval/quantile were independently reconstructed, retaining aliases. Intervals condition on fixed models/rules and exclude training-seed/retraining, adaptive source/weight/search/selection, temporal/generative uncertainty and multiplicity correction. They are not uncertainty over all training seeds. Clear-gain language requires positive conditional lower bounds on the named scope. Eight synthetic months, one variant, unknown label delays and adaptively reused development limit generalization. Month 7 was already consumed and cannot provide independent confirmation.

All sixteen completed histories, separate Newton identities and the stopped attempt remain preserved. Settings are analyst-selected and open to group review. README documents data/pins, sixteen exact selection restores and ordered execution; frozen execution/test/access/report records must not be overwritten. Codex assisted with design, implementation, execution, independent audits and interpretation using pinned repository APIs. No public notebook was copied; the group must acknowledge assistance and explain the submitted work.

![All 111 operating points and twelve matched addition curves](bagged_anchor_operating.png)
'''
    (dest/'findings.md').write_text(text);run.guard(c)
    write_json(dest/'generated_report_hashes.json',{str(p):digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name not in ['generated_report_hashes.json','report_hashes.json']})
    print('Scientific report generated; measured acceptance:',measured,flush=True)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('step',choices=['audit_development','report'])
    parser.add_argument('--config',default=run.CONFIG);args=parser.parse_args();globals()[args.step](args.config)
