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
from . import interaction_fusion_study as run


def independent_identity(rep, C, weight, form):
    n=f'{rep}_c{str(C).replace(".","p")}_{weight}'
    if form=='native':return n
    if form=='logit':return 'logit_'+n
    if (rep,C,weight,form)==('spline',.1,'unweighted',.25):return run.RECENT_FUSION
    return f'interaction_blend_{n}_a{int(100*form)}'


def expected_factors():
    """Independent Cartesian factors, without the producer's matching helpers."""
    reps=['spline','interaction'];cs=[.1,1.];weights=['unweighted','balanced']
    forms=['native','logit',.1,.25,.5,.75];policies=['pooled','latest','rank_fraction']
    ident=independent_identity
    names=[ident(r,C,w,f) for r,C,w,f in itertools.product(reps,cs,weights,forms)]
    names += [run.FUSION,'baseline_balanced','baseline_lgbm']
    rows=[]
    for n,k in itertools.product(names,policies):
        rows += [('original operating reference',n,k,ref,rk) for ref,rk in
                 [(run.RECENT_FUSION,'latest'),(run.FUSION,'pooled'),('baseline_balanced','pooled'),('baseline_lgbm','pooled')]]
        if k!='pooled':rows.append(('policy minus own pooled',n,k,n,'pooled'))
    for r,C,w,a,k in itertools.product(reps,cs,weights,[.1,.25,.5,.75],policies):
        rows += [(label,ident(r,C,w,a),k,ref,k) for label,ref in
                 [('blend minus native',ident(r,C,w,'native')),('blend minus transformed',ident(r,C,w,'logit')),
                  ('blend minus tree',run.FUSION)]]
    for C,w,f,k in itertools.product(cs,weights,forms,policies):
        rows.append(('interaction minus additive',ident('interaction',C,w,f),k,ident('spline',C,w,f),k))
    for r,C,f,k in itertools.product(reps,cs,forms,policies):
        rows.append(('balanced minus unweighted',ident(r,C,'balanced',f),k,ident(r,C,'unweighted',f),k))
    for r,w,f,k in itertools.product(reps,weights,forms,policies):
        rows.append(('larger C minus smaller C',ident(r,1.,w,f),k,ident(r,.1,w,f),k))
    for r,C,w,(lo,hi),k in itertools.product(reps,cs,weights,list(itertools.combinations([.1,.25,.5,.75],2)),policies):
        rows.append(('larger alpha minus smaller alpha',ident(r,C,w,hi),k,ident(r,C,w,lo),k))
    for r,C,w,k in itertools.product(reps,cs,weights,policies):
        rows.append(('transformed minus native',ident(r,C,w,'logit'),k,ident(r,C,w,'native'),k))
    if len(rows)!=1386 or len(set(rows))!=1386:raise ValueError('Independent factor union differs')
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


def audit_compositions(c, fp, months, phase, names):
    names=set(names)
    for m in months:
        x=run.prediction(c,phase,m,'xgb_d4_w1');b=run.prediction(c,phase,m,'baseline_balanced')
        run.aligned(x,b)
        g=.5*independent_logit(x['scores'])+.5*independent_logit(b['scores'])
        tree=run.prediction(c,phase,m,run.FUSION);run.aligned(x,tree);close(g,tree['scores'],inference=True)
        for spec in c['strategies']:
            n=spec['id']
            if n not in names or spec['kind'] not in ['logit','blend']:continue
            raw=run.prediction(c,phase,m,spec['members'][0]);actual=run.prediction(c,phase,m,n)
            run.aligned(raw,tree);run.aligned(raw,actual);u=independent_logit(raw['scores'])
            expected=u if spec['kind']=='logit' else spec['alpha']*u+(1-spec['alpha'])*g
            close(actual['scores'],expected,inference=True)
            rp,pp=run.paths(c,phase,m,n);r=read_config(rp)
            expected_parents={v:{str(p):digest(p) for p in run.paths(c,phase,m,v)} for v in spec['members']}
            if (r['dependencies']!=expected_parents or r['fingerprint']!=fp or r['spec']!=spec
                    or r['epsilon']!=1e-6 or r['prediction_sha256']!=digest(pp)):
                raise ValueError('Independent composition provenance differs')
            run.verify_hashes(r['origin_artifacts'])


def audit_tables(c, frame, base, fp=None):
    out=Path(c['output_dir']);h=run.history(c);fp=run.fingerprint() if fp is None else fp
    expected_names={independent_identity(r,C,w,f) for r,C,w,f in itertools.product(
        ['spline','interaction'],[.1,1.],['unweighted','balanced'],['native','logit',.1,.25,.5,.75])}
    expected_names.update([run.FUSION,'baseline_balanced','baseline_lgbm'])
    if set(h)!=expected_names:raise ValueError('Independent strategy union differs')
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
    check_keys(clips,['candidate','eval_month'],{(n,m) for n in run.MEMBERS+['xgb_d4_w1','baseline_balanced','baseline_lgbm'] for m in [4,5,6]})
    for r in clips.to_dict('records'):
        p=run.prediction(c,'validation',r['eval_month'],r['candidate'])['scores'];independent_logit(p)
        check_values(r,dict(n=len(p),epsilon=1e-6,input_min=p.min(),input_max=p.max(),
            below_epsilon=int((p<1e-6).sum()),above_one_minus_epsilon=int((p>1-1e-6).sum()),
            changed_count=int((p!=np.maximum(1e-6,np.minimum(1-1e-6,p))).sum())))
    div=pd.read_csv(out/'diversity.csv')
    check_keys(div,['candidate','policy','eval_month'],{(n,k,m) for n in run.MEMBERS for k in c['policies'] for m in [5,6]})
    for r in div.to_dict('records'):
        n,k,m=r['candidate'],r['policy'],r['eval_month'];a,b=h['logit_'+n][m],h[run.FUSION][m]
        pa=independent_mask(a,independent_rule(h['logit_'+n],k,m));pb=independent_mask(b,independent_rule(h[run.FUSION],k,m))
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
    wanted={f'validation-m{m}-{n}' for n in expected_names|{'xgb_d4_w1'} for m in [4,5,6]}
    for folder,suffix in [('runs','.json'),('predictions','.npz')]:
        if {p.stem for p in (out/folder).glob('validation-*'+suffix)}!=wanted:raise ValueError('Physical artifact set differs')
    audit_feature_exports(c,frame,base)
    return dict(scores=156,selectable_scores=153,operating_rows=306,macro_rows=153,contrasts=1386,
        clipping_rows=33,diversity_rows=48,feature_rows=1392,pair_rows=72,
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
    for table,cols,expected_keys,count in [(fd,['candidate','eval_month','population','feature'],expected_fd,1392),
         (pdg,['candidate','eval_month','population','left','right'],expected_pd,72),
         (coefficients,['candidate','eval_month','feature'],expected_coef,len(expected_coef))]:
        if len(table)!=count or set(table[cols].itertuples(index=False,name=None))!=expected_keys:
            raise ValueError('Complete diagnostic/coefficient keys differ')
    if any(len(t)!=24 for t in [solver,numbers,resources]):raise ValueError('Numerical/resource count differs')


def audit_development(config_path=run.CONFIG, *, write=True):
    c,fp=run.initialize(config_path);out=Path(c['output_dir'])
    if write and (out/'selection.json').exists():raise ValueError('Frozen independent audit cannot be overwritten')
    done=read_config(out/'validation_complete.json')
    if done['fingerprint']!=fp:raise ValueError('Development identity differs')
    run.verify_hashes({str(out/p):h for p,h in done['files'].items()})
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    records=run.prepare(c,fp,frame,base,allow_missing=False)
    preparation=read_config(out/'preparation.json')
    if (len(records)!=33 or preparation['base_reuses']!=sum(bool(r.get('reused')) for r in records)
            or preparation['new_fits']!=sum(not r.get('reused',False) for r in records)):
        raise ValueError('Actual development fit/reuse evidence differs')
    result=audit_tables(c,frame,base,fp)
    run.source_audit.audit_reference_values(pd.read_csv(out/'summary.csv').set_index(['candidate','policy']))
    run.guard(c)
    if write:
        write_json(out/'independent_development_audit.json',dict(fingerprint=fp,
            validation_sha256=digest(out/'validation_complete.json'),saved_model_checks=len(records),
            independent_statistics_inference_and_gradient=True,independent_policies_metrics_factors=True,
            elapsed_seconds=time.perf_counter()-start,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            created_at_utc=run.now(),**result))
    return records


def operating_figure(c,summary,dest):
    index=summary.set_index(['candidate','policy']);fig,axes=plt.subplots(1,3,figsize=(17,5))
    for kind,label in [('native','Native logistic'),('logit','Logit logistic'),('blend','Fixed blend'),('reference','Tree reference')]:
        ids=[s['id'] for s in c['strategies'] if s['kind']==kind];t=summary[summary.candidate.isin(ids)]
        axes[0].scatter(t.recall,t.precision,label=label,s=20,alpha=.7)
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 153 strategy/policy results')
    axes[0].legend(fontsize=8)
    for ax,C in zip(axes[1:],[.1,1.]):
        for rep,weight in itertools.product(['spline','interaction'],['unweighted','balanced']):
            values=[float(index.loc[(independent_identity(rep,C,weight,a),'pooled'),'f2']) for a in [.1,.25,.5,.75]]
            ax.plot([.1,.25,.5,.75],values,marker='o',label=rep+' / '+weight)
        ax.axhline(float(index.loc[(run.FUSION,'pooled'),'f2']),color='grey',linewidth=.8,label='Tree fusion / pooled')
        ax.set(xlabel='Logistic-member weight (alpha)',ylabel='Actual macro F2',title=f'Matched pooled policy, C={C}')
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'fusion_operating.png',dpi=160);plt.close(fig)


def report(config_path=run.CONFIG):
    c,s=run.load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
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
    reference_roles={'recent_fusion':(run.RECENT_FUSION,'latest'),'fusion':(run.FUSION,'pooled'),
                     'balanced':('baseline_balanced','pooled'),'unweighted':('baseline_lgbm','pooled')}
    gain={a:float(winner.f2-index.loc[nk,'f2']) for a,nk in reference_roles.items()}
    fg={r.reference:float(r.difference) for r in fint.itertuples()}
    measured=all(gain[a]>0 and fg[a]>0 for a in reference_roles)
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,development_gain=gain,consumed_gain=fg,
        measured_gain_on_both_scopes=measured,direct_draws_checked=1500,saved_development_models_checked=len(records),
        conditional_clear_development_gain=all(float(dint[(dint.scope=='macro')&(dint.reference==a)].iloc[0].ci_low)>0 for a in gain),
        conditional_clear_consumed_gain=all(float(fint[fint.reference==a].iloc[0].ci_low)>0 for a in gain),preservation_checked=True)
    write_json(dest/'audit.json',audit);operating_figure(c,summary,dest)
    comp=pd.read_csv(out/'contrasts.csv');counts=read_config(out/'preparation.json');receipt=read_config(out/'checks.json')
    text=f'''# Interaction-score fusion findings

Selected **{winner.candidate} / {winner.policy}**, macro forward F2 **{winner.f2:.6f}**. Positive measured differences against all four original references on both development and consumed transfer: **{measured}**. Finite-search completion does not itself establish the broader performance goal.

## Design and execution

Eight fixed spline/interaction members cross C=0.1/1 with unweighted/balanced training. Native and clipped-log-odds singles plus fixed logistic-member weights 0.10/0.25/0.50/0.75 produce 51 distinct strategies including the original tree references. The already-tested additive C0.1/unweighted/25% blend retains its canonical historical ID and reference precedence. Only probability-like members receive logit clipping (epsilon 1e-6); the existing tree fusion is already a finite margin. Scores are not asserted calibrated probabilities; weights are analyst-selected sensitivities, not confidence or cost estimates.

Development used {counts['base_reuses']} checked base reuses and {counts['new_fits']} new full model fits, producing 156 physical scores, 153 selectable scores, 306 forward rows, 153 macro and pooled summaries and 1,386 complete contrasts. Three canonical recent-score reuses were checked against own parent arithmetic. The new cache tests made four genuine tiny fits per full invocation; historical test-module fits are separate. The recorded full suite passed {receipt['tests_passed']} tests. No new production miniature fits were required; the preserved eight-fit source gate remains checked. Actual resource preflight covers all 903,157 eligible months-0–6 training predictors. Historical fit resource values and current verification/composition measurements are labelled separately.

Models and preprocessing use only earlier months. Month-5 policies learn on month 4; month-6 policies use months 4/5 or latest month 5. Selection maximizes actual macro positive-class F2 beta=2, then recall, precision and deterministic IDs, with no review quota or precision floor. AP, pooled F2, diversity and joint-error counts remain descriptive. Equal nominal C does not mean equal effective complexity across representations.

## Complete forward results

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

## Matched effects and original references

Every original-reference, own-policy, blend/native/transformed/tree, representation, training-weight, C and alpha comparison is retained in contrasts.csv, including ties and negative effects. Clipping has 33 rows and common-margin diversity has 48 rows with independent earlier-only decisions and explicit null reasons for constant correlations. Correlations and pair blocks are associations, not causal effects or selection rules. Source features, coefficients, training statistics, numerical objectives/gradients and serialized inference were independently checked; 24 logistic blocks, twelve pair blocks, 1,392 original-feature rows and 72 pair rows remain exported.

{markdown(comp[comp.candidate==winner.candidate][['contrast','policy','reference','reference_policy','f2_difference']])}

## Conditional development differences

{markdown(dint)}

## Already-consumed month 7

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

Only {len(s['choices'])} frozen roles transferred with deduplicated work: {done['base_reuses']} base reuses and {done['new_fits']} new fits. Exact source/score DAG, original-cache availability, tighter type bounds and final earlier-only rules were frozen before the recorded access. No losing-grid sweep, hindsight cutoff or replacement of the development winner was used.

{markdown(fint)}

## Limitations, reproduction and assistance

All 1,500 paired fixed-decision row draws and quantiles were independently reconstructed with seed 1027 and 500 draws per month. Intervals condition on fixed models/policies and exclude adaptive source/C/alpha/pair choice, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity adjustment. Clear-gain language requires a positive conditional lower bound on the named scope. Month 7 was already consumed and cannot provide independent confirmation. Synthetic data and eight monthly periods limit generalization.

All thirteen completed studies, the stopped attempt and separate Newton fit/analysis identities remain preserved. README documents the pinned data/environment, thirteen exact selection restores and ordered clean execution; existing frozen audit/selection/report artifacts must not be overwritten. Settings remain analyst-selected and open to group review. Codex assisted with design, implementation, execution, independent verification, diagnostics and reporting using preserved repository interfaces and pinned scikit-learn/SciPy/XGBoost/LightGBM APIs. No public notebook was copied. The group must acknowledge assistance and understand submitted analysis.

![All operating results and matched alpha effects](fusion_operating.png)
'''
    (dest/'findings.md').write_text(text)
    run.guard(c)
    write_json(dest/'report_hashes.json',{str(p):digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='report_hashes.json'})
    print('Report complete; measured goal acceptance:',measured,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit_development','report'])
    parser.add_argument('--config',default=run.CONFIG);args=parser.parse_args();globals()[args.command](args.config)
