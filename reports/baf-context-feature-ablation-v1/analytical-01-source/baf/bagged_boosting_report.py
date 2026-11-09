"""Independent finite-grid checks and descriptive conditional F2 reporting."""
from __future__ import annotations
import argparse, json, resource, shutil, time
from pathlib import Path
from itertools import product, combinations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import read_config, write_json, digest, load_data, features
from .interaction_report import check_values, close
from .spline_fusion_report import independent_rule, independent_mask, independent_metrics
from .spline_report import direct_draw_audit, markdown
from .direct_ranking_report import development, intervals
from .score_fusion_report import audit_intervals
from . import bagged_boosting as run


def independent_declarations():
    refs=['interaction_blend_spline_c1p0_balanced_a10','margin_blend_spline_c0p1_unweighted_a25',
          'logit_mean_xgb_d4_w1_baseline_balanced_a50','baseline_balanced','baseline_lgbm']
    singles={f'bagged_{w}_s{s}':(w,s) for w,s in product(['balanced','unweighted'],[17,29,43])}
    means={f'bagged_{w}_{f}_mean':(w,f) for w,f in product(['balanced','unweighted'],['prob','logit'])}
    return refs,singles,means


def expected_factors():
    refs,singles,means=independent_declarations();names=refs+list(singles)+list(means);out=[]
    ref_policies=dict(zip(refs,['latest','latest','pooled','pooled','pooled']))
    roles=['latest_fusion','recent_fusion','fusion','balanced','unweighted']
    for n,k in product(names,['pooled','latest','rank_fraction']):
        out += [('fixed reference '+a,n,k,r,ref_policies[r]) for a,r in zip(roles,refs)]
        if k!='pooled':out.append(('policy minus own pooled',n,k,n,'pooled'))
    for n,(w,form) in {**singles,**means}.items():
        for k in ['pooled','latest','rank_fraction']:
            ref='baseline_balanced' if w=='balanced' else 'baseline_lgbm'
            out.append(('randomized minus unbagged',n,k,ref,k))
            if n in means:
                out += [('aggregate minus member',n,k,f'bagged_{w}_s{s}',k) for s in [17,29,43]]
    for w,k in product(['balanced','unweighted'],['pooled','latest','rank_fraction']):
        out.append(('logit mean minus probability mean',f'bagged_{w}_logit_mean',k,f'bagged_{w}_prob_mean',k))
    for form,k in product(['s17','s29','s43','prob_mean','logit_mean'],['pooled','latest','rank_fraction']):
        out.append(('balanced minus unweighted',f'bagged_balanced_{form}',k,f'bagged_unweighted_{form}',k))
    return set(out)


def check_keys(table,columns,keys):
    actual=list(table[columns].itertuples(index=False,name=None))
    if len(actual)!=len(set(actual)) or set(actual)!=set(keys):raise ValueError('Exact output key set differs: '+','.join(columns))


def logit(values):
    p=np.asarray(values,dtype=float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probability domain')
    q=np.maximum(1e-6,np.minimum(1-1e-6,p))
    return np.log(q/(1-q))


def audit_compositions(c,fp,months,phase,names):
    refs,singles,means=independent_declarations();dag=run.dependencies(names)
    for m in months:
        data={n:run.prediction(c,phase,m,n) for n in dag['scores']}
        for n in dag['scores']:
            if n not in dag['derived']:continue
            d=data[n];rp,pp=run.paths(c,phase,m,n);r=read_config(rp)
            if r['candidate']!=n or r['phase']!=phase or r['eval_month']!=m or r['fingerprint']!=fp or r['prediction_sha256']!=digest(pp) or r['epsilon']!=1e-6:
                raise ValueError('Derived provenance identity differs')
            if n in means:
                w,form=means[n];members=[f'bagged_{w}_s{s}' for s in [17,29,43]]
                arrays=[logit(data[t]['scores']) if form=='logit' else np.asarray(data[t]['scores'],dtype=np.float64) for t in members]
                for a in arrays:
                    if not np.isfinite(a).all() or (form=='prob' and ((a<0)|(a>1)).any()):raise ValueError('Invalid aggregate member')
                expected=(arrays[0]+arrays[1]+arrays[2])/3.
                spec=dict(id=n,kind='aggregate',weighting=w,form=form,members=members)
                if r['spec']!=spec or r['arithmetic']!='ordered float64 sum / 3':raise ValueError('Aggregate arithmetic declaration differs')
            else:
                tree='logit_mean_xgb_d4_w1_baseline_balanced_a50'
                if n==tree:
                    members=['xgb_d4_w1','baseline_balanced'];expected=.5*logit(data[members[0]]['scores'])+.5*logit(data[members[1]]['scores'])
                elif n=='margin_blend_spline_c0p1_unweighted_a25':
                    members=['spline_c0p1_unweighted',tree];expected=.25*logit(data[members[0]]['scores'])+.75*data[tree]['scores']
                elif n=='interaction_blend_spline_c1p0_balanced_a10':
                    members=['spline_c1p0_balanced',tree];expected=.1*logit(data[members[0]]['scores'])+.9*data[tree]['scores']
                else:raise ValueError('Undeclared reference margin')
                if r['spec']!=run.old_lookup(n) or r['arithmetic']!='preserved reference':raise ValueError('Reference recipe differs')
            for member in members:run.aligned(d,data[member])
            close(d['scores'],expected,inference=True)
            expected_deps={n:{str(p):digest(p) for p in run.paths(c,phase,m,n)} for n in members}
            if r['dependencies']!=expected_deps:raise ValueError('Derived parents changed')
            run.verify_hashes(r['origin_artifacts'])


def audit_tables(c,fp,frame):
    out=Path(c['output_dir']);h=run.history(c);refs,singles,means=independent_declarations();names=refs+list(singles)+list(means)
    if set(h)!=set(names):raise ValueError('Selectable strategy union differs')
    cf=pd.read_csv(out/'crossfit.csv');policies=['pooled','latest','rank_fraction']
    check_keys(cf,['candidate','policy','eval_month'],product(names,policies,[5,6]))
    for row in cf.to_dict('records'):
        n,k,m=row['candidate'],row['policy'],row['eval_month'];rule=independent_rule(h[n],k,m)
        if rule!=json.loads(row['parameters']) or str(row['source_months'])!=','.join(map(str,rule['source_months'])):raise ValueError('Earlier-only policy differs')
        check_values(row,independent_metrics(h[n][m],independent_mask(h[n][m],rule)))
    summary=pd.read_csv(out/'summary.csv');index=summary.set_index(['candidate','policy'])
    check_keys(summary,['candidate','policy'],product(names,policies))
    for (n,k),g in cf.groupby(['candidate','policy']):
        check_values(index.loc[(n,k)],{v:float(g[v].mean()) for v in ['f1','f2','f3','precision','recall','flagged_fraction','average_precision','roc_auc']})
    ordered=summary.sort_values(['f2','recall','precision','candidate','policy'],ascending=[False,False,False,True,True])
    if list(ordered.index)!=list(range(len(summary))):raise ValueError('Actual macro winner/order differs')
    contrast=pd.read_csv(out/'contrasts.csv');check_keys(contrast,['contrast','candidate','policy','reference','reference_policy'],expected_factors())
    for row in contrast.to_dict('records'):
        a,b=index.loc[(row['candidate'],row['policy'])],index.loc[(row['reference'],row['reference_policy'])]
        check_values(row,{v+'_difference':a[v]-b[v] for v in ['f2','recall','precision','flagged_fraction','average_precision']})
    pooled=pd.read_csv(out/'pooled.csv');check_keys(pooled,['candidate','policy'],product(names,policies))
    for row in pooled.to_dict('records'):
        g=cf[(cf.candidate==row['candidate'])&(cf.policy==row['policy'])];tn,fp_,fn,tp=[int(g[v].sum()) for v in ['tn','fp','fn','tp']]
        vals=dict(tn=tn,fp=fp_,fn=fn,tp=tp,precision=tp/(tp+fp_) if tp+fp_ else 0.,recall=tp/(tp+fn) if tp+fn else 0.,flagged_fraction=(tp+fp_)/g.n.sum())
        for beta in [1,2]:
            den=(1+beta**2)*tp+beta**2*fn+fp_;vals['f'+str(beta)]=(1+beta**2)*tp/den if den else 0.
        check_values(row,vals)
    scores=pd.read_csv(out/'score_diagnostics.csv');check_keys(scores,['candidate','eval_month'],product(names,[4,5,6]))
    for row in scores.to_dict('records'):
        p=h[row['candidate']][row['eval_month']]['scores'];check_values(row,dict(n=len(p),score_min=p.min(),score_max=p.max(),score_mean=p.mean(),unique_scores=len(np.unique(p)),tied_fraction=1-len(np.unique(p))/len(p)))
    base_names=set(run.REFERENCE_BASES)|set(singles);clips=pd.read_csv(out/'clipping.csv');check_keys(clips,['candidate','eval_month'],product(base_names,[4,5,6]))
    for row in clips.to_dict('records'):
        p=run.prediction(c,'validation',row['eval_month'],row['candidate'])['scores'];logit(p)
        check_values(row,dict(n=len(p),epsilon=1e-6,input_min=p.min(),input_max=p.max(),below_epsilon=int((p<1e-6).sum()),above_one_minus_epsilon=int((p>1-1e-6).sum()),changed_count=int(((p<1e-6)|(p>1-1e-6)).sum())))
    div=pd.read_csv(out/'seed_diversity.csv');pairs=list(combinations([17,29,43],2))
    check_keys(div,['weighting','first_seed','second_seed','eval_month'],[(w,s,t,m) for w,(s,t),m in product(['balanced','unweighted'],pairs,[4,5,6])])
    for row in div.to_dict('records'):
        w,s,t,m=[row[k] for k in ['weighting','first_seed','second_seed','eval_month']];a,b=h[f'bagged_{w}_s{s}'][m]['scores'],h[f'bagged_{w}_s{t}'][m]['scores']
        constant=np.ptp(a)==0 or np.ptp(b)==0
        if row['correlation_reason']!=('constant member' if constant else 'defined'):raise ValueError('Correlation reason differs')
        if constant:
            if not pd.isna(row['pearson']) or not pd.isna(row['spearman']):raise ValueError('Undefined correlation fabricated')
        else:check_values(row,dict(pearson=np.corrcoef(a,b)[0,1],spearman=np.corrcoef(rankdata(a),rankdata(b))[0,1]))
        check_values(row,dict(mean_absolute_difference=np.abs(a-b).mean(),unequal_scores=int((a!=b).sum())))
    joint=pd.read_csv(out/'decision_overlap.csv')
    check_keys(joint,['weighting','first_seed','second_seed','eval_month','policy'],[(w,s,t,m,k) for w,(s,t),m,k in product(['balanced','unweighted'],pairs,[5,6],policies)])
    for row in joint.to_dict('records'):
        w,s,t,m,k=[row[x] for x in ['weighting','first_seed','second_seed','eval_month','policy']]
        first,second=f'bagged_{w}_s{s}',f'bagged_{w}_s{t}';a,b=h[first][m],h[second][m]
        pa=independent_mask(a,independent_rule(h[first],k,m));pb=independent_mask(b,independent_rule(h[second],k,m))
        masks=dict(both=pa&pb,first_only=pa&~pb,second_only=~pa&pb,neither=~pa&~pb)
        vals={n:int(v.sum()) for n,v in masks.items()};vals.update({label+'_'+n:int(((a['labels']==y)&v).sum()) for y,label in [(1,'fraud'),(0,'legitimate')] for n,v in masks.items()})
        check_values(row,vals)
    models=pd.read_csv(out/'model_diagnostics.csv');check_keys(models,['candidate','eval_month'],product(singles,[4,5,6]))
    pre=read_config(out/'preprocessing.json');refmeta=read_config(out/'reference_diagnostics.json')
    if set(pre)!={f'{m}:{n}' for n,m in product(singles,[4,5,6])} or set(refmeta)!={f'{m}:{n}' for n,m in product(run.REFERENCE_BASES,[4,5,6])}:raise ValueError('Preprocessing/reference key set differs')
    for row in models.to_dict('records'):
        n,m=row['candidate'],row['eval_month'];r=read_config(run.paths(c,'validation',m,n)[0]);w,s=singles[n]
        if pre[f'{m}:{n}']!=r['preprocessing']:raise ValueError('Exported train-only preprocessing changed')
        for key in ['class_weights','warnings','requested_model_params','booster_params']:
            if json.loads(row[key])!=r[key]:raise ValueError('Exported model parameter evidence changed')
        check_values(row,dict(weighting=w,bagging_seed=s,**{k:r[k] for k in ['fitted_tree_count','fit_seconds','peak_process_rss_bytes','train_rows','train_fraud']},transformed_dimension=r['preprocessing']['transformed_dimension']))
    for m,n in product([4,5,6],run.REFERENCE_BASES):
        if refmeta[f'{m}:{n}']!=read_config(run.paths(c,'validation',m,n)[0]):raise ValueError('Exported reference audit evidence changed')
    audit_compositions(c,fp,[4,5,6],'validation',names)
    dag=run.dependencies(names);wanted={f'validation-m{m}-{n}' for n,m in product(dag['scores'],[4,5,6])}
    for folder,suffix in [('runs','.json'),('predictions','.npz')]:
        if {p.stem for p in (out/folder).glob('validation-*'+suffix)}!=wanted:raise ValueError('Physical artifact key set differs')
    for m,n in product([4,5,6],dag['scores']):run.aligned(run.prediction(c,'validation',m,n),run.expected_population(frame,m))
    return dict(scores=54,selectable_scores=45,operating_rows=90,macro_rows=45,contrasts=342,diversity_rows=18,overlap_rows=36,clipping_rows=33,new_model_rows=18,reference_results=15,winner=str(summary.iloc[0].candidate),policy=str(summary.iloc[0].policy),development_f2=float(summary.iloc[0].f2))


def audit_development(config_path=run.CONFIG,*,write=True):
    c,fp=run.initialize(config_path);out=Path(c['output_dir'])
    if write and (out/'selection.json').exists():raise ValueError('Frozen independent audit cannot be overwritten')
    done=read_config(out/'validation_complete.json')
    if done['fingerprint']!=fp:raise ValueError('Development fingerprint differs')
    run.verify_hashes({str(out/p):h for p,h in done['files'].items()})
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    records=run.prepare(c,fp,frame,base,allow_missing=False)
    if len(records)!=33:raise ValueError('All33 saved bases required')
    preparation=read_config(out/'preparation.json')
    if preparation['new_bagged_fits']!=18 or preparation['reference_reuses']!=sum(bool(r.get('reused')) for r in records if r['candidate'] in run.REFERENCE_BASES):raise ValueError('Preparation work accounting differs')
    result=audit_tables(c,fp,frame)
    from .unweighted_fusion_report import audit_reference_values
    audit_reference_values(pd.read_csv(out/'summary.csv').set_index(['candidate','policy']))
    run.guard(c)
    if write:write_json(out/'independent_development_audit.json',dict(fingerprint=fp,validation_sha256=digest(out/'validation_complete.json'),saved_model_checks=len(records),independent_preprocessing_inference_and_parameters=True,independent_policies_metrics_factors=True,elapsed_seconds=time.perf_counter()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,created_at_utc=run.now(),**result))
    return records


def operating_figure(summary,dest):
    fig,axes=plt.subplots(1,3,figsize=(17,5));index=summary.set_index(['candidate','policy'])
    for k,marker in [('pooled','o'),('latest','^'),('rank_fraction','s')]:
        g=summary[summary.policy==k];axes[0].scatter(g.recall,g.precision,label=k,marker=marker,s=25)
    axes[0].set(xlabel='Recall',ylabel='Precision',title='All45 strategy/policy points');axes[0].legend()
    forms=['s17','s29','s43','prob_mean','logit_mean'];x=np.arange(5)
    for ax,k in zip(axes[1:],['pooled','latest']):
        for w,color in [('balanced','tab:blue'),('unweighted','tab:orange')]:
            values=[index.loc[(f'bagged_{w}_{f}',k),'f2'] for f in forms]
            ax.plot(x,values,'o-',color=color,label=w)
            n='baseline_balanced' if w=='balanced' else 'baseline_lgbm'
            ax.axhline(index.loc[(n,k),'f2'],color=color,linestyle=':',alpha=.7)
        ax.set_xticks(x,['seed17','seed29','seed43','prob.mean','logit.mean'],rotation=20)
        ax.set(ylabel='Macro development F2',title=k+' policy; dotted=unbagged');ax.legend()
    fig.tight_layout();fig.savefig(dest/'bagged_operating.png',dpi=160);plt.close(fig)


def report(config_path=run.CONFIG):
    c,s=run.load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'report_hashes.json').exists():raise ValueError('Frozen report cannot be overwritten')
    records=audit_development(config_path,write=False);frame=load_data(c);base=features(frame);names={n for n,k in s['choices'].values()}
    final_records=run.prepare(c,s['fingerprint'],frame,base,'exploratory',names,False);audit_compositions(c,s['fingerprint'],[7],'exploratory',names)
    access=read_config(out/'exploratory_access.json');done=read_config(out/'transfer_complete.json');sha=digest(out/'selection.json')
    if access['selection_sha256']!=sha or done['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc'] or done['table_sha256']!=digest(out/'exploratory.csv'):raise ValueError('Transfer evidence differs')
    data=run.final_data(c,s);h=run.history(c);final=pd.read_csv(out/'exploratory.csv');masks={}
    check_keys(final,['role'],[(a,) for a in s['choices']])
    for a,d in data.items():
        run.aligned(d,data['selected']);n,k=s['choices'][a]
        if s['rules'][a]!=independent_rule(h[n],k,7):raise ValueError('Final policy chronology differs')
        masks[a]=independent_mask(d,s['rules'][a]);row=final[final.role==a].iloc[0]
        if row.candidate!=n or row.policy!=k or json.loads(row.parameters)!=s['rules'][a]:raise ValueError('Final role/rule differs')
        check_values(row,independent_metrics(d,masks[a]))
    if done['base_results']!=len(final_records) or done['new_bagged_fits']!=sum(r['candidate'].startswith('bagged_') for r in final_records):raise ValueError('Transfer accounting differs')
    cf=pd.read_csv(out/'crossfit.csv');summary=pd.read_csv(out/'summary.csv')
    dint,ddraws=development(c,s,cf,h);fint,fdraws=intervals(data,masks,np.random.default_rng(c['bootstrap_seed']),500,'7_consumed')
    audit_intervals(dint,ddraws,fint,fdraws,s['choices'],500);rng=np.random.default_rng(c['bootstrap_seed'])
    for m in [5,6]:
        data_m={a:h[n][m] for a,(n,k) in s['choices'].items()};pred_m={a:independent_mask(data_m[a],independent_rule(h[n],k,m)) for a,(n,k) in s['choices'].items()}
        direct_draw_audit(data_m,pred_m,rng,ddraws[ddraws.scope==str(m)].reset_index(drop=True))
    direct_draw_audit(data,masks,np.random.default_rng(c['bootstrap_seed']),fdraws)
    for n,t in [('development_intervals',dint),('development_draws',ddraws),('consumed_intervals',fint),('consumed_draws',fdraws.assign(replicate=np.arange(len(fdraws))))]:t.to_csv(dest/(n+'.csv'),index=False)
    for n in run.TABLES+['validation_complete.json','independent_development_audit.json','selection.json','exploratory_access.json','transfer_complete.json','exploratory.csv','declaration.json','checks.json','preparation.json','resource_preflight.json','source_miniature_gate.json','miniature_gate.json']:
        shutil.copyfile(out/n,dest/n)
    shutil.copyfile(config_path,dest/'config.json');shutil.copyfile(c['protocol_path'],dest/'protocol.md')
    write_json(dest/'run_provenance.json',records);write_json(dest/'transfer_provenance.json',final_records)
    winner=summary.iloc[0];index=summary.set_index(['candidate','policy']);gain={a:float(winner.f2-index.loc[tuple(nk),'f2']) for a,nk in run.REFERENCE_ROLES.items()};fg={r.reference:float(r.difference) for r in fint.itertuples()}
    measured=all(gain[a]>0 and fg[a]>0 for a in gain)
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,development_gain=gain,consumed_gain=fg,measured_gain_on_both_scopes=measured,
               direct_draws_checked=1500,saved_development_models_checked=33,saved_final_models_checked=len(final_records),
               conditional_clear_development_gain=all(float(dint[(dint.scope=='macro')&(dint.reference==a)].iloc[0].ci_low)>0 for a in gain),
               conditional_clear_consumed_gain=all(float(fint[fint.reference==a].iloc[0].ci_low)>0 for a in gain),preservation_checked=True)
    write_json(dest/'audit.json',audit);operating_figure(summary,dest);counts=read_config(out/'preparation.json');tests=read_config(out/'checks.json')
    text=f'''# Bagged LightGBM seed findings

Selected **{winner.candidate} / {winner.policy}**, macro chronological development F2 **{winner.f2:.6f}**. Positive measured differences against all five references on both scopes: **{measured}**. Finite experiment completion does not itself establish the broader performance goal.

## Design and actual work

Original 200-round/15-leaf LightGBM preprocessing and tree settings are retained, with uniform row fraction0.8/frequency1 and explicit bagging seeds17/29/43. Balanced and unweighted training each contribute three individual scores, an equal probability mean and an equal clipped-logit mean. The five original references remain fixed. Sampling is without replacement within an iteration; bags are not exactly class-matched. Balanced weights use whole eligible training class counts. Row sampling changes source models; this is not a renamed deterministic blend. Mean logits remain finite margins; probability-like scores are not asserted calibrated.

Development made {counts['new_bagged_fits']} new full bagged fits, {counts['reference_reuses']} reference reuses and {counts['reference_reproduction_fits']} exact reference reproductions. The33 bases and21 derived results give54 physical/45 selectable scores,90 forward rows and342 complete matched contrasts. The nine-fit synthetic gate checked active/inactive sampling, designed-fixture diversity, same-seed repeatability and serialization. The full suite passed {tests['tests_passed']} tests, with two current-study tiny cache-format fits per invocation counted separately from the gate and historical test modules. Actual-host predictor-only allocation and preserved source-miniature gates passed.

Models/preprocessing use only earlier months; month5 rules use4, month6 rules use4/5 or latest5. Actual macro positive-class F2 beta2 selects the winner, without a quota or precision floor; AP, pooled F2 and diagnostics do not select alternatives.

## Complete development results

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

## Matched effects

All five-reference, own-policy, randomized-versus-unbagged, ensemble-versus-each-member, aggregation and matched-weighting factors retain ties and negative effects. Seed-score diversity has18 rows, decision overlap36 and base clipping33. New training/preprocessing/parameter/resource evidence and original reference numerical evidence accompany every source. These finite seed perturbations do not estimate uncertainty over all training seeds.

{markdown(pd.read_csv(out/'contrasts.csv').query('candidate == @winner.candidate'))}

## Conditional development uncertainty

{markdown(dint)}

## Already-consumed month7

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

Exactly {len(s['choices'])} frozen roles transferred using {done['new_bagged_fits']} new bagged fits, {done['reference_reuses']} reference reuses, {done['reproduced_additive_fits']} additive reproductions and {done['reproduced_tree_fits']} tree reproductions. Only their deduplicated frozen DAG was evaluated; losing-grid settings and hindsight cutoff/winner replacement were excluded.

{markdown(fint)}

## Limitations and reproduction

All1500 paired fixed-decision draws and quantiles were independently reconstructed using seed1027. Intervals condition on fixed models/policies and exclude training-seed/retraining, selection, repeated adaptive search, temporal/generative uncertainty and multiplicity correction. A positive conditional lower bound is required for clear-gain language on the named scope. The adaptive source choice, three seeds, fraction and aggregation forms are analyst-selected. Month7 was already consumed and cannot provide independent confirmation. Synthetic data and eight monthly periods limit generalization.

All fifteen predecessor identities, separate Newton histories and the stopped attempt remain preserved. README gives pinned environment/data identity, fifteen selection restores and ordered reproduction into an isolated destination. Do not overwrite frozen test/audit/selection/report evidence. Codex assisted with declaration, implementation, execution, independent audits, figures and interpretation using preserved repository interfaces and pinned APIs, including the official [LightGBM parameter definitions](https://lightgbm.readthedocs.io/en/stable/Parameters.html). No public notebook was copied; the group must acknowledge assistance and understand submitted work.

![Operating points and matched seed/aggregation results](bagged_operating.png)
'''
    (dest/'findings.md').write_text(text);run.guard(c)
    write_json(dest/'report_hashes.json',{str(p):digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='report_hashes.json'})
    print('Report complete; measured goal acceptance:',measured,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit_development','report']);parser.add_argument('--config',default=run.CONFIG)
    args=parser.parse_args();globals()[args.command](args.config)
