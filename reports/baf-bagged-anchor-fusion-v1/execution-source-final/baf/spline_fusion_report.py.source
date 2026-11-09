"""Independent reconstruction and reporting for fixed spline score fusion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import (confusion_matrix,fbeta_score,precision_score,recall_score,
                             average_precision_score,roc_auc_score)

from .common import digest,features,load_data,read_config,write_json
from .recall_weighted_metrics import decisions,learn
from .direct_ranking_report import development,intervals
from .score_fusion_report import audit_intervals
from .spline_report import direct_draw_audit,markdown
from .spline_fusion_scores import FUSION,REFERENCES,MEMBERS,EPSILON,ALPHAS
from . import spline_fusion_study as run


def independent_rule(h,kind,evaluation):
    months=sorted(m for m in h if m<evaluation)
    if not months:raise ValueError('No earlier predictions')
    if kind=='latest':months=months[-1:]
    y=np.concatenate([h[m]['labels'] for m in months]);s=np.concatenate([h[m]['scores'] for m in months]);ids=np.concatenate([h[m]['row_ids'] for m in months])
    order=np.lexsort((ids,-s));tp=np.cumsum(y[order]);counts=np.arange(1,len(y)+1)
    values=5*tp/(4*y.sum()+counts)
    if kind=='rank_fraction':
        index=int(np.argmax(values)) if y.sum() else -1
        return dict(kind=kind,numerator=index+1,denominator=len(y),beta=2,source_months=months)
    ends=np.flatnonzero(np.r_[s[order][1:]!=s[order][:-1],True])
    index=int(ends[np.argmax(values[ends])])
    threshold=float(s[order][index]) if y.sum() else float(np.nextafter(s.max(),np.inf))
    return dict(kind=kind,threshold=threshold,beta=2,source_months=months)


def independent_mask(d,rule):
    if rule['kind']!='rank_fraction':return d['scores']>=rule['threshold']
    n=len(d['labels']);num,den=rule['numerator'],rule['denominator'];k=-(-num*n//den)
    p=np.zeros(n,dtype=bool);p[np.lexsort((d['row_ids'],-d['scores']))[:k]]=True;return p


def independent_metrics(d,p):
    y=d['labels'];tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),precision=precision_score(y,p,zero_division=0),
        recall=recall_score(y,p,zero_division=0),f1=fbeta_score(y,p,beta=1,zero_division=0),
        f2=fbeta_score(y,p,beta=2,zero_division=0),f3=fbeta_score(y,p,beta=3,zero_division=0),
        flagged_fraction=float(p.mean()),n=len(y),fraud=int(y.sum()),average_precision=average_precision_score(y,d['scores']),
        roc_auc=roc_auc_score(y,d['scores']))


def close(values,expected):
    if any(not np.isclose(values[k],v,atol=1e-12,rtol=0,equal_nan=True) for k,v in expected.items()):
        raise ValueError('Independent numerical reconstruction differs')


def audit_development(c,fp,frame,base,check_models=True):
    out=Path(c['output_dir']);records=run.prepare(c,fp,frame,base,allow_missing=False) if check_models else []
    h=run.history(c);cf=pd.read_csv(out/'crossfit.csv');masks={}
    expected={(n,k,m) for n in h for k in c['policies'] for m in [5,6]}
    actual=set()
    for r in cf.to_dict('records'):
        n,k,m=r['candidate'],r['policy'],r['eval_month'];actual.add((n,k,m))
        rule=independent_rule(h[n],k,m)
        if json.loads(r['parameters'])!=rule:raise ValueError('Independent earlier-only rule differs')
        mask=independent_mask(h[n][m],rule);masks[n,k,m]=mask
        close(r,independent_metrics(h[n][m],mask))
    if actual!=expected or len(cf)!=len(expected):raise ValueError('Operating key set incomplete')
    generated=run.tables(c,run.crossfit(c))
    for n,t in generated.items():run.verify_table(t,out/(n+'.csv'))
    # Reconstruct compositions directly from original member values, not compose().
    for m in c['validation_months']:
        xgb=run.prediction(c,'validation',m,'xgb_d4_w1')['scores']
        bal=h['baseline_balanced'][m]['scores']
        def logits(p):
            if ((p<0)|(p>1)).any() or not np.isfinite(p).all():raise ValueError('Base probability domain differs')
            p=np.maximum(EPSILON,np.minimum(1-EPSILON,p));return np.log(p)-np.log(1-p)
        g=.5*logits(xgb)+.5*logits(bal)
        if not np.allclose(g,h[FUSION][m]['scores'],atol=1e-12,rtol=0):raise ValueError('Reference fusion arithmetic differs')
        for n in MEMBERS:
            p=h[n][m]['scores'];u=logits(p)
            if not np.allclose(u,h['logit_'+n][m]['scores'],atol=1e-12,rtol=0):raise ValueError('Newton logit differs')
            for a in ALPHAS:
                actual=h[f'margin_blend_{n}_a{int(a*100)}'][m]['scores']
                if not np.allclose(actual,a*u+(1-a)*g,atol=1e-12,rtol=0):raise ValueError('Common-margin arithmetic differs')
            row=generated['clipping'].query('candidate == @n and eval_month == @m').iloc[0]
            close(row,dict(n=len(p),below_epsilon=int((p<EPSILON).sum()),above_one_minus_epsilon=int((p>1-EPSILON).sum()),
                           changed_count=int(((p<EPSILON)|(p>1-EPSILON)).sum()),input_min=p.min(),input_max=p.max(),epsilon=EPSILON))
            div=generated['diversity'][(generated['diversity'].candidate=='logit_'+n)&(generated['diversity'].eval_month==m)].iloc[0]
            if not div.correlation_undefined:close(div,dict(pearson=pearsonr(u,g).statistic,spearman=spearmanr(u,g).statistic))
    for r in generated['joint_errors'].to_dict('records'):
        n,k,m=r['candidate'],r['policy'],r['eval_month'];a=masks[n,k,m];b=masks[FUSION,k,m];y=h[n][m]['labels'].astype(bool)
        for tag,mask in [('fraud',y),('legitimate',~y)]:
            pairs=np.stack((a[mask],b[mask]),axis=1);counts=np.bincount(pairs[:,0].astype(int)*2+pairs[:,1],minlength=4)
            close(r,{tag+'_neither':counts[0],tag+'_fusion_only':counts[1],tag+'_newton_only':counts[2],tag+'_both_flagged':counts[3]})
    summary=generated['summary'].set_index(['candidate','policy'])
    for n,k in summary.index:
        sub=cf[(cf.candidate==n)&(cf.policy==k)]
        close(summary.loc[(n,k)],{metric:sub[metric].mean() for metric in ['f2','precision','recall','flagged_fraction','average_precision']})
        merged={key:np.concatenate([h[n][m][key] for m in [5,6]]) for key in ['labels','scores','row_ids']}
        pooled_mask=np.concatenate([masks[n,k,m] for m in [5,6]])
        independent=independent_metrics(merged,pooled_mask)
        pooled=generated['pooled'][(generated['pooled'].candidate==n)&(generated['pooled'].policy==k)].iloc[0]
        close(pooled,{key:independent[key] for key in ['tn','fp','fn','tp','precision','recall','f1','f2','flagged_fraction']})
    factors=run.contrast_factors();contrasts=generated['contrasts']
    if list(contrasts[['contrast','candidate','policy','reference','reference_policy']].itertuples(index=False,name=None))!=factors:raise ValueError('Contrast factors differ')
    for r in contrasts.to_dict('records'):
        a,b=summary.loc[(r['candidate'],r['policy'])],summary.loc[(r['reference'],r['reference_policy'])]
        close(r,{key+'_difference':a[key]-b[key] for key in ['f2','precision','recall','flagged_fraction','average_precision']})
    expected_counts=dict(crossfit=138,summary=69,pooled=69,contrasts=433,score_diagnostics=69,clipping=12,diversity=12,joint_errors=24)
    if any(len(generated[n])!=size for n,size in expected_counts.items()):raise ValueError('Table cardinality differs')
    for folder,suffix in [('predictions','.npz'),('runs','.json')]:
        files={p.stem for p in (out/folder).glob('validation-*'+suffix)}
        wanted={f'validation-m{m}-{n}' for m in [4,5,6] for n in list(h)+['xgb_d4_w1']}
        if files!=wanted:raise ValueError('Score artifact set differs')
    for n,value in zip(REFERENCES,[.3289065701775199,.32780157809618937,.3226536849212962]):
        close(summary.loc[(n,'pooled')],dict(f2=value))
    run.guard(c);return records


def audit(config_path=run.CONFIG):
    c=read_config(config_path);run.guard(c);fp=run.fingerprint(config_path);frame=load_data(c);frame=frame[frame.month<7]
    records=audit_development(c,fp,frame,features(frame));out=Path(c['output_dir'])
    done=read_config(out/'validation_complete.json')
    if done['fingerprint']!=fp:raise ValueError('Validation identity changed')
    run.verify_hashes({str(out/p):v for p,v in done['files'].items()})
    write_json(out/'independent_development_audit.json',dict(fingerprint=fp,validation_sha256=digest(out/'validation_complete.json'),
        saved_model_checks=len(records),scores_checked=72,operating_rows=138,contrasts=433,clipping_rows=12,
        diversity_rows=12,joint_error_rows=24,independent_rules_and_metrics=True,independent_compositions=True,
        created_at_utc=run.now()))
    print('Independent development audit passed: 21 models, 72 scores, 138 decisions, 433 contrasts',flush=True)


def report(config_path=run.CONFIG):
    c,s=run.load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    frame=load_data(c);base=features(frame);records=audit_development(c,s['fingerprint'],frame[frame.month<7],base.loc[frame.index[frame.month<7]])
    names={n for n,k in s['choices'].values()};final_records=run.prepare(c,s['fingerprint'],frame,base,'exploratory',names,False)
    access=read_config(out/'exploratory_access.json');done=read_config(out/'transfer_complete.json');sha=digest(out/'selection.json')
    if access['selection_sha256']!=sha or done['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Transfer chronology differs')
    if done['table_sha256']!=digest(out/'exploratory.csv'):raise ValueError('Transfer table differs')
    data=run.final_data(c,s);final=pd.read_csv(out/'exploratory.csv');masks={}
    for a,d in data.items():
        rule=s['rules'][a];n,k=s['choices'][a]
        if rule!=independent_rule(run.history(c)[n],k,7):raise ValueError('Final rule differs')
        masks[a]=independent_mask(d,rule);close(final[final.role==a].iloc[0],independent_metrics(d,masks[a]))
    h=run.history(c);cf=pd.read_csv(out/'crossfit.csv');summary=pd.read_csv(out/'summary.csv')
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
    for n in run.TABLES+['validation_complete.json','independent_development_audit.json','selection.json','exploratory_access.json','transfer_complete.json','exploratory.csv']:
        shutil.copyfile(out/n,dest/n)
    shutil.copyfile(config_path,dest/'config.json');shutil.copyfile(c['protocol_path'],dest/'protocol.md')
    write_json(dest/'run_provenance.json',records);write_json(dest/'transfer_provenance.json',final_records)
    winner=summary.iloc[0];lookup=summary.set_index(['candidate','policy']);gain={r:float(winner.f2-lookup.loc[(n,'pooled'),'f2']) for r,n in zip(['fusion','balanced','unweighted'],REFERENCES)}
    fg={r.reference:float(r.difference) for r in fint.itertuples()};measured=all(gain[n]>0 and fg[n]>0 for n in gain)
    write_json(dest/'audit.json',dict(fingerprint=s['fingerprint'],selection_sha256=sha,development_gain=gain,consumed_gain=fg,
        measured_gain_on_both_scopes=measured,direct_draws_checked=1500,saved_development_models_checked=len(records),
        conditional_clear_development_gain=all(float(dint[(dint.scope=='macro')&(dint.reference==n)].iloc[0].ci_low)>0 for n in gain),
        conditional_clear_consumed_gain=all(float(fint[fint.reference==n].iloc[0].ci_low)>0 for n in gain),preservation_checked=True))
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for kind,label in [('raw','Newton raw'),('logit','Newton logit'),('reference','Reference'),('margin_blend','Blend')]:
        ids=[x['id'] for x in c['strategies'] if x['kind']==kind];t=summary[summary.candidate.isin(ids)]
        axes[0].scatter(t.recall,t.precision,label=label,s=25,alpha=.75)
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 69 strategy/policy results');axes[0].legend(fontsize=8)
    for member in MEMBERS:
        values=[float(lookup.loc[(f'margin_blend_{member}_a{int(a*100)}','pooled'),'f2'])-float(lookup.loc[(FUSION,'pooled'),'f2']) for a in ALPHAS]
        axes[1].plot(ALPHAS,values,marker='o',label=member.replace('_c0p1',''))
    axes[1].axhline(0,color='grey',linewidth=.8);axes[1].set(xlabel='Newton weight',ylabel='Macro F2 minus fusion / pooled',title='Fixed composition, common pooled policy');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'fusion_operating.png',dpi=160);plt.close(fig)
    comp=pd.read_csv(out/'contrasts.csv');bestblend=summary[summary.candidate.str.startswith('margin_blend_')].iloc[0]
    counts=read_config(out/'validation_complete.json')
    text=f'''# Spline-score fusion findings

Selected **{winner.candidate} / {winner.policy}**, macro forward F2 **{winner.f2:.6f}**. Best blend: **{bestblend.candidate} / {bestblend.policy}**, F2 **{bestblend.f2:.6f}**. Positive measured F2 differences against fusion, balanced and unweighted on both development and consumed transfer: **{measured}**. Search completion alone does not achieve the broader performance goal.

Four frozen C=0.1 Newton models (linear/spline crossed with unweighted/balanced training) were combined with the existing fixed tree fusion at Newton weights 0.1/0.25/0.5. Only Newton probability-like scores received a clipped logit, epsilon 1e-6; the existing fusion is already a margin. Composition is fixed row-wise and uses neither labels nor batch statistics. Scores are not asserted calibrated probabilities. C/source choices are adaptive to earlier exploratory results.

Development used **{counts['base_reuses']} verified model/score reuses**, **{counts['new_fits']} new fits**, 51 derived results and zero miniature model fits. Every saved base model reproduced inference; Newton representation, independent objective/gradient and fallback evidence were checked under the frozen recipes. There are 72 score artifacts, 69 selectable scores, 138 forward operating rows, 69 macro and pooled summaries, 433 contrasts, 12 clipping/diversity rows each and 24 joint-error rows. All eleven completed identities and the stopped attempt remain preserved.

Month-5 policies learn on month 4; month-6 policies on months 4/5 or latest month 5. F2 has beta=2, without a review quota or precision floor. Selection uses actual macro F2, then recall, precision and deterministic IDs. Pooled F2, AP, correlations and joint errors are descriptive.

## Forward results

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

## Interpretation and matched effects

Original versus logit singles separate the clipping/transformation effect from adding a second member. Every blend is compared with both common-margin members at the same policy; all representation, training-weight and blend-weight contrasts retain negatives and ties. Non-pooled versus own pooled comparisons isolate policy changes. The operating figure's right panel holds the policy type fixed at pooled. Complete numeric effects are in contrasts.csv; score correlation and joint-error overlap are descriptive and did not change weights or selection.

{markdown(comp[comp.candidate==bestblend.candidate][['contrast','policy','reference','reference_policy','f2_difference']])}

## Conditional development differences

{markdown(dint)}

## Already-consumed month 7

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

Only the frozen winner-type roles transferred with deduplicated work: {done['base_reuses']} base reuses and {done['new_fits']} new fits, including {done['new_newton_fits']} Newton fits. Freeze/access hashes and exact dependencies are exported. No losing-grid sweep or hindsight threshold was evaluated.

{markdown(fint)}

Uncertainty uses 500 paired fixed-decision draws per month, seed 1027; all 1,000 development and 500 consumed draws were independently reconstructed by direct sampled Boolean rows. Intervals condition on fixed models/policies and exclude adaptive source/C choice, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity adjustment. Month 7 was consumed before this study and cannot supply independent confirmation. Analyst-selected settings remain open to group review.

## Reproduction and assistance

See README's ordered commands and protocol.md. Configuration, original fit/analysis ancestry, package/Python/data hashes, model/prediction parents, preservation presence, frozen rules, transfer counts and checked tables are exported. Codex assisted with the study design, implementation, numerical/provenance audits, diagnostics, reporting and tests, reusing preserved repository interfaces and pinned public-library APIs. The group must explain and verify any submitted analysis.
'''
    (dest/'findings.md').write_text(text)
    run.guard(c);write_json(dest/'report_hashes.json',{str(p):digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='report_hashes.json'})
    print('Report complete; measured goal acceptance:',measured,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit','report']);parser.add_argument('--config',default=run.CONFIG)
    args=parser.parse_args();globals()[args.command](args.config)
