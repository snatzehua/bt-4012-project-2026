"""Analysis-only repair preserving every original Newton fit fingerprint.

The original verifier compared sorted DataFrame indices to a reset CSV index.
This entry point fixes that comparison and declares its own complete analysis
identity; it delegates all fitting to the unchanged original numerical recipe.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from . import newton_spline_study as fits
from .common import read_config,write_json,digest,load_data,features
from .newton_spline_study import (CONFIG,FUSION,REFERENCE_IDS,AdditiveInputs,now,
    fingerprint,model_path,paths,reference_bases,fit_new,audit_new,audit_grid,
    history,crossfit,export_tables,export_numerical,numerical_tables,verify_table,
    order_summary,choices_for,learn,final_data,aligned,measure,contrast_factors)

TABLES=fits.TABLES+['analysis_declaration.json']
ANALYSIS_SOURCES=fits.SOURCES+['baf/newton_spline_audit.py','baf/newton_spline_audit_report.py']


def analysis_fingerprint(config_path=CONFIG):
    identity=fingerprint(config_path)
    identity['sources']={p:digest(p) for p in ANALYSIS_SOURCES}
    return identity


def guard(c):
    fits.guard(c)
    p=Path(c['output_dir'])/'analysis_declaration.json'
    if p.exists():
        r=read_config(p)
        if r['analysis_fingerprint']!=analysis_fingerprint(r['config_path']):
            raise ValueError('Analysis repair identity changed')
        if any(not Path(p).is_file() or digest(p)!=h for p,h in r['original_artifacts'].items()):
            raise ValueError('Original Newton fit artifact changed during analysis repair')


def initialize_analysis(config_path):
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);p=out/'analysis_declaration.json'
    if not p.exists():
        existing=[r for r in (out/'runs').glob('validation-*.json') if read_config(r)['spec'].get('representation')]
        write_json(p,dict(created_at_utc=now(),config_path=str(config_path),analysis_fingerprint=analysis_fingerprint(config_path),
            original_fit_fingerprint=fingerprint(config_path),preexisting_new_fit_count=len(existing),
            reason='Verification index reset only; no numerical recipe, policy or model changes',
            original_artifacts={str(p):digest(p) for folder in ['runs','models','predictions'] for p in (out/folder).glob('*') if p.is_file()}))
    guard(c)


def miniature(config_path=CONFIG):
    initialize_analysis(config_path);fits.miniature(config_path)


def load_selection(config_path=CONFIG):
    c,s=fits.load_selection(config_path);guard(c)
    if s['analysis_fingerprint']!=analysis_fingerprint(config_path):raise ValueError('Frozen analysis identity changed')
    return c,s



def audit_development(c,identity,frame,base):
    records=audit_grid(c,identity,frame,base)
    from .spline_report import audit_tables
    cf=crossfit(c);verify_table(cf,Path(c['output_dir'])/'crossfit.csv')
    verify_table(order_summary(cf).reset_index(drop=True),Path(c['output_dir'])/'summary.csv');audit_tables(c,cf,history(c))
    numbers=pd.read_csv(Path(c['output_dir'])/'numerical.csv')
    expected=[]
    for m in c['validation_months']:
        for spec in c['candidates']:
            r=read_config(paths(c,'validation',m,spec['id'])[0]);expected.append(dict(candidate=spec['id'],eval_month=m,**r['numerical']))
    verify_table(pd.DataFrame(expected),Path(c['output_dir'])/'numerical.csv')
    if len(numbers)!=36 or len(read_config(Path(c['output_dir'])/'feature_blocks.json'))!=36:raise ValueError('Incomplete numerical/features evidence')
    _,comparisons=numerical_tables(c,frame,base)
    verify_table(comparisons,Path(c['output_dir'])/'old_solver_comparisons.csv')
    blocks=read_config(Path(c['output_dir'])/'feature_blocks.json');coeff=[];diagnostics=[]
    for m in c['validation_months']:
        for spec in c['candidates']:
            a=joblib.load(model_path(c,'validation',m,spec['id']));inp=a['inputs'];model=a['model']
            if blocks[f"m{m}-{spec['id']}"]!=inp.metadata():raise ValueError('Feature block export differs')
            for population,ids in [('training',frame.index[frame.month<m]),('prediction',frame.index[frame.month==m])]:
                diagnostics.extend(dict(candidate=spec['id'],eval_month=m,population=population,**d) for d in inp.diagnostics(base.loc[ids]))
            coeff.extend(dict(candidate=spec['id'],eval_month=m,feature=n,coefficient=float(v)) for n,v in zip(inp.names,model.coef_[0]))
            coeff.append(dict(candidate=spec['id'],eval_month=m,feature='__INTERCEPT__',coefficient=float(model.intercept_[0])))
    verify_table(pd.DataFrame(coeff),Path(c['output_dir'])/'coefficients.csv')
    verify_table(pd.DataFrame(diagnostics),Path(c['output_dir'])/'feature_diagnostics.csv')
    guard(c);return records

def validate(config_path=CONFIG):
    initialize_analysis(config_path)
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Frozen study cannot be retuned')
    preflight=read_config(out/'miniature.json')
    if preflight['fingerprint']!=identity or preflight['fit_count']!=24 or len(preflight['results'])!=12:
        raise ValueError('Missing or stale miniature gate')
    if digest(out/'preservation.json')!=preflight['preservation_sha256']:raise ValueError('Preservation presence snapshot changed')
    out.mkdir(parents=True,exist_ok=True);frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
    refs=reference_bases(c,identity,frame,base);new=[]
    for m in c['validation_months']:
        train=frame.index[frame.month<m]
        for rep in ['linear','spline']:
            inp=AdditiveInputs(rep).fit(base.loc[train]);matrix=inp.transform(base.loc[train])
            for s in c['candidates']:
                if s['representation']==rep:
                    rp,pp=paths(c,'validation',m,s['id']);mp=model_path(c,'validation',m,s['id'])
                    if fits.old.cache_state([rp,pp,mp]):new.append(read_config(rp))
                    else:new.append(fit_new(c,identity,frame,base,s,m,inputs=inp,matrix=matrix))
    cf=crossfit(c);summary=export_tables(c,cf,frame,base);export_numerical(c,frame,base)
    audit_development(c,identity,frame,base)
    for name,value in [(FUSION,.3289065701775199),('baseline_balanced',.32780157809618937),('baseline_lgbm',.3226536849212962)]:
        if not np.isclose(summary[(summary.candidate==name)&(summary.policy=='pooled')].iloc[0].f2,value,atol=1e-12,rtol=0):
            raise ValueError('Preserved pooled reference differs')
    write_json(out/'validation_complete.json',dict(fingerprint=identity,analysis_fingerprint=analysis_fingerprint(config_path),created_at_utc=now(),
        files={p:digest(out/p) for p in TABLES+['miniature.json']},new_logistic_fits=len(new),
        reference_reuses=sum(r.get('reused',False) for r in refs),reference_results=len(refs),
        reference_fits=sum(not r.get('reused',False) for r in refs),forward_rows=90,
        selectable_scores=45,all_score_artifacts=48,miniature_fits=24))
    guard(c);print(summary.head(12).to_string(index=False),flush=True)

def freeze(config_path=CONFIG):
    initialize_analysis(config_path)
    c=read_config(config_path);guard(c);identity=fingerprint(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    complete=read_config(out/'validation_complete.json')
    if complete['fingerprint']!=identity or any(digest(out/p)!=h for p,h in complete['files'].items()):
        raise ValueError('Stale development evidence')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);audit_development(c,identity,frame,base)
    cf=crossfit(c);verify_table(cf,out/'crossfit.csv');summary=order_summary(cf).reset_index(drop=True)
    verify_table(summary,out/'summary.csv');winner=summary.iloc[0];choices=choices_for(c,str(winner.candidate),str(winner.policy));h=history(c)
    write_json(out/'selection.json',dict(created_at_utc=now(),fingerprint=identity,analysis_fingerprint=analysis_fingerprint(config_path),config=c,
        selected_candidate=str(winner.candidate),selected_policy=str(winner.policy),selected_development_f2=float(winner.f2),
        choices=choices,rules={a:learn(h[n],k,7) for a,(n,k) in choices.items()},
        validation_files=complete['files'],artifacts={str(p):digest(p) for folder in ['runs','predictions','models']
            for p in (out/folder).glob('*') if p.is_file()},month7_previously_consumed=True,exploratory=True))
    print('Frozen',winner.candidate,winner.policy,float(winner.f2),flush=True)

def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);ap=out/'exploratory_access.json';sha=digest(out/'selection.json')
    if not ap.exists():write_json(ap,dict(access_at_utc=now(),selection_sha256=sha,month7_previously_consumed=True))
    access=read_config(ap)
    if access['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Freeze/access chronology differs')
    frame=load_data(c);base=features(frame);refs=reference_bases(c,s['fingerprint'],frame,base,'exploratory');new=[]
    names={n for n,k in s['choices'].values()}
    for spec in c['candidates']:
        if spec['id'] in names:new.append(fit_new(c,s['fingerprint'],frame,base,spec,7,'exploratory'))
    data=final_data(c,s);rows=[]
    for a,d in data.items():
        aligned(d,data['balanced']);rule=s['rules'][a]
        rows.append(dict(role=a,candidate=s['choices'][a][0],policy=s['choices'][a][1],eval_month=7,
            parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    write_json(out/'transfer_complete.json',dict(selection_sha256=sha,table_sha256=digest(out/'exploratory.csv'),
        new_logistic_fits=len(new),reference_results=len(refs),reference_reuses=sum(r.get('reused',False) for r in refs),
        reference_fits=sum(not r.get('reused',False) for r in refs),role_count=len(rows)))
    guard(c);print(table[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']].to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['miniature','validate','freeze','exploratory']);parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args();globals()[args.command](args.config)
