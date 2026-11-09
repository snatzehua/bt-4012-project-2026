"""Independent feature, coefficient and moment reconstruction for pair surfaces.

The numerical reconstruction deliberately does not call the transformer pair
methods; feature values, output factors, decisions and uncertainty are checked
separately from their producers.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.special import expit

from .interaction_inputs import PAIRS, MAX_CHUNK_ROWS
from .spline_inputs import AdditiveInputs, NUMERIC
from .common import CATEGORICAL, read_config, write_json, digest, load_data, features
from . import interaction_study as run
from .spline_fusion_report import independent_rule, independent_mask, independent_metrics
from .spline_report import direct_draw_audit, markdown
from .direct_ranking_report import development, intervals
from .score_fusion_report import audit_intervals


def close(actual, expected, *, inference=False):
    if not np.allclose(actual, expected, atol=1e-12, rtol=1e-12 if inference else 1e-10):
        raise ValueError("Independent interaction reconstruction differs")


def marginal(inputs, x, column):
    a = inputs.additive
    v = x[column].fillna(a.medians[column]).to_numpy(dtype=np.float64)
    if column not in a.splines:
        return v[:, None]
    saved = a.splines[column].bsplines_[0]
    block = next(b for b in a.blocks if b['feature'] == column)
    clipped = np.clip(v, block['knots'][0], block['knots'][-1])
    return BSpline(saved.t, saved.c, saved.k)(clipped)[:, :-1]


def main_matrix(inputs, x):
    """Dense reconstruction of ONE bounded chunk, including original scaling."""
    a = inputs.additive
    numeric = np.column_stack([*(marginal(inputs, x, c) for c in NUMERIC),
                                x[NUMERIC].isna().to_numpy(dtype=float)])
    numeric /= a.scaler.scale_
    categories = x[CATEGORICAL].fillna('__MISSING__').astype(str)
    cat = np.column_stack([(categories[c].to_numpy() == v).astype(float)
                           for c, levels in zip(CATEGORICAL, a.encoder.categories_) for v in levels])
    return np.column_stack([numeric, cat])


def pair_matrix(inputs, x, b):
    u = marginal(inputs, x, b['left']) - np.asarray(b['left_mean'])
    v = marginal(inputs, x, b['right']) - np.asarray(b['right_mean'])
    # Explicit index loops provide an independent check of the reshape order.
    t = np.column_stack([u[:, i] * v[:, j]
                         for i in range(u.shape[1]) for j in range(v.shape[1])])
    z = (t - np.asarray(b['mean'])) / np.asarray(b['scale'])
    z[:, np.asarray(b['inactive'], dtype=bool)] = 0.
    return z


def reconstructed_matrix(inputs, x):
    if len(x) > MAX_CHUNK_ROWS:
        raise ValueError("Independent dense reconstruction must be chunked")
    return np.column_stack([main_matrix(inputs, x),
                            *(pair_matrix(inputs, x, b) for b in inputs.pair_blocks)])


def audit_statistics(inputs, train):
    """Two-pass long-double moments; precision is reported for the actual host."""
    original = AdditiveInputs('spline').fit(train)
    if original.metadata() != inputs.additive.metadata():
        raise ValueError("Main representation is not the unchanged training-only additive fit")
    if set(original.splines) != set(inputs.additive.splines):
        raise ValueError("Saved marginal transformer set changed")
    for col, transformer in original.splines.items():
        expected = transformer.bsplines_[0]
        actual = inputs.additive.splines[col].bsplines_[0]
        if (expected.k != actual.k or not np.array_equal(expected.t, actual.t)
                or not np.array_equal(expected.c, actual.c)):
            raise ValueError("Saved marginal spline coefficients/knots changed")
    if [(b['left'], b['right']) for b in inputs.pair_blocks] != list(PAIRS):
        raise ValueError("Pair identity/order changed")
    names = list(original.names)
    for b in inputs.pair_blocks:
        ld, rd = b['left_dimension'], b['right_dimension']
        totals = [np.zeros(ld, dtype=np.longdouble), np.zeros(rd, dtype=np.longdouble)]
        mins = [np.full(ld, np.inf), np.full(rd, np.inf)]
        maxs = [np.full(ld, -np.inf), np.full(rd, -np.inf)]
        for start in range(0, len(train), MAX_CHUNK_ROWS):
            part = train.iloc[start:start + MAX_CHUNK_ROWS]
            for i, col in enumerate([b['left'], b['right']]):
                v = marginal(inputs, part, col)
                if v.shape[1] != len(totals[i]):
                    raise ValueError("Marginal dimension changed")
                totals[i] += v.sum(axis=0, dtype=np.longdouble)
                mins[i] = np.minimum(mins[i], v.min(axis=0))
                maxs[i] = np.maximum(maxs[i], v.max(axis=0))
        means = [v / len(train) for v in totals]
        close(b['left_mean'], means[0]); close(b['right_mean'], means[1])
        for i, tag in enumerate(['left', 'right']):
            close(b[tag + '_minimum'], mins[i]); close(b[tag + '_maximum'], maxs[i])
        degenerate = bool(np.all(mins[0] == maxs[0]) or np.all(mins[1] == maxs[1]))
        if b['marginal_degenerate'] != degenerate or b['training_rows'] != len(train):
            raise ValueError("Training degeneracy/population changed")
        # Products use the saved marginal means: they define the fitted map.
        def products(part):
            u = marginal(inputs, part, b['left']) - np.asarray(b['left_mean'])
            v = marginal(inputs, part, b['right']) - np.asarray(b['right_mean'])
            return np.column_stack([u[:, i] * v[:, j] for i in range(ld) for j in range(rd)])
        total = np.zeros(ld * rd, dtype=np.longdouble)
        lo, hi = np.full(ld * rd, np.inf), np.full(ld * rd, -np.inf)
        for start in range(0, len(train), MAX_CHUNK_ROWS):
            t = products(train.iloc[start:start + MAX_CHUNK_ROWS])
            total += t.sum(axis=0, dtype=np.longdouble)
            lo = np.minimum(lo, t.min(axis=0)); hi = np.maximum(hi, t.max(axis=0))
        mean = total / len(train)
        m2 = np.zeros(ld * rd, dtype=np.longdouble)
        for start in range(0, len(train), MAX_CHUNK_ROWS):
            t = products(train.iloc[start:start + MAX_CHUNK_ROWS]).astype(np.longdouble)
            m2 += ((t - mean) ** 2).sum(axis=0)
        variance = m2 / len(train)
        variance[lo == hi] = 0.
        inactive = (lo == hi) | degenerate
        scale = np.sqrt(variance); scale[inactive] = 1.
        close(b['mean'], mean); close(b['variance'], variance); close(b['scale'], scale)
        close(b['minimum'], lo); close(b['maximum'], hi)
        if not np.array_equal(b['inactive'], inactive):
            raise ValueError("Inactive tensor columns changed")
        if b['start'] != len(names) or b['stop'] != len(names) + ld * rd:
            raise ValueError("Pair offsets changed")
        names += [f"pair__{b['left']}__{b['right']}__basis_{i}_{j}"
                  for i in range(ld) for j in range(rd)]
    if inputs.names != names or inputs.main_dimension != len(original.names):
        raise ValueError("Interaction feature names/order changed")
    return dict(training_rows=len(train), pairs=len(PAIRS), dimension=len(names),
                statistic_method='two-pass long-double accumulation; direct SciPy marginal bases',
                accumulator_bits=int(np.finfo(np.longdouble).bits))


def audit_inference(inputs, model, x, expected_scores=None):
    if list(model.classes_) != [0, 1] or model.n_features_in_ != len(inputs.names):
        raise ValueError("Independent class/dimension check failed")
    predictions = []
    for start in range(0, len(x), MAX_CHUNK_ROWS):
        part = x.iloc[start:start + MAX_CHUNK_ROWS]
        direct = reconstructed_matrix(inputs, part)
        actual = inputs.transform(part)
        close(actual.toarray(), direct, inference=True)
        scores = expit(direct @ model.coef_[0] + model.intercept_[0])
        close(model.predict_proba(actual)[:, 1], scores, inference=True)
        predictions.append(scores)
    result = np.concatenate(predictions)
    if expected_scores is not None:
        close(result, expected_scores, inference=True)
    return result


def independent_objective(model, inputs, x, labels, spec):
    """Chunked coefficient reconstruction; independently accumulated gradient."""
    y = np.asarray(labels)
    if set(np.unique(y)) != {0, 1} or len(y) != len(x):
        raise ValueError("Invalid independent objective population")
    counts = np.bincount(y.astype(int), minlength=2)
    coefficient = model.coef_[0]
    penalty = 1 / spec['C']
    loss = np.longdouble(0); weight_sum = np.longdouble(0)
    gradient = np.zeros(len(coefficient) + 1, dtype=np.longdouble)
    for start in range(0, len(x), MAX_CHUNK_ROWS):
        part = x.iloc[start:start + MAX_CHUNK_ROWS]
        z = reconstructed_matrix(inputs, part)
        target = y[start:start + len(part)]
        w = (len(y) / (2 * counts[target.astype(int)]) if spec['class_weight'] == 'balanced'
             else np.ones(len(part)))
        margin = z @ coefficient + model.intercept_[0]
        residual = w * (expit(margin) - target)
        loss += np.sum(w * (np.logaddexp(0, margin) - target * margin), dtype=np.longdouble)
        weight_sum += w.sum(dtype=np.longdouble)
        gradient[:-1] += z.T @ residual
        gradient[-1] += residual.sum(dtype=np.longdouble)
    value = (loss + penalty * np.dot(coefficient, coefficient) / 2) / weight_sum
    gradient[:-1] += penalty * coefficient
    gradient /= weight_sum
    return dict(objective=float(value), gradient=gradient.astype(float).tolist(),
                gradient_max_abs=float(np.max(np.abs(gradient))), weight_sum=float(weight_sum))


def expected_factors(c):
    """Independent Cartesian-product enumeration, not the producer's factor list."""
    result=[]
    names=[s['id'] for s in c['strategies']]
    refs=[(run.RECENT_FUSION,'latest'),(run.FUSION,'pooled'),
          ('baseline_balanced','pooled'),('baseline_lgbm','pooled')]
    for n in names:
        for k in ['pooled','latest','rank_fraction']:
            for ref,rk in refs:result.append(('original operating reference',n,k,ref,rk))
            if k!='pooled':result.append(('policy minus own pooled',n,k,n,'pooled'))
    specs=c['candidates']
    def name(rep,C,w):
        return next(s['id'] for s in specs if s['representation']==rep and s['C']==C and s['class_weight']==w)
    for C,w,k in itertools.product([.1,1.],[None,'balanced'],['pooled','latest','rank_fraction']):
        result.append(('interaction minus additive',name('interaction',C,w),k,name('spline',C,w),k))
    for rep,C,k in itertools.product(['spline','interaction'],[.1,1.],['pooled','latest','rank_fraction']):
        result.append(('balanced minus unweighted',name(rep,C,'balanced'),k,name(rep,C,None),k))
    for rep,w,k in itertools.product(['spline','interaction'],[None,'balanced'],['pooled','latest','rank_fraction']):
        result.append(('larger C minus smaller C',name(rep,1.,w),k,name(rep,.1,w),k))
    return result


def independent_pair_diagnostics(inp,x,b):
    total=np.zeros(b['stop']-b['start'],dtype=np.longdouble);square=np.zeros_like(total)
    lo=np.inf;hi=-np.inf;nnz=0
    for start in range(0,len(x),MAX_CHUNK_ROWS):
        z=pair_matrix(inp,x.iloc[start:start+MAX_CHUNK_ROWS],b)
        total+=z.sum(axis=0,dtype=np.longdouble)
        square+=(z.astype(np.longdouble)**2).sum(axis=0)
        lo=min(lo,z.min());hi=max(hi,z.max());nnz+=np.count_nonzero(z)
    return dict(n=len(x),dimension=len(total),inactive_columns=sum(b['inactive']),
        marginal_degenerate=b['marginal_degenerate'],nnz=int(nnz),max_abs_mean=float(np.max(np.abs(total/len(x)))),
        mean_square=float(square.sum()/(len(x)*len(total))),minimum=float(lo),maximum=float(hi))


def check_values(row,expected):
    for k,v in expected.items():
        if isinstance(v,(str,bool)):
            if row[k]!=v:raise ValueError('Independent field differs: '+k)
        elif not np.isclose(row[k],v,atol=1e-12,rtol=1e-10):
            raise ValueError('Independent value differs: '+k)


def audit_reference_values(lookup):
    for (n,k),value in zip([(run.RECENT_FUSION,'latest'),(run.FUSION,'pooled'),
            ('baseline_balanced','pooled'),('baseline_lgbm','pooled')],
            [.3290099447918403,.3289065701775199,.32780157809618937,.3226536849212962]):
        close([lookup.loc[(n,k),'f2']],[value])


def audit_tables(c,frame,base):
    out=Path(c['output_dir']);h=run.history(c);cf=pd.read_csv(out/'crossfit.csv')
    expected={(n,k,m) for n in h for k in ['pooled','latest','rank_fraction'] for m in [5,6]}
    if len(cf)!=72 or set(cf[['candidate','policy','eval_month']].itertuples(index=False,name=None))!=expected:
        raise ValueError('Forward key set differs')
    for r in cf.to_dict('records'):
        n,k,m=r['candidate'],r['policy'],r['eval_month'];rule=independent_rule(h[n],k,m)
        if rule!=json.loads(r['parameters']):raise ValueError('Independent earlier-only policy differs')
        check_values(r,independent_metrics(h[n][m],independent_mask(h[n][m],rule)))
    summary=pd.read_csv(out/'summary.csv');lookup=summary.set_index(['candidate','policy'])
    if len(summary)!=36 or set(lookup.index)!={(n,k) for n in h for k in c['policies']}:
        raise ValueError('Macro key set differs')
    for (n,k),g in cf.groupby(['candidate','policy']):
        check_values(lookup.loc[(n,k)],{v:float(g[v].mean()) for v in ['f2','f1','f3','recall','precision',
            'flagged_fraction','average_precision','roc_auc'] if v in lookup.columns})
    independent_order=summary.sort_values(['f2','recall','precision','candidate','policy'],
        ascending=[False,False,False,True,True]).reset_index(drop=True)
    run.verify_table(independent_order,out/'summary.csv')
    contrasts=pd.read_csv(out/'contrasts.csv')
    factors=list(contrasts[['contrast','candidate','policy','reference','reference_policy']].itertuples(index=False,name=None))
    if len(contrasts)!=204 or factors!=expected_factors(c):raise ValueError('Independent contrast factors differ')
    for r in contrasts.to_dict('records'):
        a,b=lookup.loc[(r['candidate'],r['policy'])],lookup.loc[(r['reference'],r['reference_policy'])]
        check_values(r,{v+'_difference':a[v]-b[v] for v in ['f2','recall','precision','flagged_fraction','average_precision']})
    pooled=pd.read_csv(out/'pooled.csv')
    if len(pooled)!=36 or set(pooled[['candidate','policy']].itertuples(index=False,name=None))!=set(lookup.index):
        raise ValueError('Pooled key set differs')
    for r in pooled.to_dict('records'):
        g=cf[(cf.candidate==r['candidate'])&(cf.policy==r['policy'])];tn,fp,fn,tp=[int(g[k].sum()) for k in ['tn','fp','fn','tp']]
        check_values(r,dict(tn=tn,fp=fp,fn=fn,tp=tp,precision=tp/(tp+fp) if tp+fp else 0.,
            recall=tp/(tp+fn) if tp+fn else 0.,f2=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,
            flagged_fraction=(tp+fp)/g.n.sum()))
    score=pd.read_csv(out/'score_diagnostics.csv')
    if len(score)!=36 or set(score[['candidate','eval_month']].itertuples(index=False,name=None))!={(n,m) for n in h for m in [4,5,6]}:
        raise ValueError('Score diagnostic key set differs')
    for r in score.to_dict('records'):
        p=h[r['candidate']][r['eval_month']]['scores']
        check_values(r,dict(n=len(p),score_min=p.min(),score_max=p.max(),score_mean=p.mean(),
                           unique_scores=len(np.unique(p)),tied_fraction=1-len(np.unique(p))/len(p)))
    for m in [4,5,6]:
        def logit(p):
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Reference probability domain changed')
            p=np.maximum(1e-6,np.minimum(1-1e-6,p));return np.log(p)-np.log(1-p)
        xgb=run.prediction(c,'validation',m,'xgb_d4_w1')['scores']
        tree_margin=.5*logit(xgb)+.5*logit(h['baseline_balanced'][m]['scores'])
        close(h[run.FUSION][m]['scores'],tree_margin,inference=True)
        recent=.25*logit(h['spline_c0p1_unweighted'][m]['scores'])+.75*tree_margin
        close(h[run.RECENT_FUSION][m]['scores'],recent,inference=True)
    wanted={f'validation-m{m}-{n}' for m in [4,5,6] for n in list(h)+['xgb_d4_w1']}
    for folder,suffix in [('runs','.json'),('predictions','.npz')]:
        if {p.stem for p in (out/folder).glob('validation-*'+suffix)}!=wanted:raise ValueError('Exact artifact set differs')
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
    audit_reference_values(lookup)
    return dict(scores=39,selectable_scores=36,operating_rows=72,macro_rows=36,contrasts=204,
                feature_rows=len(fd),pair_rows=len(pdg),coefficient_rows=len(coefficients),
                winner=str(summary.iloc[0].candidate),policy=str(summary.iloc[0].policy),
                development_f2=float(summary.iloc[0].f2))


def audit_development(config_path=run.CONFIG):
    c=read_config(config_path);run.guard(c);fp=run.fingerprint(config_path);out=Path(c['output_dir'])
    done=read_config(out/'validation_complete.json')
    if done['fingerprint']!=fp:raise ValueError('Development fingerprint differs')
    run.verify_hashes({str(out/p):h for p,h in done['files'].items()})
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
    records=run.prepare(c,fp,frame,base,allow_missing=False)
    actual=dict(base_results=len(records),base_reuses=sum(bool(r.get('reused')) for r in records),
        new_interaction_fits=sum(r['spec'].get('representation')=='interaction' for r in records),
        reproduced_base_fits=sum(not r.get('reused',False) and r['spec'].get('representation')!='interaction' for r in records),
        recent_derived_reuses=sum(read_config(run.paths(c,'validation',m,run.RECENT_FUSION)[0])['reused'] for m in [4,5,6]),
        total_scores=len(list((out/'predictions').glob('validation-*.npz'))),selectable_scores=36,
        derived_results=6,forward_rows=72,miniature_fits=8)
    if any(done[k]!=v for k,v in actual.items()):raise ValueError('Actual fit/reuse/output counts differ')
    if any(r.get('reused') for r in records if r['spec'].get('representation')=='interaction'):
        raise ValueError('New interaction incorrectly marked reused')
    result=audit_tables(c,frame,base);run.guard(c)
    write_json(out/'independent_development_audit.json',dict(fingerprint=fp,
        validation_sha256=digest(out/'validation_complete.json'),saved_model_checks=len(records),
        independent_statistics_inference_and_gradient=True,independent_policies_metrics_factors=True,
        independent_moment_accumulator_bits=int(np.finfo(np.longdouble).bits),
        created_at_utc=run.now(),**result))
    print('Independent development audit passed',result,flush=True)
    return records


def surfaces(c,s,summary,dest):
    name=s['selected_candidate']
    if not name.startswith('interaction_'):
        name=str(summary[summary.candidate.str.startswith('interaction_')].iloc[0].candidate)
        reason='best development interaction; diagnostic only; plotting does not authorize transfer'
    else:reason='selected interaction; month-6 coefficients'
    a=joblib.load(run.model_path(c,'validation',6,name));inp,model=a['inputs'],a['model']
    fixed=pd.DataFrame([inp.additive.reference_values],columns=inp.columns);rows=[]
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for ax,b in zip(axes,inp.pair_blocks):
        left,right=b['left'],b['right'];lv=np.linspace(*inp.additive.training_quantiles[left],21)
        rv=np.linspace(*inp.additive.training_quantiles[right],21)
        x=pd.concat([fixed]*441,ignore_index=True);x[left]=np.repeat(lv,21);x[right]=np.tile(rv,21)
        coef=model.coef_[0,b['start']:b['stop']]
        reference=float((pair_matrix(inp,fixed,b)@coef)[0])
        direct=pair_matrix(inp,x,b)@coef-reference
        actual=inp.pair_matrix(x,b)@coef-float((inp.pair_matrix(fixed,b)@coef)[0])
        close(actual,direct,inference=True)
        rows += [dict(candidate=name,eval_month=6,left=left,right=right,left_index=i,right_index=j,
            left_value=float(lv[i]),right_value=float(rv[j]),pair_log_odds_difference=float(direct[i*21+j]),
            left_reference=float(fixed[left].iloc[0]),right_reference=float(fixed[right].iloc[0]),diagnostic=reason)
            for i,j in itertools.product(range(21),range(21))]
        grid=direct.reshape(21,21);limit=max(float(np.max(np.abs(grid))),1e-12)
        im=ax.imshow(grid,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        positions=[0,5,10,15,20];ax.set_xticks(positions,[f'{rv[i]:.3g}' for i in positions])
        ax.set_yticks(positions,[f'{lv[i]:.3g}' for i in positions])
        ax.set(xlabel=right.replace('_',' '),ylabel=left.replace('_',' '),title='Pair block relative to reference')
        fig.colorbar(im,ax=ax,label='Pair-block log-odds change',shrink=.8)
    table=pd.DataFrame(rows)
    if len(table)!=1323:raise ValueError('Pair surface row count differs')
    table.to_csv(dest/'pair_surfaces.csv',index=False);fig.savefig(dest/'pair_surfaces.png',dpi=160);plt.close(fig)
    return dict(candidate=name,rows=len(rows),diagnostic=reason)


def report(config_path=run.CONFIG):
    c,s=run.load_selection(config_path);out=Path(c['output_dir']);dest=Path(c['report_dir']);dest.mkdir(parents=True,exist_ok=True)
    test_counts=read_config(dest/'test_fit_counts.json')
    if (test_counts['latest']['fingerprint']!=s['fingerprint']
            or test_counts['latest']['test_source_sha256']!=digest('tests/test_spline_interactions.py')):
        raise ValueError('Methodological test receipt does not cover the frozen implementation')
    done=read_config(out/'validation_complete.json');ind=read_config(out/'independent_development_audit.json')
    if ind['validation_sha256']!=digest(out/'validation_complete.json'):raise ValueError('Independent audit stale')
    sha=digest(out/'selection.json');access=read_config(out/'exploratory_access.json');transfer=read_config(out/'transfer_complete.json')
    if access['selection_sha256']!=sha or transfer['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:
        raise ValueError('Freeze/access chronology differs')
    if transfer['table_sha256']!=digest(out/'exploratory.csv'):raise ValueError('Transfer table changed')
    frame=load_data(c);base=features(frame);dev=frame[frame.month<7]
    audit_tables(c,dev,base.loc[dev.index]);h=run.history(c);cf=pd.read_csv(out/'crossfit.csv');summary=pd.read_csv(out/'summary.csv')
    records=[read_config(run.paths(c,'validation',m,spec['id'])[0]) for m in [4,5,6] for spec in c['candidates']+c['references']+c['presets']]
    final_records=run.prepare(c,s['fingerprint'],frame,base,'exploratory',{n for n,k in s['choices'].values()},False)
    if {r['candidate'] for r in final_records}!={spec['id'] for spec in s['fit_specs']}:
        raise ValueError('Transfer fitted losing settings or omitted a dependency')
    data=run.final_data(c,s);masks={};final=pd.read_csv(out/'exploratory.csv')
    if len(final)!=len(s['choices']) or set(final.role)!=set(s['choices']):raise ValueError('Transfer role set differs')
    for a,(n,k) in s['choices'].items():
        rule=independent_rule(h[n],k,7)
        if s['rules'][a]!=rule:raise ValueError('Independent final earlier-only rule differs')
        masks[a]=independent_mask(data[a],rule)
        row=final[final.role==a].iloc[0]
        if row.candidate!=n or row.policy!=k or json.loads(row.parameters)!=rule:raise ValueError('Final role recipe differs')
        check_values(row,independent_metrics(data[a],masks[a]))
    dint,ddraws=development(c,s,cf,h)
    fint,fdraws=intervals(data,masks,np.random.default_rng(c['bootstrap_seed']),c['bootstrap_replicates'],'7_consumed')
    audit_intervals(dint,ddraws,fint,fdraws,s['choices'],c['bootstrap_replicates'])
    rng=np.random.default_rng(c['bootstrap_seed'])
    for m in [5,6]:
        ds={a:h[n][m] for a,(n,k) in s['choices'].items()}
        ms={a:independent_mask(ds[a],independent_rule(h[n],k,m)) for a,(n,k) in s['choices'].items()}
        draws=ddraws[ddraws.scope.astype(str)==str(m)].reset_index(drop=True)
        direct_draw_audit(ds,ms,rng,draws)
    direct_draw_audit(data,masks,np.random.default_rng(c['bootstrap_seed']),fdraws)
    for n,t in [('development_intervals',dint),('development_draws',ddraws),('consumed_intervals',fint),
                ('consumed_draws',fdraws.assign(replicate=np.arange(len(fdraws))))]:t.to_csv(dest/(n+'.csv'),index=False)
    for p in run.TABLES+['reference_preparation.json','validation_complete.json','independent_development_audit.json',
        'selection.json','exploratory_access.json','transfer_complete.json','exploratory.csv']:
        shutil.copyfile(out/p,dest/p)
    shutil.copyfile(config_path,dest/'config.json');shutil.copyfile(c['protocol_path'],dest/'protocol.md')
    write_json(dest/'run_provenance.json',records);write_json(dest/'transfer_provenance.json',final_records)
    surface=surfaces(c,s,summary,dest);contrasts=pd.read_csv(out/'contrasts.csv')
    matched=contrasts[contrasts.contrast.isin(['interaction minus additive','balanced minus unweighted','larger C minus smaller C'])]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    for prefix,label in [('interaction_','interaction'),('spline_','additive')]:
        g=summary[summary.candidate.str.startswith(prefix)];axes[0].scatter(g.recall,g.precision,label=label,s=28)
    refs=summary[~summary.candidate.str.startswith(('interaction_','spline_'))]
    axes[0].scatter(refs.recall,refs.precision,label='fixed references',marker='x',s=40)
    axes[0].set(xlabel='Macro forward recall',ylabel='Macro forward precision',title='All 36 strategy/policy results');axes[0].legend()
    g=matched[matched.contrast=='interaction minus additive'].reset_index(drop=True)
    labels=[f"C={next(v['C'] for v in c['candidates'] if v['id']==r.candidate):g} "
            f"{'B' if r.candidate.endswith('balanced') and not r.candidate.endswith('unweighted') else 'U'} / {r.policy}" for r in g.itertuples()]
    axes[1].barh(np.arange(len(g)),g.f2_difference,color=np.where(g.f2_difference>0,'#267f6d','#aa4e54'))
    axes[1].set_yticks(np.arange(len(g)),labels,fontsize=8);axes[1].axvline(0,color='black',linewidth=.8)
    axes[1].set(xlabel='Macro F2 difference',title='Interaction minus matched additive');axes[1].invert_yaxis()
    fig.savefig(dest/'interaction_operating.png',dpi=160);plt.close(fig)
    required=['recent_fusion','fusion','balanced','unweighted'];dmacro=dint[dint.scope=='macro'].set_index('reference');fi=fint.set_index('reference')
    dg={a:float(dmacro.loc[a,'difference']) for a in required};fg={a:float(fi.loc[a,'difference']) for a in required}
    achieved=all(dg[a]>0 and fg[a]>0 for a in required)
    audit=dict(fingerprint=s['fingerprint'],selection_sha256=sha,measured_gain_on_both_scopes=achieved,
        development_gain=dg,consumed_gain=fg,conditional_clear_development_gain=all(dmacro.loc[a,'ci_low']>0 for a in required),
        conditional_clear_consumed_gain=all(fi.loc[a,'ci_low']>0 for a in required),independent_draws_checked=1500,
        development_interval_rows=len(dint),consumed_interval_rows=len(fint),pair_surface=surface,
        development_counts={k:v for k,v in done.items() if k not in ['files','fingerprint']},transfer_counts=transfer,
        original_reference_policies=run.ORIGINAL_POLICIES,broader_goal_achieved=achieved,
        independent_moment_accumulator_bits=int(np.finfo(np.longdouble).bits))
    audit['new_cache_test_reproduction_fits']=test_counts['new_cache_reproduction_fits']
    # NumPy scalar comparisons must become ordinary JSON booleans.
    audit['conditional_clear_development_gain']=bool(audit['conditional_clear_development_gain'])
    audit['conditional_clear_consumed_gain']=bool(audit['conditional_clear_consumed_gain'])
    write_json(dest/'audit.json',audit)
    winner=summary.iloc[0];best=summary[summary.candidate.str.startswith('interaction_')].iloc[0]
    text=f'''# Fixed spline-interaction study findings

Selected **{s['selected_candidate']} / {s['selected_policy']}**, actual macro month-5/6 F2 **{winner.f2:.6f}**. Best interaction: **{best.candidate} / {best.policy}**, F2 **{best.f2:.6f}**. Positive measured differences against all four original references on both development and consumed transfer: **{achieved}**. Finishing this finite search does not itself establish the performance goal.

## Design and actual execution

Three fixed training-only centered/scaled tensor pairs augment the unchanged additive spline representation. C=0.1/1 and unweighted/balanced recipes provide all matched controls. There were {done['new_interaction_fits']} new full interaction development fits, {done['base_reuses']} base reuses, {done['reproduced_base_fits']} exact missing-base reproduction fits, and {done['recent_derived_reuses']} recent-fusion derived reuses. Eight miniature integration fits are separate. The new cache-reproduction tests made {test_counts['new_cache_reproduction_fits']} separately counted tiny estimator fits across their recorded invocations. The latest full methodological run passed {test_counts['latest']['tests_passed']} tests. Development contains 33 bases, six derived references, 39 total score artifacts, 36 selectable scores, 72 forward operating rows and 204 contrasts. Feature-only/analytical fixtures fit no estimators; existing historical test-module fits are also separate from this development count.

Eligible training ends before the evaluation month; month-5 policies use month 4 and month-6 policies use months 4/5 or latest month 5. Selection uses actual macro F2 beta=2, then recall/precision and deterministic IDs. No review quota or precision floor is imposed. The recent selected fusion retains its original latest policy; tree fusion and balanced/unweighted retain pooled policies. Fixed margin composition never treats the existing tree-fusion margin as a probability. Settings and pair choices remain analyst-selected and adaptive to earlier development experiments.

All original and pair feature statistics are training-only and independent of class weighting. Exact constants remain zero at inference, dimensions are retained, and dense construction is limited to one bounded pair chunk. Independent SciPy/two-pass statistics, coefficient inference, normalized objective/full gradient, warnings/iterations and genuine Newton execution were audited. The independent moment accumulator has {np.finfo(np.longdouble).bits} bits on this host; no extra precision over float64 is implied when those widths agree. Solver and resource tables expose actual dimensions, sparsity, storage, Hessian estimates, timings and process RSS. Centering does not prove orthogonality, and equal nominal C does not imply equal effective complexity.

## Complete forward results

{markdown(summary[['candidate','policy','f2','precision','recall','flagged_fraction','average_precision']])}

## Matched effects

{markdown(matched[['contrast','candidate','policy','reference','f2_difference','recall_difference','flagged_fraction_difference']])}

All 204 contrasts, including reference and policy-only differences, negative values and ties, remain in contrasts.csv. AP and pooled confusion metrics supplement the macro objective.

## Conditional development differences

{markdown(dint)}

## Already-consumed month 7: frozen exploratory transfer

{markdown(final[['role','candidate','policy','f2','precision','recall','flagged_fraction','tp','fp','fn']])}

{markdown(fint)}

Only {len(s['choices'])} predeclared frozen roles transferred. Unique dependencies required {transfer['new_interaction_fits']} interaction fits, {transfer['reproduced_additive_fits']} additive reproductions, {transfer['reproduced_tree_fits']} tree reproductions and {transfer['base_reuses']} base reuses. Freeze/access timestamps and hashes agree; duplicate roles retain aliases and share work. No losing-grid sweep, hindsight cutoff or replacement of the development winner was used. Month 7 was already consumed and cannot provide fresh independent confirmation.

## Contribution surfaces and limitations

pair_surfaces.csv contains 1,323 independently reconstructed pair-block contributions for **{surface['candidate']}**, using month-6 coefficients and training reference values over training 1–99% ranges. Each panel isolates one pair block relative to its reference pair; it is neither a total model effect nor a causal effect. A diagnostic losing interaction is not transferred merely because it is plotted.

Intervals use 500 paired fixed-decision row draws per month, seed 1027. All 1,500 draws and quantiles were independently checked. They condition on fixed models/policies and exclude adaptive pair/C/source choice, repeated search, selection, retraining, temporal/generative uncertainty and multiplicity correction. Clear-gain language needs a positive conditional lower bound on the named scope; measured goal acceptance requires positive differences against all four references on both scopes. Synthetic data and eight monthly periods limit generalization.

All twelve historical studies and the stopped attempt remain preserved, including separate original Newton fit/corrected analysis identities. Exact config, provenance, selection/access records, numerical/feature evidence and uncertainty are exported. README documents pinned data/environment, twelve selection restores and ordered execution. Codex assisted with design, implementation, execution and verification; scikit-learn/SciPy supply basis/solver APIs, and no public notebook was copied. The group must acknowledge assistance in the eventual course report and understand the submitted work.

![Operating and matched effects](interaction_operating.png)

![Isolated pair-block contributions](pair_surfaces.png)
'''
    (dest/'findings.md').write_text(text)
    run.guard(c)
    write_json(dest/'report_hashes.json',{str(p):digest(p) for p in dest.iterdir() if p.is_file() and p.name!='report_hashes.json'})
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit','report']);parser.add_argument('--config',default=run.CONFIG)
    args=parser.parse_args();(audit_development if args.command=='audit' else report)(args.config)
