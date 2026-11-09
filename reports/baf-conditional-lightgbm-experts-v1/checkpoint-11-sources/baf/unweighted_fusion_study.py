"""Isolated fixed unweighted-reference additions; prior studies remain immutable."""
from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .common import digest, read_config, write_json, load_data, features
from .focal_study import expected_record, now
from .improvements import paths, prediction
from .recall_extension import order_summary
from .recall_weighted_metrics import POLICIES, aligned, measure, learn, decisions
from .improvement_metrics import confusion_metrics
from .interaction_inputs import InteractionInputs, storage
from . import interaction_fusion_study as parent
from . import interaction_report as source_audit
from . import newton_spline_study as newton
from . import score_fusion_study as tree
from . import spline_fusion_study as additive
from .unweighted_fusion_scores import (LAMBDAS, EPSILON, ANCHORS, ADDED, BASES,
    FUSION, RECENT_FUSION, ALIAS, ORIGINAL_POLICIES, REFERENCE_ROLES,
    recipes, strategies, lookup, compose, choices_for, contrast_factors,
    clipping, diversity, dependencies, leaf_coefficients)

CONFIG = 'configs/unweighted-reference-fusion.json'
SOURCES = parent.SOURCES + ['baf/unweighted_fusion_scores.py',
    'baf/unweighted_fusion_study.py', 'baf/unweighted_fusion_report.py']
COUNTS = dict(strategies=40, base_results=18, derived_results=114, all_scores=132,
    selectable_scores=120, forward_rows=240, macro_rows=120, pooled_rows=120,
    score_rows=120, contrasts=1199, clipping_rows=18, diversity_rows=48,
    logistic_blocks=9, interaction_blocks=3, feature_diagnostic_rows=522,
    pair_diagnostic_rows=18, local_coefficient_rows=1728, local_base_reuses=18,
    local_new_development_fits=0, local_new_final_fits=0, local_derived_reuses=15,
    local_new_derived_computations=99, new_production_miniature_fits=0,
    new_cache_test_fits_per_full_invocation=4, preserved_source_miniature_fits=8,
    development_draws=1000, consumed_draws=500)
TABLES = list(parent.TABLES)
verify_hashes = parent.verify_hashes
cache_paths = parent.cache_paths
model_path = parent.model_path
audit_base = parent.audit_base



def fingerprint(path=CONFIG):
    fp = parent.fingerprint(path)
    fp['sources'] = {p: digest(p) for p in SOURCES}
    return fp

def mandatory_preservation(source):
    result = dict(source['preserve_sha256'])
    for p in parent.SOURCES[-3:] + [parent.CONFIG, source['protocol_path'],
            str(Path(source['output_dir'])/'selection.json'),
            str(Path(source['report_dir'])/'report_hashes.json')]:
        result[p] = digest(p)
    result.update(read_config(Path(source['report_dir'])/'report_hashes.json'))
    return result


def guard(c):
    source=read_config(c['fusion_source_config']); parent.guard(source)
    if c['fusion_source_config'] != parent.CONFIG or parent.fingerprint(parent.CONFIG) != c['fusion_source_fingerprint']:
        raise ValueError('Frozen parent composition identity changed')
    verify_hashes(c['preserve_sha256'])
    if c['preserve_sha256'] != mandatory_preservation(source) or len(c['preserve_sha256']) != 529:
        raise ValueError('Mandatory preservation set changed')
    selections=[p for p in c['preserve_sha256'] if p.startswith('outputs/') and p.endswith('/selection.json')]
    if len(selections)!=14: raise ValueError('Fourteen selection pairs required')
    for p in selections:
        if c['preserve_sha256'].get(p.replace('outputs/','reports/',1)) != c['preserve_sha256'][p]:
            raise ValueError('Original/exported selection pair differs')
    changed={'strategies','alphas','source_paths','preserve_sha256','historical_output_roots',
        'expected_counts','output_dir','report_dir','protocol_path','candidates','references','presets',
        'original_reference_policies'}
    for k,v in source.items():
        if k not in changed and c.get(k)!=v: raise ValueError('Frozen source setting changed: '+k)
    expected=dict(strategies=strategies(),lambdas=LAMBDAS,anchors=ANCHORS,added_members=ADDED,
        source_paths=SOURCES,expected_counts=COUNTS,candidates=recipes(),
        references=[s for s in source['references'] if s['id'] in BASES],
        presets=[s for s in source['presets'] if s['id'] in BASES],
        original_reference_policies=ORIGINAL_POLICIES,reference_roles=REFERENCE_ROLES,
        output_dir='outputs/baf-unweighted-reference-fusion-v1',
        report_dir='reports/baf-unweighted-reference-fusion-v1',
        protocol_path='docs/unweighted-reference-fusion-protocol.md',
        historical_output_roots=sorted(set(source['historical_output_roots']+[source['output_dir']])))
    if any(c.get(k)!=v for k,v in expected.items()) or len(set(SOURCES))!=49:
        raise ValueError('Finite addition declaration changed')
    if any(k in c for k in ['primary_fraction','review_quota','precision_floor']):
        raise ValueError('Undeclared operating constraint')
    dest=[Path(c[k]).resolve() for k in ['output_dir','report_dir']]
    for a,b in itertools.combinations(dest+[Path(p).resolve() for p in c['historical_output_roots']],2):
        if (a in dest or b in dest) and (a==b or a in b.parents or b in a.parents):
            raise ValueError('Destination overlap')
    out=Path(c['output_dir']); initial=out/'preimplementation_preservation.json'
    if initial.exists(): verify_hashes(read_config(initial))
    revisions=out/'implementation_revisions.json'
    if revisions.exists(): verify_hashes(read_config(revisions)['prior_attempt_files'])
    if (out/'preservation.json').exists():
        r=read_config(out/'preservation.json')
        if r['mandatory']!=c['preserve_sha256'] or r['roots']!=c['historical_output_roots']:
            raise ValueError('Preservation declaration changed')
        verify_hashes(r['optional_present'])


def source_origins(c,spec):
    return [(read_config(c['fusion_source_config']),c['fusion_source_fingerprint'])] + parent.source_origins(c,spec)


def initialize(config_path=CONFIG):
    c = read_config(config_path); guard(c)
    out = Path(c['output_dir']); fp = fingerprint(config_path)
    p = out/'preservation.json'
    if not p.exists():
        initial = out/'preimplementation_preservation.json'
        present = read_config(initial) if initial.exists() else {
            str(p): digest(p) for root in c['historical_output_roots'] for p in Path(root).rglob('*') if p.is_file()}
        write_json(p, dict(mandatory=c['preserve_sha256'], optional_present=present,
                          roots=c['historical_output_roots'], created_at_utc=now()))
    declaration = dict(fingerprint=fp, config_path=str(config_path),
        protocol_sha256=digest(c['protocol_path']), preservation_sha256=digest(p),
        strategies=c['strategies'], expected_counts=c['expected_counts'], month7_previously_consumed=True)
    dp = out/'declaration.json'
    if dp.exists():
        if any(read_config(dp).get(k) != v for k, v in declaration.items()):
            raise ValueError('Prospective execution identity changed')
    else:
        write_json(dp, dict(**declaration, created_at_utc=now()))
    return c, fp

def prepare_base(c, fp, frame, base, spec, month, phase='validation', allow_missing=True):
    if spec not in recipes()+c['references']+c['presets']:
        raise ValueError('Undeclared individual recipe')
    own = cache_paths(c, spec, month, phase)
    if tree.cache_state(own):
        return audit_base(c, fp, frame, base, spec, month, phase)
    if not allow_missing:
        raise ValueError('Required individual cache missing')
    failure = Path(c['output_dir'])/'failures'/(own[0].stem+'.json')
    if failure.exists():
        raise ValueError('Prior failed gate retained; no automatic retry')
    for source, identity in source_origins(c, spec):
        origin = cache_paths(source, spec, month, phase)
        if not tree.cache_state(origin):
            continue
        r = audit_base(source, identity, frame, base, spec, month, phase)
        for src, dst in zip(origin[1:], own[1:]):
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
        result = dict(r, fingerprint=fp, reused=True,
                      origin_artifacts={str(p): digest(p) for p in origin})
        if spec.get('representation') == 'spline':
            result.update(original_fit_fingerprint=c['newton_fit_fingerprint'],
                          original_analysis_fingerprint=c['newton_analysis_fingerprint'])
        elif spec['id'] == 'xgb_d4_w1':
            result['model_path'] = str(own[2])
        elif spec.get('representation') != 'interaction':
            result.update(reference_model_path=str(own[2]), reference_model_new_fit=False)
        write_json(own[0], result)
        return audit_base(c, fp, frame, base, spec, month, phase)
    try:
        if spec.get('representation') == 'interaction':
            parent.parent.fit_interaction(c, fp, frame, base, spec, month, phase)
        elif spec.get('representation') == 'spline':
            newton.fit_new(c, fp, frame, base, spec, month, phase)
        elif spec['id'] == 'xgb_d4_w1':
            tree.old.fit_model(c, fp, frame, base, spec, month, phase)
        else:
            additive.reproduce_reference(c, fp, frame, base, spec, month, phase)
        return audit_base(c, fp, frame, base, spec, month, phase)
    except Exception as exc:
        if not failure.exists():
            write_json(failure, dict(**expected_record(frame, spec, month, phase, fp),
                error=repr(exc), numerical=getattr(exc, 'diagnostics', {}), created_at_utc=now()))
        raise

def derived_origins(c,spec):
    if spec['kind']=='anchor':
        return [(read_config(c['fusion_source_config']),c['fusion_source_fingerprint'])]
    if spec['id']==ALIAS:
        return [(read_config(c['source_config']),c['source_fingerprint'])]
    return []


def origin_dependency_hashes(source,spec,month,phase,record):
    expected={};hashes={}
    for member in spec['members']:
        rp,pp=paths(source,phase,month,member)
        hashes.update({str(rp):digest(rp),str(pp):digest(pp)})
        expected[member]=({'record':digest(rp),'prediction':digest(pp)}
            if spec['id'] in [FUSION,ALIAS] else {str(rp):digest(rp),str(pp):digest(pp)})
    if record['dependencies']!=expected:
        raise ValueError('Canonical original dependency schema/hashes differ')
    verify_hashes(hashes)
    return hashes


def derived(c,fp,spec,month,phase='validation',allow_missing=True):
    rp,pp=paths(c,phase,month,spec['id'])
    items={n:prediction(c,phase,month,n) for n in spec['members']}
    d=compose(spec,items)
    deps={n:{str(p):digest(p) for p in paths(c,phase,month,n)} for n in spec['members']}
    expected=dict(candidate=spec['id'],phase=phase,eval_month=month,fingerprint=fp,
        spec=spec,dependencies=deps,epsilon=EPSILON,leaf_coefficients=leaf_coefficients(spec['id']),
        arithmetic=spec.get('arithmetic','fixed multiply then add'),
        domains=['probability-like' if n in BASES else 'finite margin' for n in spec['members']])
    if tree.cache_state([rp,pp]):
        r=read_config(rp);saved=prediction(c,phase,month,spec['id']);aligned(saved,d)
        if (any(r.get(k)!=v for k,v in expected.items()) or digest(pp)!=r['prediction_sha256']
                or not np.array_equal(saved['scores'],d['scores'])):
            raise ValueError('Derived score identity/arithmetic changed')
        verify_hashes(r['origin_artifacts']);return r
    if not allow_missing: raise ValueError('Derived score missing')
    origin={}
    for source,identity in derived_origins(c,spec):
        sr,sp=paths(source,phase,month,spec['id'])
        if not tree.cache_state([sr,sp]):continue
        r=read_config(sr);original=prediction(source,phase,month,spec['id']);aligned(d,original)
        original_spec=(source['fusion_spec'] if spec['id']==FUSION else
            next(s for s in source['strategies'] if s['id']==spec['id']))
        if (r['fingerprint']!=identity or digest(sp)!=r['prediction_sha256']
                or r.get('epsilon')!=EPSILON or not np.array_equal(original['scores'],d['scores'])):
            raise ValueError('Canonical origin identity/arithmetic differs')
        if spec['id'] in [FUSION,ALIAS]:
            if r.get('spec')!=original_spec: raise ValueError('Canonical tree origin recipe changed')
        elif spec['id']==RECENT_FUSION:
            if r['spec']!=original_spec: raise ValueError('Recent anchor origin recipe changed')
        else:
            if r['spec']!=original_spec: raise ValueError('Anchor origin recipe changed')
        if set(r['dependencies'])!=set(spec['members']): raise ValueError('Canonical origin parents differ')
        origin.update(origin_dependency_hashes(source,spec,month,phase,r))
        verify_hashes(r.get('origin_artifacts',{}))
        origin.update({str(sr):digest(sr),str(sp):digest(sp)});break
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**d)
    r=dict(**expected,prediction_sha256=digest(pp),reused=bool(origin),origin_artifacts=origin)
    write_json(rp,r);return r


def prepare(c,fp,frame,base,phase='validation',names=None,allow_missing=True):
    names=[s['id'] for s in strategies()] if names is None else names
    dag=dependencies(names);records=[]
    specs={s['id']:s for s in recipes()+c['references']+c['presets']}
    for m in c['validation_months'] if phase=='validation' else [7]:
        for n in dag['bases']:
            print('Checking',phase,m,n,flush=True)
            records.append(prepare_base(c,fp,frame,base,specs[n],m,phase,allow_missing))
        for s in strategies():
            if s['id'] in dag['derived']:
                derived(c,fp,s,m,phase,allow_missing)
    return records


def history(c):
    return {s['id']: {m: prediction(c, 'validation', m, s['id']) for m in c['validation_months']}
            for s in strategies()}

def crossfit(c):
    h = history(c); rows = []
    for m, n, k in itertools.product([5, 6], h, POLICIES):
        aligned(h[n][m], h[FUSION][m])
        rule = learn({t: d for t, d in h[n].items() if t < m}, k, m)
        rows.append(dict(candidate=n, policy=k, eval_month=m,
            source_months=','.join(map(str, rule['source_months'])), parameters=json.dumps(rule, sort_keys=True),
            **measure(h[n][m], rule)))
    return pd.DataFrame(rows)

def tables(c, cf, frame, base):
    summary = order_summary(cf).reset_index(drop=True); index = summary.set_index(['candidate', 'policy'])
    rows = []
    for label, n, k, r, rk in contrast_factors():
        a, b = index.loc[(n, k)], index.loc[(r, rk)]
        rows.append(dict(contrast=label, candidate=n, policy=k, reference=r, reference_policy=rk,
            **{v+'_difference': float(a[v]-b[v]) for v in ['f2', 'recall', 'precision', 'flagged_fraction', 'average_precision']}))
    pooled = []
    for (n, k), g in cf.groupby(['candidate', 'policy']):
        t = g[['tn', 'fp', 'fn', 'tp']].sum()
        pooled.append(dict(candidate=n, policy=k,
            **confusion_metrics(*[int(t[v]) for v in ['tn', 'fp', 'fn', 'tp']]),
            flagged_fraction=float((t.tp+t.fp)/g.n.sum())))
    h = history(c); scores = []; clips = []; div = []
    for n, items in h.items():
        for m, d in items.items():
            p = d['scores']; scores.append(dict(candidate=n, eval_month=m, n=len(p), score_min=float(p.min()),
                score_max=float(p.max()), score_mean=float(p.mean()), unique_scores=len(np.unique(p)),
                tied_fraction=1-len(np.unique(p))/len(p)))
    for m in c['validation_months']:
        for n in BASES:
            clips.append(dict(candidate=n, eval_month=m, **clipping(prediction(c, 'validation', m, n)['scores'])))
    for tag, added, k, m in itertools.product(ANCHORS, ADDED, POLICIES, [5,6]):
        n=ADDED[added];anchor=ANCHORS[tag];a,b=h['logit_'+n][m],h[anchor][m]
        pa=decisions(a,learn({t:d for t,d in h['logit_'+n].items() if t<m},k,m))
        pb=decisions(b,learn({t:d for t,d in h[anchor].items() if t<m},k,m))
        div.append(dict(anchor=tag,added=added,candidate=n,policy=k,eval_month=m,**diversity(a,b,pa,pb)))
    solver=[]; numerical=[]; fd=[]; paird=[]; coeff=[]; main={}; pairs={}; resources=[]
    for m, spec in itertools.product(c['validation_months'], recipes()):
        n=spec['id'];r=read_config(paths(c,'validation',m,n)[0]);a=joblib.load(model_path(c,'validation',m,n))
        inp,model=a['inputs'],a['model'];key=f'{m}:{n}';reused=bool(r.get('reused'))
        solver.append(dict(candidate=n,eval_month=m,reused=reused,
            **{k:r[k] for k in ['iterations','transformed_dimension','matrix_nnz','matrix_density','fit_seconds','peak_process_rss_bytes']},
            warnings=json.dumps(r['warnings']),solver_params=json.dumps(r['solver_params'],sort_keys=True)))
        numerical.append(dict(candidate=n,eval_month=m,reused=reused,**r['numerical']))
        rr=dict(candidate=n,eval_month=m,reused=reused,measurement_scope='original source fit',
                fit_seconds=r['fit_seconds'],peak_process_rss_bytes=r['peak_process_rss_bytes'])
        if 'resources' in r:rr.update(r['resources'])
        else:
            z=inp.transform(base.loc[frame.index[frame.month<m]]);rr.update(storage(z));del z
            rr['measurement_scope']='original fit time/RSS; current exact matrix storage'
        resources.append(rr)
        main[key]=dict(reused=reused,metadata=inp.additive.metadata() if spec['representation']=='interaction' else inp.metadata())
        if spec['representation']=='interaction':pairs[key]=inp.metadata()
        for population, mask in [('training', frame.month<m), ('evaluation', frame.month==m)]:
            x=base.loc[frame.index[mask]]
            fd += [dict(candidate=n,eval_month=m,population=population,**row) for row in inp.diagnostics(x)]
            if spec['representation']=='interaction':
                paird += [dict(candidate=n,eval_month=m,population=population,**row) for row in inp.pair_diagnostics(x)]
        coeff += [dict(candidate=n,eval_month=m,feature=name,coefficient=float(value))
                  for name,value in zip(inp.names,model.coef_[0])]
        coeff.append(dict(candidate=n,eval_month=m,feature='__intercept__',coefficient=float(model.intercept_[0])))
    result=dict(crossfit=cf,summary=summary,pooled=pd.DataFrame(pooled),contrasts=pd.DataFrame(rows),
        score_diagnostics=pd.DataFrame(scores),clipping=pd.DataFrame(clips),diversity=pd.DataFrame(div),
        solver=pd.DataFrame(solver),numerical=pd.DataFrame(numerical),feature_diagnostics=pd.DataFrame(fd),
        pair_diagnostics=pd.DataFrame(paird),coefficients=pd.DataFrame(coeff),resources=pd.DataFrame(resources))
    return result, main, pairs

def record_checks(config_path=CONFIG):
    c=read_config(config_path);guard(c);dest=Path(c['report_dir']);xml=dest/'tests.xml'
    if (Path(c['output_dir'])/'selection.json').exists():raise ValueError('Frozen check receipt is immutable')
    root=ET.parse(xml).getroot();cases=list(root.iter('testcase'))
    if not cases or list(root.iter('failure')) or list(root.iter('error')) or list(root.iter('skipped')):
        raise ValueError('Full passing pytest evidence required')
    own=[t.attrib.get('name','') for t in cases if t.attrib.get('classname','').endswith('test_unweighted_fusion')]
    if sum(n.startswith('test_four_missing_cache_formats_fit_once[') for n in own)!=4:
        raise ValueError('Four genuine cache-format tests required')
    if sum(n.startswith('test_all_winner_policy_roles[') for n in own)!=120:
        raise ValueError('Every winner/policy role must be tested')
    r=dict(fingerprint=fingerprint(config_path),junit_sha256=digest(xml),tests_passed=len(cases),
        own_tests=len(own),new_cache_test_fits=4,historical_test_fits_separate=True,
        test_source_sha256=digest('tests/test_unweighted_fusion.py'),created_at_utc=now())
    p=dest/'test_fit_counts.json';ledger=read_config(p)['invocations'] if p.exists() else []
    if not any(v['junit_sha256']==r['junit_sha256'] for v in ledger):ledger.append(r)
    write_json(p,dict(invocations=ledger,latest=r,total_new_cache_test_fits=sum(v['new_cache_test_fits'] for v in ledger)))
    write_json(Path(c['output_dir'])/'checks.json',r)
    return r

def source_miniature_gate(c,fp):
    source=read_config(c['fusion_source_config'])
    exported=Path(source['report_dir'])/'source_miniature_gate.json'
    receipt=read_config(exported)
    if receipt['fingerprint']!=c['fusion_source_fingerprint'] or receipt['preserved_fits']!=8:
        raise ValueError('Parent miniature gate changed')
    path=Path(receipt['source']);mini=read_config(path)
    if digest(path)!=receipt['sha256'] or mini['fingerprint']!=c['interaction_source_fingerprint'] or mini['fit_count']!=8:
        raise ValueError('Source miniature ancestry changed')
    present={r['artifact']:r['sha256'] for r in mini['results'] if Path(r['artifact']).exists()}
    verify_hashes(present)
    write_json(Path(c['output_dir'])/'source_miniature_gate.json',dict(fingerprint=fp,
        source=str(path),sha256=digest(path),parent_receipt=str(exported),parent_sha256=digest(exported),
        preserved_fits=8,optional_source_artifacts_present=len(present),new_production_miniature_fits=0))


def preflight(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir']);checks=read_config(out/'checks.json')
    if (checks['fingerprint']!=fp or checks['junit_sha256']!=digest(Path(c['report_dir'])/'tests.xml')
            or checks['test_source_sha256']!=digest('tests/test_unweighted_fusion.py')):
        raise ValueError('Missing/stale test gate')
    source_miniature_gate(c,fp)
    p=out/'resource_preflight.json'
    if p.exists():
        if read_config(p)['fingerprint']!=fp:raise ValueError('Resource gate changed')
        return read_config(p)
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    if (out/'failures'/'resource-preflight.json').exists():raise ValueError('Prior resource failure retained')
    try:
        inp=InteractionInputs().fit(base);sample=inp.transform(base.iloc[:8192]);n,d=len(base),len(inp.names)
        matrix_upper=12*n*d+8*(n+1);workspace_upper=4*matrix_upper+8*(d+1)**2+4*1024**3
        ram=(int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)) if sys.platform=='darwin'
             else os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'))
        r=dict(fingerprint=fp,host_ram_bytes=ram,training_rows=n,training_months=list(range(7)),dimension=d,
            max_chunk_rows=8192,sample=storage(sample),matrix_upper_bytes=matrix_upper,
            workspace_upper_bytes=workspace_upper,allowance_fraction=.7,elapsed_seconds=time.perf_counter()-start,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,predictor_only=True)
        if workspace_upper>ram*.7:raise ValueError('Largest transfer allocation exceeds 70% host allowance')
        write_json(p,r);return r
    except Exception as exc:
        write_json(out/'failures'/'resource-preflight.json',dict(fingerprint=fp,error=repr(exc),created_at_utc=now()))
        raise

def execution_gate(c, fp):
    out=Path(c['output_dir'])
    for n in ['checks.json','source_miniature_gate.json','resource_preflight.json']:
        if read_config(out/n)['fingerprint']!=fp:raise ValueError('Execution gate stale: '+n)

def prepare_development(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Development is frozen')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    records=prepare(c,fp,frame,base)
    result=dict(fingerprint=fp,base_results=len(records),base_reuses=sum(bool(r.get('reused')) for r in records),
        new_fits=sum(not r.get('reused',False) for r in records),
        derived_reuses=sum(read_config(paths(c,'validation',m,s['id'])[0])['reused']
            for m in [4,5,6] for s in strategies() if s['kind']!='native'),
        derived_results=114,
        elapsed_seconds=time.perf_counter()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['base_results']!=18:raise ValueError('Required development base set differs')
    result['new_derived_computations']=result['derived_results']-result['derived_reuses']
    write_json(out/'preparation.json',result);return result

def validate(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Development is frozen')
    if read_config(out/'preparation.json')['fingerprint']!=fp:raise ValueError('Preparation missing/stale')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
    cf=crossfit(c);generated,main,pairs=tables(c,cf,frame,base)
    for name,t in generated.items():t.to_csv(out/(name+'.csv'),index=False)
    write_json(out/'feature_blocks.json',main);write_json(out/'interaction_blocks.json',pairs)
    counts=dict(all_scores=len(list((out/'predictions').glob('validation-*.npz'))),
        selectable_scores=120,derived_results=114,forward_rows=len(cf))
    if counts['all_scores']!=132 or len(cf)!=240:raise ValueError('Finite development outputs differ')
    write_json(out/'validation_complete.json',dict(fingerprint=fp,created_at_utc=now(),
        files={p:digest(out/p) for p in TABLES},**counts))
    from .unweighted_fusion_report import audit_development
    audit_development(config_path)
    print(generated['summary'].head(12).to_string(index=False),flush=True)

def required_final_specs(c, choices):
    needed=dependencies([n for n,k in choices.values()])['bases']
    return [s for s in recipes()+c['references']+c['presets'] if s['id'] in needed]

def final_fit_bounds(c, choices):
    specs=required_final_specs(c,choices);availability={};counts=dict(interaction=0,spline=0,tree=0)
    for spec in specs:
        found=None
        for source,identity in source_origins(c,spec):
            ps=cache_paths(source,spec,7,'exploratory')
            if tree.cache_state(ps):
                found={str(p):digest(p) for p in ps};break
        availability[spec['id']]=found
        if found is None:counts[spec.get('representation','tree')]+=1
    return dict(unique_base_results=len(specs),max_new_interaction_fits=counts['interaction'],
        max_reproduced_additive_fits=counts['spline'],max_reproduced_tree_fits=counts['tree'],
        original_availability=availability)

def freeze(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    done=read_config(out/'validation_complete.json');audit=read_config(out/'independent_development_audit.json')
    if (done['fingerprint']!=fp or audit['fingerprint']!=fp
            or audit['validation_sha256']!=digest(out/'validation_complete.json')):
        raise ValueError('Independent audit missing/stale')
    verify_hashes({str(out/p):h for p,h in done['files'].items()})
    winner=pd.read_csv(out/'summary.csv').iloc[0];choices=choices_for(str(winner.candidate),str(winner.policy));h=history(c)
    write_json(out/'selection.json',dict(created_at_utc=now(),fingerprint=fp,config=c,
        selected_candidate=str(winner.candidate),selected_policy=str(winner.policy),selected_development_f2=float(winner.f2),
        choices=choices,rules={a:learn(h[n],k,7) for a,(n,k) in choices.items()},
        fit_specs=required_final_specs(c,choices),score_dependencies=dependencies([n for n,k in choices.values()]),
        final_fit_bounds=final_fit_bounds(c,choices),validation_files=done['files'],
        independent_audit_sha256=digest(out/'independent_development_audit.json'),
        artifacts={str(p):digest(p) for folder in ['runs','predictions','models'] for p in (out/folder).glob('*') if p.is_file()},
        month7_previously_consumed=True,exploratory=True))
    print('Frozen',winner.candidate,winner.policy,float(winner.f2),'roles',len(choices),flush=True)

def load_selection(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['config']!=c or s['fingerprint']!=fp:raise ValueError('Frozen execution identity changed')
    verify_hashes(s['artifacts']);verify_hashes({str(out/p):h for p,h in s['validation_files'].items()})
    if s['independent_audit_sha256']!=digest(out/'independent_development_audit.json'):raise ValueError('Frozen audit changed')
    winner=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0]
    if ((s['selected_candidate'],s['selected_policy'])!=(str(winner.candidate),str(winner.policy))
            or not np.isclose(s['selected_development_f2'],winner.f2,atol=1e-12,rtol=0)):
        raise ValueError('Frozen actual macro winner differs')
    roles=choices_for(str(winner.candidate),str(winner.policy))
    if (s['choices']!=roles or s['fit_specs']!=required_final_specs(c,roles)
            or s['score_dependencies']!=dependencies([n for n,k in roles.values()])
            or s['final_fit_bounds']!=final_fit_bounds(c,roles)):
        raise ValueError('Frozen role/dependency/cache bounds changed')
    h=history(c)
    if any(s['rules'][a]!=learn(h[n],k,7) for a,(n,k) in roles.items()):raise ValueError('Frozen earlier-only rule differs')
    return c,s

def final_data(c,s):
    return {a:prediction(c,'exploratory',7,n) for a,(n,k) in s['choices'].items()}

def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);sha=digest(out/'selection.json');ap=out/'exploratory_access.json'
    if not ap.exists():write_json(ap,dict(access_at_utc=now(),selection_sha256=sha,month7_previously_consumed=True))
    access=read_config(ap)
    if access['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:
        raise ValueError('Freeze/access chronology differs')
    frame=load_data(c);base=features(frame);names={n for n,k in s['choices'].values()}
    records=prepare(c,s['fingerprint'],frame,base,'exploratory',names)
    data=final_data(c,s);rows=[]
    for a,d in data.items():
        aligned(d,data['selected']);rule=s['rules'][a]
        rows.append(dict(role=a,candidate=s['choices'][a][0],policy=s['choices'][a][1],eval_month=7,
                         parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    fresh=[r for r in records if not r.get('reused',False)]
    counts=dict(base_results=len(records),base_reuses=len(records)-len(fresh),new_fits=len(fresh),
        new_interaction_fits=sum(r['spec'].get('representation')=='interaction' for r in fresh),
        reproduced_additive_fits=sum(r['spec'].get('representation')=='spline' for r in fresh),
        reproduced_tree_fits=sum(r['spec'].get('representation') not in ['spline','interaction'] for r in fresh),
        role_count=len(rows),total_scores=len(list((out/'predictions').glob('exploratory-*.npz'))))
    if {r['candidate'] for r in records}!={s['id'] for s in s['fit_specs']}:
        raise ValueError('Transfer base set differs')
    for actual,bound in [('new_interaction_fits','max_new_interaction_fits'),
                         ('reproduced_additive_fits','max_reproduced_additive_fits'),
                         ('reproduced_tree_fits','max_reproduced_tree_fits')]:
        if counts[actual]>s['final_fit_bounds'][bound]:raise ValueError('Frozen transfer bound exceeded')
    if counts['total_scores']!=len(s['score_dependencies']['scores']):raise ValueError('Transfer score DAG differs')
    write_json(out/'transfer_complete.json',dict(selection_sha256=sha,table_sha256=digest(out/'exploratory.csv'),**counts))
    print(table[['role','candidate','policy','f2','precision','recall','tp','fp','fn']].to_string(index=False),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['initialize','record_checks','preflight',
        'prepare_development','validate','freeze','exploratory']);parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args();globals()[args.command](args.config);print('Completed',args.command,flush=True)
