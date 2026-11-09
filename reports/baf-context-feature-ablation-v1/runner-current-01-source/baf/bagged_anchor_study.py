"""Isolated bagged-anchor experiment and individual immutable source adapters."""
from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import os
import resource
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import bagged_boosting as parent
from . import bagged_anchor_scores as scores
from .common import digest, read_config, write_json, load_data, features
from .focal_study import expected_record, now
from .improvements import paths, prediction
from .recall_weighted_metrics import POLICIES, aligned, learn, measure, decisions
from .recall_extension import order_summary, expected_population
from .improvement_metrics import confusion_metrics
from .spline_inputs import AdditiveInputs
from .interaction_inputs import storage

CONFIG = 'configs/bagged-anchor-fusion.json'
SOURCES = parent.SOURCES + ['baf/bagged_anchor_scores.py', 'baf/bagged_anchor_study.py',
                           'baf/bagged_anchor_report.py']
COUNTS = scores.declaration_counts()
TABLES = ['crossfit.csv', 'summary.csv', 'pooled.csv', 'contrasts.csv',
          'score_diagnostics.csv', 'clipping.csv', 'diversity.csv', 'decision_overlap.csv',
          'model_diagnostics.csv', 'expressions.csv', 'preprocessing.json', 'reference_diagnostics.json']
verify_hashes = parent.verify_hashes
cache_state = parent.cache_state
reference_specs = parent.reference_specs


def fingerprint(path=CONFIG):
    fp = parent.fingerprint(path)
    fp['sources'] = {p:digest(p) for p in SOURCES}
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
    source = read_config(c['parent_config'])
    # Parent guards receive their own config, never this new descendant's config.
    parent.guard(source)
    if c['parent_config']!=parent.CONFIG or parent.fingerprint()!=c['parent_fingerprint']:
        raise ValueError('Frozen parent identity changed')
    preserved = mandatory_preservation(source)
    if c['preserve_sha256']!=preserved or len(preserved)!=735:
        raise ValueError('Mandatory preservation union changed')
    verify_hashes(preserved)
    selections = [p for p in preserved if p.startswith('outputs/') and p.endswith('/selection.json')]
    if len(selections)!=16 or any(preserved.get(p.replace('outputs/','reports/',1))!=preserved[p] for p in selections):
        raise ValueError('Sixteen original/exported selection pairs required')
    expected = dict(parent_config=parent.CONFIG, parent_fingerprint=parent.fingerprint(),
        preserve_sha256=preserved, source_paths=SOURCES, strategies=scores.strategies(),
        expected_counts=COUNTS, alphas=scores.ALPHAS, anchor=scores.ANCHOR,
        output_dir='outputs/baf-bagged-anchor-fusion-v1',
        report_dir='reports/baf-bagged-anchor-fusion-v1',
        protocol_path='docs/bagged-anchor-fusion-protocol.md',
        historical_output_roots=sorted(set(source['historical_output_roots']+[source['output_dir']])))
    if set(c)!=set(source)|set(expected) or any(c.get(k)!=v for k,v in expected.items()):
        raise ValueError('Prospective finite declaration changed')
    if any(c.get(k)!=v for k,v in source.items() if k not in expected):
        raise ValueError('Frozen source recipe changed')
    if len(SOURCES)!=len(set(SOURCES)) or len(SOURCES)!=55:
        raise ValueError('Source closure changed')
    dest = [Path(c[k]).resolve() for k in ['output_dir','report_dir']]
    for a,b in combinations(dest+[Path(p).resolve() for p in c['historical_output_roots']],2):
        if (a in dest or b in dest) and (a==b or a in b.parents or b in a.parents):
            raise ValueError('Destination overlap')
    out = Path(c['output_dir'])
    for name in ['preimplementation_preservation.json','preservation.json']:
        p = out/name
        if p.exists():
            r = read_config(p)
            verify_hashes(r if name.startswith('preimplementation') else r['optional_present'])
            if name=='preservation.json' and (r['mandatory']!=preserved or r['roots']!=c['historical_output_roots']):
                raise ValueError('Historical snapshot declaration differs')


def initialize(config_path=CONFIG):
    """Freeze execution identity only after implementation and checks are ready."""
    c = read_config(config_path);guard(c);fp = fingerprint(config_path)
    out = Path(c['output_dir']);p = out/'preservation.json'
    if not p.exists():
        initial = out/'preimplementation_preservation.json'
        present = read_config(initial) if initial.exists() else {
            str(p):digest(p) for root in c['historical_output_roots'] for p in Path(root).rglob('*') if p.is_file()}
        write_json(p,dict(mandatory=c['preserve_sha256'],optional_present=present,roots=c['historical_output_roots']))
    record = dict(fingerprint=fp,config_path=str(config_path),protocol_sha256=digest(c['protocol_path']),
        preservation_sha256=digest(p),strategies=c['strategies'],expected_counts=COUNTS,
        month7_previously_consumed=True)
    p = out/'declaration.json'
    if p.exists():
        if any(read_config(p).get(k)!=v for k,v in record.items()):
            raise ValueError('Frozen execution declaration changed')
    else:
        write_json(p,dict(**record,created_at_utc=now()))
    return c,fp


def source_origins(c,spec):
    source = read_config(c['parent_config'])
    origins = [(source,c['parent_fingerprint'])]
    return origins if spec in scores.recipes() else origins+parent.source_origins(source,spec)


def source_paths(c,spec,month,phase):
    return (parent.bagged_paths(c,phase,month,spec['id']) if spec in scores.recipes()
            else parent.previous.cache_paths(c,spec,month,phase))


def audit_source(c,fp,frame,base,spec,month,phase='validation'):
    method = parent.audit_bagged if spec in scores.recipes() else parent.previous.audit_base
    record = method(c,fp,frame,base,spec,month,phase)
    verify_hashes(record.get('origin_artifacts',{}))
    return record


def prepare_source(c,fp,frame,base,spec,month,phase='validation',allow_missing=True):
    """Own -> current parent -> declared older origins; partial sets fail in place."""
    if spec not in reference_specs(c)+scores.recipes():
        raise ValueError('Undeclared individual recipe')
    own = source_paths(c,spec,month,phase)
    if cache_state(own):
        return audit_source(c,fp,frame,base,spec,month,phase)
    if not allow_missing:
        raise ValueError('Required individual cache missing')
    failure = Path(c['output_dir'])/'failures'/(own[0].stem+'.json')
    if failure.exists():
        raise ValueError('Prior failed source gate retained; no automatic retry')
    try:
        for source,identity in source_origins(c,spec):
            origin = source_paths(source,spec,month,phase)
            if not cache_state(origin):
                continue
            record = audit_source(source,identity,frame,base,spec,month,phase)
            for src,dst in zip(origin[1:],own[1:]):
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            record = dict(record,fingerprint=fp,reused=True,
                          origin_artifacts={str(p):digest(p) for p in origin})
            if spec in scores.recipes():
                pass  # Bagged serialized models are path independent.
            elif spec.get('representation')=='spline':
                record.update(original_fit_fingerprint=c['newton_fit_fingerprint'],
                              original_analysis_fingerprint=c['newton_analysis_fingerprint'])
            elif spec['id']=='xgb_d4_w1':
                record['model_path']=str(own[2])
            else:
                record.update(reference_model_path=str(own[2]),reference_model_new_fit=False)
            write_json(own[0],record)
            return audit_source(c,fp,frame,base,spec,month,phase)
        if spec in scores.recipes():
            parent.fit_bagged(c,fp,frame,base,spec,month,phase)
        elif spec.get('representation')=='spline':
            parent.previous.newton.fit_new(c,fp,frame,base,spec,month,phase)
        elif spec['id']=='xgb_d4_w1':
            parent.previous.tree.old.fit_model(c,fp,frame,base,spec,month,phase)
        else:
            parent.previous.additive.reproduce_reference(c,fp,frame,base,spec,month,phase)
        return audit_source(c,fp,frame,base,spec,month,phase)
    except Exception as exc:
        if not failure.exists():
            write_json(failure,dict(**expected_record(frame,spec,month,phase,fp),error=repr(exc),
                numerical=getattr(exc,'diagnostics',{}),created_at_utc=now()))
        raise


def derived_spec(name):
    if name in parent.dependencies(list(scores.REFERENCE_POLICIES))['derived']:
        return parent.old_lookup(name)
    return scores.lookup(name)


def arithmetic(spec):
    return ('preserved reference' if spec['kind']=='anchor' else
            'ordered float64 sum / 3' if spec['kind']=='aggregate' else
            'log(p)-log1p(-p) after clip' if spec['kind']=='logit' else
            '(1-alpha)*anchor + alpha*margin')


def derived(c,fp,name,month,phase='validation',allow_missing=True):
    spec = derived_spec(name)
    items = {n:prediction(c,phase,month,n) for n in spec['members']}
    d = parent.old_compose(spec,items) if spec['kind']=='anchor' else scores.compose(spec,items)
    deps = {n:{str(p):digest(p) for p in paths(c,phase,month,n)} for n in spec['members']}
    expected = dict(candidate=name,phase=phase,eval_month=month,fingerprint=fp,spec=spec,
                    dependencies=deps,epsilon=scores.EPSILON,arithmetic=arithmetic(spec))
    rp,pp = paths(c,phase,month,name)
    if cache_state([rp,pp]):
        r = read_config(rp);saved=prediction(c,phase,month,name);aligned(d,saved)
        if any(r.get(k)!=v for k,v in expected.items()) or digest(pp)!=r['prediction_sha256'] or not np.array_equal(d['scores'],saved['scores']):
            raise ValueError('Derived identity/arithmetic changed')
        verify_hashes(r['origin_artifacts']);return r
    if not allow_missing:
        raise ValueError('Required derived cache absent')
    origin = {}
    if name in parent.dependencies([s['id'] for s in parent.strategies()])['derived']:
        source = read_config(c['parent_config']);sr,sp = paths(source,phase,month,name)
        if cache_state([sr,sp]):
            r = read_config(sr);original=prediction(source,phase,month,name);aligned(d,original)
            parents = {n:{str(p):digest(p) for p in paths(source,phase,month,n)} for n in spec['members']}
            check = dict(candidate=name,phase=phase,eval_month=month,fingerprint=c['parent_fingerprint'],
                         spec=spec,dependencies=parents,epsilon=scores.EPSILON,arithmetic=arithmetic(spec))
            if any(r.get(k)!=v for k,v in check.items()) or digest(sp)!=r['prediction_sha256'] or not np.array_equal(original['scores'],d['scores']):
                raise ValueError('Preserved parent composition changed')
            verify_hashes(r.get('origin_artifacts',{}))
            origin = {str(p):digest(p) for p in [sr,sp]}
            for member in parents.values():
                origin.update(member)
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**d)
    record = dict(**expected,prediction_sha256=digest(pp),reused=bool(origin),origin_artifacts=origin)
    write_json(rp,record);return record


def derived_order(names):
    dag = scores.dependencies(names);done=set(dag['bases']);order=[]
    while done!=set(dag['scores']):
        ready = [n for n in dag['derived'] if n not in done and set(derived_spec(n)['members'])<=done]
        if not ready:
            raise ValueError('Unresolvable derived DAG')
        order.extend(ready);done.update(ready)
    return order


def prepare(c,fp,frame,base,phase='validation',names=None,allow_missing=True):
    names = [s['id'] for s in scores.strategies()] if names is None else names
    dag = scores.dependencies(names);specs={s['id']:s for s in reference_specs(c)+scores.recipes()}
    records,derived_records,work = [],[],[]
    for m in c['validation_months'] if phase=='validation' else [7]:
        for n in dag['bases']:
            print('Preparing',phase,m,n,flush=True)
            own_existed=cache_state(source_paths(c,specs[n],m,phase))
            r=prepare_source(c,fp,frame,base,specs[n],m,phase,allow_missing)
            records.append(r)
            work.append(dict(candidate=n,eval_month=m,kind='bagged' if n in dag['bagged_bases'] else 'reference',
                work='audit existing' if own_existed else 'reuse origin' if r.get('reused') else 'fit exact recipe'))
            if allow_missing:
                write_json(Path(c['output_dir'])/(phase+'_progress.json'),dict(fingerprint=fp,work=work))
        for n in derived_order(names):
            existed=cache_state(paths(c,phase,m,n));r=derived(c,fp,n,m,phase,allow_missing)
            derived_records.append(r)
            work.append(dict(candidate=n,eval_month=m,kind='derived',
                work='audit existing' if existed else 'reuse origin' if r['reused'] else 'calculate'))
            if allow_missing:
                write_json(Path(c['output_dir'])/(phase+'_progress.json'),dict(fingerprint=fp,work=work))
    return records,derived_records,work


def history(c):
    return {s['id']:{m:prediction(c,'validation',m,s['id']) for m in c['validation_months']}
            for s in scores.strategies()}


def crossfit(c):
    h=history(c);rows=[]
    for m,n,k in product([5,6],h,POLICIES):
        aligned(h[n][m],h[scores.ANCHOR][m])
        rule=learn({t:d for t,d in h[n].items() if t<m},k,m)
        rows.append(dict(candidate=n,policy=k,eval_month=m,source_months=','.join(map(str,rule['source_months'])),
                         parameters=json.dumps(rule,sort_keys=True),**measure(h[n][m],rule)))
    return pd.DataFrame(rows)


def tables(c,cf):
    summary=order_summary(cf).reset_index(drop=True)
    auc=cf.groupby(['candidate','policy'])['roc_auc'].mean()
    summary['roc_auc']=[float(auc.loc[(r.candidate,r.policy)]) for r in summary.itertuples()]
    index=summary.set_index(['candidate','policy']);contrasts=[];pooled=[]
    for label,n,k,ref,rk in scores.contrast_factors():
        a,b=index.loc[(n,k)],index.loc[(ref,rk)]
        contrasts.append(dict(contrast=label,candidate=n,policy=k,reference=ref,reference_policy=rk,
            **{v+'_difference':float(a[v]-b[v]) for v in ['f2','recall','precision','flagged_fraction','average_precision']}))
    for (n,k),g in cf.groupby(['candidate','policy']):
        t=g[['tn','fp','fn','tp']].sum()
        pooled.append(dict(candidate=n,policy=k,**confusion_metrics(*[int(t[v]) for v in ['tn','fp','fn','tp']]),
                           flagged_fraction=float((t.tp+t.fp)/g.n.sum())))
    h=history(c);monthly=[];clips=[];diversity=[];joint=[];models=[];pre={};refmeta={}
    for n,months in h.items():
        for m,d in months.items():
            p=d['scores'];monthly.append(dict(candidate=n,eval_month=m,n=len(p),score_min=float(p.min()),
                score_max=float(p.max()),score_mean=float(p.mean()),unique_scores=len(np.unique(p)),tied_fraction=1-len(np.unique(p))/len(p)))
    for m,n in product([4,5,6],scores.dependencies(list(h))['bases']):
        r=read_config(paths(c,'validation',m,n)[0])
        if n.startswith('bagged_'):
            pre[f'{m}:{n}']=r['preprocessing']
            models.append(dict(candidate=n,eval_month=m,weighting=r['spec']['weighting'],bagging_seed=r['spec']['bagging_seed'],
                **{v:r[v] for v in ['fitted_tree_count','fit_seconds','peak_process_rss_bytes','train_rows','train_fraud']},
                transformed_dimension=r['preprocessing']['transformed_dimension'],
                **{v:json.dumps(r[v],sort_keys=True) for v in ['class_weights','warnings','requested_model_params','booster_params']}))
        else:
            refmeta[f'{m}:{n}']=r
    probabilities=scores.dependencies(list(h))['bases']+[scores.probability_mean(w) for w in scores.WEIGHTINGS]
    for m,n in product([4,5,6],probabilities):
        clips.append(dict(candidate=n,eval_month=m,**parent.clipping(prediction(c,'validation',m,n)['scores'])))
    for (label,n,r),m in product(scores.diagnostic_pairs(),[4,5,6]):
        a,b=h[n][m],h[r][m];aligned(a,b)
        diversity.append(dict(pair=label,first=n,second=r,eval_month=m,**parent.correlations(a['scores'],b['scores'])))
        if m>4:
            for k in POLICIES:
                pa=decisions(a,learn({i:d for i,d in h[n].items() if i<m},k,m))
                pb=decisions(b,learn({i:d for i,d in h[r].items() if i<m},k,m))
                joint.append(dict(pair=label,first=n,second=r,eval_month=m,policy=k,**parent.overlap(a,b,pa,pb)))
    return dict(crossfit=cf,summary=summary,pooled=pd.DataFrame(pooled),contrasts=pd.DataFrame(contrasts),
        score_diagnostics=pd.DataFrame(monthly),clipping=pd.DataFrame(clips),diversity=pd.DataFrame(diversity),
        decision_overlap=pd.DataFrame(joint),model_diagnostics=pd.DataFrame(models),
        expressions=pd.DataFrame(scores.expression_rows())),pre,refmeta


def verify_declaration(config_path=CONFIG):
    """No execution freeze, model fits, blend evaluation or consumed-data access."""
    c=read_config(config_path);guard(c)
    evidence=read_config(planning_evidence_path())
    if (c['source_paths']!=evidence['planned_source_paths']
            or c['preserve_sha256']!=evidence['mandatory_preservation_sha256']
            or [s['id'] for s in scores.strategies()]!=evidence['strategies']
            or COUNTS!=evidence['expected_counts']
            or set(scores.contrast_factors())!={tuple(x) for x in evidence['contrasts']}):
        raise ValueError('Prospective planning declaration differs')
    # Ignored historical models are optional in a clean checkout. Present files
    # retain their original hashes; preparation rejects partial required sets.
    present={p:h for p,h in evidence['historical_ignored_file_sha256'].items() if Path(p).exists()}
    verify_hashes(present)
    for case in evidence['winner_policy_cases']:
        roles=scores.choices_for(case['winner'],case['policy'])
        if roles!=case['roles'] or scores.dependencies([n for n,k in roles.values()])!=case['dependencies']:
            raise ValueError('Winner-role dependency declaration differs')
    if fingerprint(config_path)['sources'].keys()!=dict.fromkeys(SOURCES).keys():
        raise ValueError('Source closure incomplete')
    return dict(verified_at_utc=now(),config_sha256=digest(config_path),
        implementation_sources={p:digest(p) for p in SOURCES[-3:]},protocol_sha256=digest(c['protocol_path']),
        source_paths=55,preserved_paths=735,selection_pairs=16,historical_files=len(present),
        declared_historical_snapshot_files=3732,
        **COUNTS)


def planning_evidence_path(root=Path('openspec/changes')):
    active=root/'test-bagged-anchor-fusion-f2/planning-evidence.json'
    matches=[active] if active.is_file() else list((root/'archive').glob('*-test-bagged-anchor-fusion-f2/planning-evidence.json'))
    if len(matches)!=1:
        raise ValueError('Exactly one prospective planning evidence artifact required')
    return matches[0]


def record_checks(config_path=CONFIG):
    """Record a closed full-suite receipt without freezing unfinished execution code."""
    c=read_config(config_path);guard(c);out=Path(c['output_dir']);dest=Path(c['report_dir'])
    if (out/'selection.json').exists():
        raise ValueError('Selection-era checks cannot be overwritten')
    xml=dest/'tests.xml';root=ET.parse(xml).getroot();cases=list(root.iter('testcase'))
    if not cases or any(list(root.iter(n)) for n in ['failure','error','skipped']):
        raise ValueError('Complete passing methodological suite required')
    own=[t.get('name','') for t in cases if t.get('classname','').endswith('test_bagged_anchor')]
    formats=[n for n in own if n.startswith('test_six_genuine_missing_cache_formats_fit_audit_reuse_and_reject_corruption[')]
    expected={'bagged_balanced','bagged_unweighted','spline_balanced','spline_unweighted','xgboost','lightgbm'}
    if {n.split('[')[-1].removesuffix(']') for n in formats}!=expected or len(formats)!=6:
        raise ValueError('All six genuine tiny cache-format fits required')
    if sum(n.startswith('test_every_winner_policy[') for n in own)!=111:
        raise ValueError('All 111 winner/policy cases required')
    # Require every collected test, not a hand-picked passing subset.
    required=[]
    for path in sorted(Path('tests').glob('test_*.py')):
        required.append('tests.'+path.stem)
    actual_classes={t.get('classname','') for t in cases}
    if any(not any(v==n or v.startswith(n+'.') for v in actual_classes) for n in required):
        raise ValueError('Every historical/current methodological module required')
    receipt=dict(fingerprint=fingerprint(config_path),junit_sha256=digest(xml),tests_passed=len(cases),
        own_tests=len(own),new_cache_test_fits=6,historical_test_fits_separate=True,
        test_source_sha256=digest('tests/test_bagged_anchor.py'),created_at_utc=now())
    ledger_path=dest/'test_model_fits.json';ledger=read_config(ledger_path)
    if ledger['exit_code']!=0 or ledger['junit_sha256']!=digest(xml):
        raise ValueError('Complete actual-fit ledger required')
    if ledger['deselected_nodes']:
        raise ValueError('The execution receipt cannot use a selected subset')
    expected_keys=[]
    for node in ledger['collected_nodes']:
        parts=node.split('::')
        expected_keys.append(('.'.join([parts[0].removesuffix('.py').replace('/','.'),*parts[1:-1]]),parts[-1]))
    actual_keys=[(t.get('classname',''),t.get('name','')) for t in cases]
    if len(actual_keys)!=len(set(actual_keys)) or set(actual_keys)!=set(expected_keys):
        raise ValueError('Exact collected/executed test identities differ')
    verify_hashes(ledger['execution_source_sha256'])
    own_fits=[r for r in ledger['fits'] if r['nodeid'].startswith('tests/test_bagged_anchor.py::')]
    if len(own_fits)!=6 or any(r['status']!='completed' for r in own_fits):
        raise ValueError('Exactly six actual current-study tiny fits required')
    if any(not r['nodeid'].startswith('tests/test_bagged_anchor.py::test_six_genuine_missing_cache_formats') for r in own_fits):
        raise ValueError('Unexpected current-study test fit')
    historical=[r for r in ledger['fits'] if r not in own_fits]
    if any(r['nodeid'] not in ledger['collected_nodes'] for r in ledger['fits']):
        raise ValueError('Unattributed model-fit call')
    receipt.update(actual_fit_ledger_sha256=digest(ledger_path),actual_current_test_fits=len(own_fits),
                   actual_historical_test_fit_attempts=len(historical),
                   actual_historical_test_fits=sum(r['status']=='completed' for r in historical),
                   actual_historical_test_fit_failures=sum(r['status']=='failed' for r in historical))
    p=dest/'test_fit_counts.json';invocations=read_config(p)['invocations'] if p.exists() else []
    if not any(r['junit_sha256']==receipt['junit_sha256'] for r in invocations):
        invocations.append(receipt)
    write_json(p,dict(invocations=invocations,total_new_cache_test_fits=sum(r['new_cache_test_fits'] for r in invocations)))
    write_json(out/'checks.json',receipt)
    return receipt


def execution_gate(c,fp):
    out=Path(c['output_dir']);dest=Path(c['report_dir'])
    for name in ['checks.json','resource_preflight.json','source_miniature_gate.json']:
        if read_config(out/name)['fingerprint']!=fp:
            raise ValueError('Missing/stale execution gate '+name)
    checks=read_config(out/'checks.json')
    if (checks['test_source_sha256']!=digest('tests/test_bagged_anchor.py')
            or checks['junit_sha256']!=digest(dest/'tests.xml')
            or checks['actual_fit_ledger_sha256']!=digest(dest/'test_model_fits.json')):
        raise ValueError('Execution test evidence changed')
    verify_hashes(read_config(dest/'test_model_fits.json')['execution_source_sha256'])
    gate=read_config(out/'source_miniature_gate.json');verify_hashes(gate['artifacts'])


def _source_miniature_gate(c,fp):
    """Audit preserved sampling evidence, fitting no new miniature model."""
    out=Path(c['output_dir']);source=read_config(c['parent_config'])
    exported=Path(source['report_dir'])/'miniature_gate.json';r=read_config(exported)
    if (r['fingerprint']!=c['parent_fingerprint'] or r['fit_count']!=9
            or not all(r.get(k) is True for k in ['same_seed_repeatable','inactive_seed_controls_equal','active_seed_diversity'])):
        raise ValueError('Preserved nine-fit sampling gate differs')
    expected=[(s['id'],s,True) for s in scores.recipes()]+[
        ('repeat',scores.recipes()[0],True),('inactive17',scores.recipes()[0],False),
        ('inactive43',scores.recipes()[2],False)]
    if len(r['results'])!=9 or set(r['files'])!={p for item in r['results'] for p in item['files']}:
        raise ValueError('Preserved sampling artifact union differs')
    for item,(tag,spec,active) in zip(r['results'],expected):
        if (item['tag'],item['spec'],item['active'])!=(tag,spec,active):
            raise ValueError('Preserved sampling recipe differs')
    present={p:h for p,h in r['files'].items() if Path(p).exists()}
    if present and len(present)!=len(r['files']):
        raise ValueError('Partial optional sampling miniature set')
    verify_hashes(present)
    if present:
        frame,base=parent.synthetic_data();train=frame.index[frame.month<2];ev=frame.index[frame.month==2]
        values={}
        for item in r['results']:
            model_path,prediction_path=list(item['files'])
            model=joblib.load(model_path);saved=np.load(prediction_path)['scores']
            parent.audit_preprocessing(model,base.loc[train])
            parent.effective_parameters(model.named_steps['model'],item['spec'],c['threads'],active=item['active'])
            with threadpool_limits(limits=c['threads']):
                actual=model.predict_proba(base.loc[ev])[:,1]
            if not np.allclose(actual,saved,atol=1e-12,rtol=0):
                raise ValueError('Preserved sampling saved inference differs')
            values[item['tag']]=saved
        if not np.array_equal(values['repeat'],values['bagged_balanced_s17']) or not np.array_equal(values['inactive17'],values['inactive43']):
            raise ValueError('Preserved repeat/inactive sampling evidence differs')
        for w in scores.WEIGHTINGS:
            if all(np.array_equal(values[f'bagged_{w}_s17'],values[f'bagged_{w}_s{s}']) for s in [29,43]):
                raise ValueError('Preserved active sampling has no diversity')
    ancestry_dir=out/'source-ancestry'
    parent.previous.source_miniature_gate(dict(c,output_dir=str(ancestry_dir)),fp)
    ancestry=ancestry_dir/'source_miniature_gate.json'
    result=dict(fingerprint=fp,preserved_sampling_fits=9,preserved_older_fits=8,
        optional_sampling_artifacts_present=len(present),new_production_miniature_fits=0,
        artifacts={str(exported):digest(exported),str(ancestry):digest(ancestry),**present},
        older_ancestry=read_config(ancestry),created_at_utc=now())
    p=out/'source_miniature_gate.json'
    if p.exists():
        old=read_config(p)
        for k in ['fingerprint','preserved_sampling_fits','preserved_older_fits','new_production_miniature_fits']:
            if old.get(k)!=result[k]:raise ValueError('Preserved sampling receipt changed')
        verify_hashes(old['artifacts']);return old
    write_json(p,result);return result


def source_miniature_gate(c,fp):
    failure=Path(c['output_dir'])/'failures'/'source-miniature.json'
    if failure.exists():raise ValueError('Prior source-miniature failure retained')
    try:
        return _source_miniature_gate(c,fp)
    except Exception as exc:
        write_json(failure,dict(fingerprint=fp,error=repr(exc),new_production_miniature_fits=0,created_at_utc=now()))
        raise


def preflight(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir']);checks=read_config(out/'checks.json')
    if checks['fingerprint']!=fp:
        raise ValueError('Current complete methodological checks required')
    if checks['actual_fit_ledger_sha256']!=digest(Path(c['report_dir'])/'test_model_fits.json'):
        raise ValueError('Current actual-fit ledger required')
    if (checks['junit_sha256']!=digest(Path(c['report_dir'])/'tests.xml')
            or checks['test_source_sha256']!=digest('tests/test_bagged_anchor.py')):
        raise ValueError('Current complete test execution evidence required')
    verify_hashes(read_config(Path(c['report_dir'])/'test_model_fits.json')['execution_source_sha256'])
    source_miniature_gate(c,fp);p=out/'resource_preflight.json'
    if p.exists():
        if read_config(p)['fingerprint']!=fp:raise ValueError('Resource gate identity changed')
        return read_config(p)
    failure=out/'failures'/'resource-preflight.json'
    if failure.exists():raise ValueError('Prior resource failure retained')
    start=time.perf_counter()
    try:
        frame=load_data(c);frame=frame[frame.month<7];base=features(frame)
        inp=AdditiveInputs('spline').fit(base);sample=inp.transform(base.iloc[:8192]);n,d=len(base),len(inp.names)
        pre=parent.factory(c,scores.recipes()[0],base.columns).named_steps['preprocess'].fit(base)
        oldsample=pre.transform(base.iloc[:8192]);original_dimension=oldsample.shape[1]
        maximum=max(d,original_dimension);matrix=12*n*maximum+8*(n+1);workspace=4*matrix+8*(maximum+1)**2+4*1024**3
        ram=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)) if sys.platform=='darwin' else os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
        result=dict(fingerprint=fp,host_ram_bytes=ram,training_rows=n,training_months=list(range(7)),
            additive_dimension=d,original_tree_dimension=original_dimension,max_dimension=maximum,max_chunk_rows=8192,
            additive_sample=storage(sample),original_tree_sample=storage(oldsample),matrix_upper_bytes=matrix,
            workspace_upper_bytes=workspace,allowance_fraction=.7,predictor_only=True,
            elapsed_seconds=time.perf_counter()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if workspace>ram*.7:raise ValueError('Largest required allocation exceeds70% hostRAM')
        write_json(p,result);return result
    except Exception as exc:
        write_json(failure,dict(fingerprint=fp,error=repr(exc),created_at_utc=now()));raise


def work_accounting(c,records,derived_records,work):
    specs={s['id']:s for s in reference_specs(c)+scores.recipes()}
    def count(kind,reused):
        return sum(bool(r.get('reused'))==reused and (
            'bagged' if r['candidate'].startswith('bagged_') else
            'spline' if specs[r['candidate']].get('representation')=='spline' else 'tree')==kind for r in records)
    return dict(base_results=len(records),bagged_reuses=count('bagged',True),new_bagged_fits=count('bagged',False),
        reference_reuses=count('spline',True)+count('tree',True),reproduced_additive_fits=count('spline',False),
        reproduced_tree_fits=count('tree',False),derived_results=len(derived_records),
        derived_reuses=sum(bool(r['reused']) for r in derived_records),
        new_derived_computations=sum(not r['reused'] for r in derived_records),invocation_work=work)


def prepare_development(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Development is frozen')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    records,ds,work=prepare(c,fp,frame,base)
    result=work_accounting(c,records,ds,work)
    if (len(records),len(ds))!=(33,87):raise ValueError('Complete required development DAG differs')
    result.update(fingerprint=fp,elapsed_seconds=time.perf_counter()-start,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(out/'preparation.json',result);return result


def validate(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Frozen validation cannot be overwritten')
    if read_config(out/'preparation.json')['fingerprint']!=fp:raise ValueError('Preparation missing/stale')
    generated,pre,refs=tables(c,crossfit(c))
    for n,table in generated.items():table.to_csv(out/(n+'.csv'),index=False)
    write_json(out/'preprocessing.json',pre);write_json(out/'reference_diagnostics.json',refs)
    write_json(out/'validation_complete.json',dict(fingerprint=fp,created_at_utc=now(),files={n:digest(out/n) for n in TABLES}))
    from .bagged_anchor_report import audit_development
    audit_development(config_path)
    print(generated['summary'].head(15).to_string(index=False),flush=True)


def final_fit_bounds(c,choices):
    dag=scores.dependencies([n for n,k in choices.values()]);specs={s['id']:s for s in reference_specs(c)+scores.recipes()}
    available={};missing=dict(bagged=0,spline=0,tree=0)
    for n in dag['bases']:
        spec=specs[n];found=None
        # Freeze availability in ORIGINAL origins. New-destination transfers must
        # not change a previously frozen availability record.
        for source,identity in source_origins(c,spec):
            ps=source_paths(source,spec,7,'exploratory')
            if cache_state(ps):found={str(p):digest(p) for p in ps};break
        available[n]=found
        if found is None:missing['bagged' if n in dag['bagged_bases'] else 'spline' if spec.get('representation')=='spline' else 'tree']+=1
    return dict(unique_base_results=len(dag['bases']),physical_scores=len(dag['scores']),
        max_new_bagged_fits=missing['bagged'],max_reproduced_additive_fits=missing['spline'],
        max_reproduced_tree_fits=missing['tree'],original_availability=available)


def freeze(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    done=read_config(out/'validation_complete.json');audit=read_config(out/'independent_development_audit.json')
    if done['fingerprint']!=fp or audit['fingerprint']!=fp or audit['validation_sha256']!=digest(out/'validation_complete.json'):
        raise ValueError('Independent prefreeze audit missing/stale')
    if not audit['actual_source_models_audited'] or audit['saved_model_checks']!=33:
        raise ValueError('All actual source audits required')
    verify_hashes({str(out/p):h for p,h in done['files'].items()})
    w=pd.read_csv(out/'summary.csv').iloc[0];choices=scores.choices_for(str(w.candidate),str(w.policy));h=history(c)
    if (audit['winner'],audit['policy'])!=(str(w.candidate),str(w.policy)) or not np.isclose(audit['development_f2'],w.f2,atol=1e-12,rtol=0):
        raise ValueError('Independently audited actual winner differs')
    record=dict(created_at_utc=now(),fingerprint=fp,config=c,selected_candidate=str(w.candidate),
        selected_policy=str(w.policy),selected_development_f2=float(w.f2),choices=choices,
        rules={a:learn(h[n],k,7) for a,(n,k) in choices.items()},score_dependencies=scores.dependencies([n for n,k in choices.values()]),
        final_fit_bounds=final_fit_bounds(c,choices),validation_files=done['files'],
        independent_audit_sha256=digest(out/'independent_development_audit.json'),
        artifacts={str(p):digest(p) for folder in ['runs','predictions','models'] for p in (out/folder).glob('*') if p.is_file()},
        month7_previously_consumed=True,exploratory=True)
    write_json(out/'selection.json',record)
    print('Frozen',w.candidate,w.policy,float(w.f2),'roles',len(choices),flush=True)
    return record


def load_selection(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['fingerprint']!=fp or s['config']!=c:raise ValueError('Frozen execution identity changed')
    verify_hashes(s['artifacts']);verify_hashes({str(out/p):h for p,h in s['validation_files'].items()})
    if s['independent_audit_sha256']!=digest(out/'independent_development_audit.json'):raise ValueError('Frozen audit changed')
    w=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0];roles=scores.choices_for(str(w.candidate),str(w.policy));h=history(c)
    if (s['selected_candidate'],s['selected_policy'])!=(str(w.candidate),str(w.policy)) or not np.isclose(s['selected_development_f2'],w.f2,atol=1e-12,rtol=0):
        raise ValueError('Actual macro winner differs')
    if s['choices']!=roles or s['score_dependencies']!=scores.dependencies([n for n,k in roles.values()]) or s['final_fit_bounds']!=final_fit_bounds(c,roles):
        raise ValueError('Frozen roles/DAG/bounds differ')
    if any(s['rules'][a]!=learn(h[n],k,7) for a,(n,k) in roles.items()):raise ValueError('Final earlier-only rules differ')
    return c,s


def final_data(c,s):
    return {a:prediction(c,'exploratory',7,n) for a,(n,k) in s['choices'].items()}


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);out=Path(c['output_dir']);sha=digest(out/'selection.json');p=out/'exploratory_access.json'
    if (out/'transfer_complete.json').exists():raise ValueError('Completed transfer cannot be overwritten')
    if not p.exists():write_json(p,dict(access_at_utc=now(),selection_sha256=sha,month7_previously_consumed=True))
    access=read_config(p)
    if access['selection_sha256']!=sha or not s['created_at_utc']<access['access_at_utc']:raise ValueError('Freeze/access chronology differs')
    frame=load_data(c);base=features(frame)
    records,ds,work=prepare(c,s['fingerprint'],frame,base,'exploratory',{n for n,k in s['choices'].values()})
    data=final_data(c,s);rows=[]
    for a,d in data.items():
        aligned(d,data['selected']);rule=s['rules'][a]
        rows.append(dict(role=a,candidate=s['choices'][a][0],policy=s['choices'][a][1],eval_month=7,
                         parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory.csv',index=False)
    counts=work_accounting(c,records,ds,work);bounds=s['final_fit_bounds']
    for key,limit in [('new_bagged_fits','max_new_bagged_fits'),('reproduced_additive_fits','max_reproduced_additive_fits'),('reproduced_tree_fits','max_reproduced_tree_fits')]:
        if counts[key]>bounds[limit]:raise ValueError('Frozen fit bound exceeded')
    total_scores=len(list((out/'predictions').glob('exploratory-*.npz')))
    if counts['base_results']!=bounds['unique_base_results'] or total_scores!=bounds['physical_scores']:
        raise ValueError('Transfer DAG work differs')
    write_json(out/'transfer_complete.json',dict(fingerprint=s['fingerprint'],selection_sha256=sha,
        table_sha256=digest(out/'exploratory.csv'),role_count=len(rows),total_scores=total_scores,**counts))
    print(table.to_string(index=False),flush=True);return counts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['verify_declaration','record_checks','preflight','prepare_development','validate','freeze','exploratory'])
    parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args()
    print(json.dumps(globals()[args.step](args.config),indent=2))
