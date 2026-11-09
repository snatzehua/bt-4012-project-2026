"""Isolated shallow CatBoost execution with explicit immutable reference ancestry."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path
import resource
import sys
import time
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import pearsonr,spearmanr

from . import conditional_expert_study as parent
from . import shallow_catboost as scores
from .common import read_config,write_json,digest,load_data,features,make_model
from .focal_study import now
from .improvements import paths,prediction
from .recall_weighted_metrics import POLICIES,aligned,learn,measure,decisions
from .recall_extension import order_summary,expected_population
from .improvement_metrics import confusion_metrics

CONFIG='configs/shallow-catboost.json'
CHANGE='test-shallow-catboost-f2'
PLANNING='openspec/changes/'+CHANGE+'/planning-evidence.json'
SOURCES=parent.SOURCES+['baf/shallow_catboost.py','baf/shallow_catboost_study.py','baf/shallow_catboost_report.py']
TEST_SOURCE='tests/test_shallow_catboost.py'
TABLES=[n+'.csv' for n in ['crossfit','summary','pooled','contrasts','physical_score_diagnostics','score_diagnostics','diversity','decision_overlap']]
WORK_TABLES=['model_work.csv','native_parameters.json','native_preprocessing.json']
verify_hashes=parent.verify_hashes
cache_state=parent.cache_state


def planning_evidence_path(root=Path('openspec/changes')):
    active=Path(root)/CHANGE/'planning-evidence.json'
    if active.is_file():return active
    found=list((Path(root)/'archive').glob('*-'+CHANGE+'/planning-evidence.json'))
    if len(found)!=1:raise ValueError('Exactly one original planning evidence required')
    return found[0]


def fingerprint(path=CONFIG):
    fp=parent.fingerprint(path);fp['sources']={p:digest(p) for p in SOURCES};return fp


def mandatory_preservation(source):
    result=dict(source['preserve_sha256'])
    for p in parent.SOURCES[-3:]+[parent.CONFIG,source['protocol_path'],
        str(Path(source['output_dir'])/'selection.json'),str(Path(source['report_dir'])/'report_hashes.json')]:result[p]=digest(p)
    result.update(read_config(Path(source['report_dir'])/'report_hashes.json'));return result


def declaration(source):
    p=read_config(planning_evidence_path())
    return dict(parent_config=parent.CONFIG,parent_fingerprint=parent.fingerprint(),
        reference_adapter_config=source['parent_config'],reference_adapter_fingerprint=source['parent_fingerprint'],
        preserve_sha256=mandatory_preservation(source),source_paths=SOURCES,
        strategies=scores.strategies(),native_recipes=scores.recipes(),reference_roles=scores.REFERENCE_ROLES,
        native_alphas=scores.ALPHAS,expected_counts=p['expected_counts'],
        planning_evidence_path=PLANNING,planning_evidence_sha256=digest(planning_evidence_path()),
        output_dir='outputs/baf-shallow-catboost-v1',report_dir='reports/baf-shallow-catboost-v1',
        protocol_path='docs/shallow-catboost-protocol.md',
        historical_output_roots=sorted(set(source['historical_output_roots']+[source['output_dir']])))


def guard(c):
    source=read_config(parent.CONFIG);parent.guard(source);expected=declaration(source)
    if set(c)!=set(source)|set(expected) or any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Prospective shallow declaration changed')
    if any(c.get(k)!=v for k,v in source.items() if k not in expected):raise ValueError('Frozen reference recipe changed')
    if len(SOURCES)!=len(set(SOURCES)) or len(SOURCES)!=61 or len(c['preserve_sha256'])!=1860:raise ValueError('Source/preservation closure changed')
    verify_hashes(c['preserve_sha256'])
    pairs=[p for p in c['preserve_sha256'] if p.startswith('outputs/') and p.endswith('/selection.json')]
    if len(pairs)!=18 or any(c['preserve_sha256'].get(p.replace('outputs/','reports/',1))!=c['preserve_sha256'][p] for p in pairs):raise ValueError('Eighteen matching selection pairs required')
    dest=[Path(c[k]).resolve() for k in ['output_dir','report_dir']]
    for a,b in combinations(dest+[Path(p).resolve() for p in c['historical_output_roots']],2):
        if (a in dest or b in dest) and (a==b or a in b.parents or b in a.parents):raise ValueError('Destination overlap')
    if any(k in c for k in ['primary_fraction','review_quota','precision_floor']):raise ValueError('Undeclared operating constraint')


def verify_declaration(config_path=CONFIG):
    c=read_config(config_path);guard(c);p=read_config(planning_evidence_path())
    if scores.recipes()!=p['recipes'] or scores.strategies()!=p['strategies'] or scores.contrast_factors()!=p['contrasts']:raise ValueError('Independent prospective identities differ')
    if c['preserve_sha256']!=p['mandatory_preservation_sha256'] or c['source_paths']!=p['planned_source_paths']:raise ValueError('Prospective closure differs')
    for case in p['winner_policy_cases']:
        roles=scores.choices_for(case['winner'],case['policy']);dag=scores.dependencies([n for n,k in roles.values()])
        if roles!=case['roles'] or dag!=case['dependencies']:raise ValueError('Prospective role/DAG differs')
        if len(dag['catboost_bases'])!=case['new_catboost_fit_bound'] or len(dag['full_scores'])!=case['physical_score_bound']:raise ValueError('Prospective work bounds differ')
    present=parent.present_history(c,p['historical_ignored_file_sha256']);out=Path(c['output_dir'])
    initial=out/'preimplementation_preservation.json'
    if initial.exists():verify_hashes(read_config(initial))
    else:write_json(initial,present)
    if digest(c['data_path'])!=c['data_sha256']:raise ValueError('Base checksum changed')
    receipt=dict(config_sha256=digest(config_path),planning_sha256=digest(planning_evidence_path()),
        planned_source_paths=SOURCES,mandatory_paths=1860,historical_original_files=4848,
        historical_present_files=len(present),selection_pairs=18,strategies=54,winner_policy_cases=162,
        contrasts=1728,new_model_fits=0,new_score_evaluations=0)
    path=out/'contract_verification.json'
    if path.exists() and read_config(path)!=receipt:raise ValueError('Original contract verification changed')
    if not path.exists():write_json(path,receipt)
    return receipt


def adapter_view(c):
    return dict(c,parent_config=c['reference_adapter_config'],parent_fingerprint=c['reference_adapter_fingerprint'])


def required_sources(c):return parent.required_sources(adapter_view(c))


def prepare_source(c,fp,frame,base,spec,month,phase='validation',allow_missing=True):
    if spec not in required_sources(c):raise ValueError('Unrelated reference source')
    return parent.parent.prepare_source(adapter_view(c),fp,frame,base,spec,month,phase,allow_missing)


def derived_reference(c,fp,name,month,phase='validation',allow_missing=True):
    return parent.derived_reference(adapter_view(c),fp,name,month,phase,allow_missing)


def physical_names():return scores.dependencies([s['id'] for s in scores.strategies()])['full_scores']


def history(c,*,physical=False):
    names=physical_names() if physical else [s['id'] for s in scores.strategies()]
    return {n:{m:prediction(c,'validation',m,n) for m in [4,5,6]} for n in names}


def new_derived(c,fp,name,month,phase='validation',allow_missing=True):
    s=scores.lookup(name)
    if s['kind']!='blend':raise ValueError('Not a declared blend')
    items=[prediction(c,phase,month,n) for n in s['members']]
    actual=scores.blend(*items,s['alpha_numerator']/100)
    deps={n:{str(p):digest(p) for p in paths(c,phase,month,n)} for n in s['members']}
    expected=dict(candidate=name,phase=phase,eval_month=month,fingerprint=fp,spec=s,
        dependencies=deps,arithmetic='(1-alpha)*anchor + alpha*native_raw_margin',score_domain='finite raw margin')
    rp,pp=paths(c,phase,month,name)
    if cache_state([rp,pp]):
        r=read_config(rp);saved=prediction(c,phase,month,name);aligned(actual,saved)
        if any(r.get(k)!=v for k,v in expected.items()) or digest(pp)!=r['prediction_sha256'] or not np.array_equal(actual['scores'],saved['scores']):raise ValueError('Blend identity/arithmetic differs')
        return r
    if not allow_missing:raise ValueError('Required blend absent')
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**actual)
    r=dict(**expected,prediction_sha256=digest(pp));write_json(rp,r);return r


def prepare(c,fp,frame,base,phase='validation',names=None,allow_missing=True):
    names=[s['id'] for s in scores.strategies()] if names is None else sorted(set(names));dag=scores.dependencies(names)
    refs={s['id']:s for s in required_sources(c)};native={s['id']:s for s in scores.recipes()}
    records,derived,work=[],[],[]
    for m in [4,5,6] if phase=='validation' else [7]:
        for n in dag['old_bases']:
            print('Preparing source',phase,m,n,flush=True)
            existed=cache_state(parent.parent.source_paths(c,refs[n],m,phase))
            r=prepare_source(c,fp,frame,base,refs[n],m,phase,allow_missing);records.append(r)
            work.append(dict(candidate=n,eval_month=m,kind='reference',work='audit existing' if existed else 'reuse origin' if r.get('reused') else 'fit exact recipe'))
        for n in parent.parent.derived_order([n for n,k in scores.REFERENCE_ROLES.values()]):
            existed=cache_state(paths(c,phase,m,n));r=derived_reference(c,fp,n,m,phase,allow_missing);derived.append(r)
            work.append(dict(candidate=n,eval_month=m,kind='old_derived',work='audit existing' if existed else 'reuse origin' if r['reused'] else 'calculate'))
        for n in dag['catboost_bases']:
            print('Preparing native',phase,m,n,flush=True);existed=cache_state(scores.model_paths(c,phase,m,n))
            r=scores.fit_model(c,fp,frame,base,native[n],m,phase,allow_missing);records.append(r)
            work.append(dict(candidate=n,eval_month=m,kind='native',work='audit existing' if existed else 'fit native recipe'))
            if allow_missing:write_json(Path(c['output_dir'])/(phase+'_progress.json'),dict(fingerprint=fp,work=work))
        for n in dag['new_blends']:
            existed=cache_state(paths(c,phase,m,n));r=new_derived(c,fp,n,m,phase,allow_missing);derived.append(r)
            work.append(dict(candidate=n,eval_month=m,kind='new_derived',work='audit existing' if existed else 'calculate'))
    return records,derived,work


def crossfit(c,h=None):
    h=history(c) if h is None else h;rows=[]
    for m,s,k in product([5,6],scores.strategies(),POLICIES):
        n=s['id'];aligned(h[n][m],h['baseline_balanced'][m]);rule=learn({t:d for t,d in h[n].items() if t<m},k,m)
        rows.append(dict(candidate=n,policy=k,eval_month=m,source_months=','.join(map(str,rule['source_months'])),parameters=json.dumps(rule,sort_keys=True),**measure(h[n][m],rule)))
    return pd.DataFrame(rows)


def score_diagnostic(n,m,d):
    v=np.asarray(d['scores']);unique=len(np.unique(v));old=scores.dependencies([])['old_bases']
    domain='probability-like original source' if n in old else 'finite margin'
    if not np.isfinite(v).all() or (n in old and ((v<0)|(v>1)).any()):raise ValueError('Score domain differs')
    return dict(candidate=n,eval_month=m,n=len(v),domain=domain,score_min=float(v.min()),score_max=float(v.max()),
        score_mean=float(v.mean()),unique_scores=unique,tied_fraction=(len(v)-unique)/len(v),
        outside_unit=int(((v<0)|(v>1)).sum()),**{'q'+str(q):float(np.quantile(v,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]})


def tables(c,frame,h=None):
    h=history(c,physical=True) if h is None else h
    if set(h)!=set(physical_names()) or any(set(v)!={4,5,6} for v in h.values()):raise ValueError('Complete physical score grid required')
    for n,m in product(h,[4,5,6]):aligned(h[n][m],expected_population(frame,m))
    cf=crossfit(c,h);summary=order_summary(cf).reset_index(drop=True);index=summary.set_index(['candidate','policy'])
    contrasts=[];pooled=[];diagnostics=[];diversity=[];overlap=[]
    for f in scores.contrast_factors():
        a,b=index.loc[(f['candidate'],f['policy'])],index.loc[(f['comparator'],f['comparator_policy'])]
        contrasts.append(dict(contrast=f['kind'],candidate=f['candidate'],policy=f['policy'],reference=f['comparator'],reference_policy=f['comparator_policy'],
            factor_details=json.dumps(f['factors'],separators=(',',':')),**{v+'_difference':float(a[v]-b[v]) for v in ['f2','recall','precision','flagged_fraction','average_precision']}))
    masks={}
    for row in cf.to_dict('records'):
        n,k,m=row['candidate'],row['policy'],row['eval_month'];masks[n,k,m]=decisions(h[n][m],json.loads(row['parameters']))
    for (n,k),group in cf.groupby(['candidate','policy']):
        t=group[['tn','fp','fn','tp']].sum()
        pooled.append(dict(candidate=n,policy=k,**confusion_metrics(*[int(t[v]) for v in ['tn','fp','fn','tp']]),flagged_fraction=float((t.tp+t.fp)/group.n.sum())))
    for n,m in product(h,[4,5,6]):diagnostics.append(score_diagnostic(n,m,h[n][m]))
    for s,m in product(scores.recipes(),[4,5,6]):
        x,y=h[s['id']][m]['scores'],h[scores.ANCHOR][m]['scores'];undefined=bool(np.ptp(x)==0 or np.ptp(y)==0)
        diversity.append(dict(candidate=s['id'],reference=scores.ANCHOR,eval_month=m,pearson=None if undefined else float(pearsonr(x,y).statistic),
            spearman=None if undefined else float(spearmanr(x,y).statistic),correlation_undefined=undefined,reason='constant member' if undefined else 'defined'))
    for s,k,m in product(scores.strategies(),POLICIES,[5,6]):
        if s['kind']=='reference':continue
        n=s['id'];refs=[scores.ANCHOR]+([scores.native(s['depth'],s['iterations'],s['positive_class_cost'])] if s['kind']=='blend' else [])
        for b in refs:
            a_mask,b_mask=masks[n,k,m],masks[b,k,m];y=h[n][m]['labels'].astype(bool);r=dict(candidate=n,policy=k,eval_month=m,reference=b)
            for tag,take in [('fraud',y),('legitimate',~y)]:
                for label,t in [('both_flagged',a_mask&b_mask),('first_only',a_mask&~b_mask),('second_only',~a_mask&b_mask),('neither',~a_mask&~b_mask)]:r[tag+'_'+label]=int(np.sum(take&t))
            overlap.append(r)
    diagnostic=pd.DataFrame(diagnostics)
    return dict(crossfit=cf,summary=summary,pooled=pd.DataFrame(pooled),contrasts=pd.DataFrame(contrasts),physical_score_diagnostics=diagnostic,
        score_diagnostics=diagnostic[diagnostic.candidate.isin([s['id'] for s in scores.strategies()])].reset_index(drop=True),diversity=pd.DataFrame(diversity),decision_overlap=pd.DataFrame(overlap))


def initialize(config_path=CONFIG):
    """Freeze the execution identity; never silently update it after code edits."""
    c = read_config(config_path); guard(c); fp = fingerprint(config_path)
    out = Path(c['output_dir']); preservation = out / 'preservation.json'
    expected = dict(mandatory=c['preserve_sha256'],
        optional_present=read_config(out / 'preimplementation_preservation.json'),
        roots=c['historical_output_roots'])
    verify_hashes(expected['optional_present'])
    if preservation.exists():
        if read_config(preservation) != expected:
            raise ValueError('Frozen preservation declaration changed')
    else:
        write_json(preservation, expected)
    record = dict(fingerprint=fp, config_path=str(config_path),
        protocol_sha256=digest(c['protocol_path']), preservation_sha256=digest(preservation),
        strategies=c['strategies'], expected_counts=c['expected_counts'], month7_previously_consumed=True)
    path = out / 'declaration.json'
    if path.exists():
        if any(read_config(path).get(k) != v for k, v in record.items()):
            raise ValueError('Frozen execution declaration changed')
    else:
        write_json(path, dict(**record, created_at_utc=now()))
    return c, fp


def check_receipt(c, fp):
    out = Path(c['output_dir']); dest = Path(c['report_dir'])
    r = read_config(out / 'checks.json')
    if r['fingerprint'] != fp or r['test_source_sha256'] != digest(TEST_SOURCE):
        raise ValueError('Current complete methodological checks required')
    if r['junit_sha256'] != digest(dest / 'tests.xml') or r['actual_fit_ledger_sha256'] != digest(dest / 'test_model_fits.json'):
        raise ValueError('Execution test evidence changed')
    verify_hashes(read_config(dest / 'test_model_fits.json')['execution_source_sha256'])
    return r


def record_checks(config_path=CONFIG):
    """Attest exact full-suite identities and observed fits before freezing code."""
    c = read_config(config_path); guard(c); out = Path(c['output_dir']); dest = Path(c['report_dir'])
    if (out / 'selection.json').exists():
        raise ValueError('Frozen check receipt is immutable')
    xml = dest / 'tests.xml'; root = ET.parse(xml).getroot(); cases = list(root.iter('testcase'))
    if not cases or any(list(root.iter(n)) for n in ['failure', 'error', 'skipped']):
        raise ValueError('Complete passing methodological suite required')
    own = [t.get('name', '') for t in cases if t.get('classname', '').endswith('test_shallow_catboost')]
    expected_formats = {'bagged_balanced','bagged_unweighted','spline','xgboost','lightgbm_balanced','lightgbm_unweighted'}
    formats = [n for n in own if n.startswith('test_six_reference_formats_through_new_destination[')]
    if len(formats) != 6 or {n.split('[')[-1].removesuffix(']') for n in formats} != expected_formats:
        raise ValueError('All six genuine reference-format fits required')
    if sum(n.startswith('test_twelve_native_formats_fit_audit_reuse_and_reject_corruption[') for n in own) != 12:
        raise ValueError('All twelve genuine native-format fits required')
    if sum(n.startswith('test_all_winner_policy_cases_preserve_roles_and_bounds[') for n in own) != 162:
        raise ValueError('All 162 winner/policy cases required')
    actual = [(t.get('classname', ''), t.get('name', '')) for t in cases]
    for path in sorted(Path('tests').glob('test_*.py')):
        prefix = 'tests.' + path.stem
        if not any(cls == prefix or cls.startswith(prefix + '.') for cls, name in actual):
            raise ValueError('Every methodological module required')
    lp = dest / 'test_model_fits.json'; ledger = read_config(lp)
    if ledger['exit_code'] != 0 or ledger['junit_sha256'] != digest(xml) or ledger['deselected_nodes']:
        raise ValueError('Complete actual-fit ledger required')
    expected = []
    for node in ledger['collected_nodes']:
        parts = node.split('::')
        expected.append(('.'.join([parts[0].removesuffix('.py').replace('/', '.'), *parts[1:-1]]), parts[-1]))
    if len(actual) != len(set(actual)) or len(expected) != len(set(expected)) or set(actual) != set(expected):
        raise ValueError('Exact collected/executed identities differ')
    sources=set(SOURCES+[CONFIG,c['protocol_path'],'scripts/check_shallow_catboost.py']+[
        str(p) for p in sorted(Path('tests').glob('test_*.py'))])
    if set(ledger['execution_source_sha256'])!=sources:
        raise ValueError('Complete execution source closure required')
    verify_hashes(ledger['execution_source_sha256'])
    current = [r for r in ledger['fits'] if r['nodeid'].startswith(TEST_SOURCE + '::')]
    if len(current) != 18 or any(r['status'] != 'completed' for r in current):
        raise ValueError('Exactly eighteen actual current-study tiny fits required')
    allowed = [TEST_SOURCE + '::test_' + s for s in [
        'twelve_native_formats_fit_audit_reuse_and_reject_corruption[',
        'six_reference_formats_through_new_destination[']]
    if any(not any(r['nodeid'].startswith(p) for p in allowed) for r in current):
        raise ValueError('Unexpected current-study tiny fit')
    if any(r['nodeid'] not in ledger['collected_nodes'] or r['status'] not in ['completed', 'failed'] for r in ledger['fits']):
        raise ValueError('Unattributed or unterminated classifier fit')
    historical = [r for r in ledger['fits'] if r not in current]
    receipt = dict(fingerprint=fingerprint(config_path), junit_sha256=digest(xml),
        tests_passed=len(cases), own_tests=len(own), actual_fit_ledger_sha256=digest(lp),
        actual_current_test_fits=18, actual_historical_test_fit_attempts=len(historical),
        actual_historical_test_fits=sum(r['status'] == 'completed' for r in historical),
        actual_historical_test_fit_failures=sum(r['status'] == 'failed' for r in historical),
        test_source_sha256=digest(TEST_SOURCE), created_at_utc=now())
    path = dest / 'test_fit_counts.json'
    invocations = read_config(path)['invocations'] if path.exists() else []
    if not any(r['junit_sha256'] == receipt['junit_sha256'] for r in invocations):
        invocations.append(receipt)
    write_json(path, dict(invocations=invocations,
        total_current_tiny_fits=sum(r['actual_current_test_fits'] for r in invocations)))
    write_json(out / 'checks.json', receipt)
    return receipt


import os
import subprocess

def source_miniature_gate(c,fp):
    return parent.source_miniature_gate(adapter_view(c),fp)


def miniature_data(columns):
    rng=np.random.default_rng(1026);n=600
    base=pd.DataFrame({k:(np.where(np.arange(n)%3==0,'a','b') if k in scores.CATEGORICAL else rng.normal(size=n)) for k in columns})
    base['intended_balcon_amount']=np.abs(base.intended_balcon_amount)
    signal=base['name_email_similarity'].to_numpy().copy()
    y=(signal+.3*rng.normal(size=n)>1.1).astype(np.int8)
    base.loc[np.arange(n)%23==0,'session_length_in_minutes']=np.nan
    base.loc[:479,'prev_address_months_count']=np.nan
    for k in scores.CATEGORICAL:base.loc[[0,480],k]=None;base.loc[481,k]='unseen'
    frame=base.assign(device_fraud_count=0,month=np.where(np.arange(n)<480,np.arange(n)//240,2),fraud_bool=y)
    return frame,base


def miniature(config_path=CONFIG):
    c,fp=initialize(config_path);check_receipt(c,fp);out=Path(c['output_dir'])
    for name in ['resource_preflight.json','source_miniature_gate.json','older_miniature_audit.json']:
        if read_config(out/name)['fingerprint']!=fp:raise ValueError('Resource/source gates required')
    path=out/'miniature_gate.json';failure=out/'failures'/'miniature.json'
    if path.exists():
        result=read_config(path)
        if result['fingerprint']!=fp or result['fit_count']!=14:raise ValueError('Miniature gate changed')
        verify_hashes(result['artifacts']);return result
    if failure.exists() or (out/'miniature_progress.json').exists():raise ValueError('Prior miniature attempt retained')
    columns=pd.read_csv(c['data_path'],nrows=0).drop(columns=['fraud_bool','month','device_fraud_count']).columns
    frame,base=miniature_data(columns);attempts=[];results=[];artifacts={};values={};start=time.perf_counter()
    try:
        recipes=scores.recipes();repeats=[s for s in recipes if s['id'] in ['cb_d4_r200_w1','cb_d6_r400_w100']]
        for i,spec in enumerate(recipes+repeats):
            tag=spec['id'] if i<12 else 'repeat_'+spec['id'];attempt=dict(tag=tag,status='started');attempts.append(attempt)
            write_json(out/'miniature_progress.json',dict(fingerprint=fp,attempts=attempts))
            local=dict(c,output_dir=str(out/'miniature'/tag));r=scores.fit_model(local,fp,frame,base,spec,2)
            attempt['status']='completed';ps=scores.model_paths(local,'validation',2,spec['id'])
            model=CatBoostClassifier();model.load_model(str(ps[2]));adapter=joblib.load(ps[3])['adapter']
            saved=prediction(local,'validation',2,spec['id']);ev=frame.index[frame.month==2];x=base.loc[ev]
            checks=[scores.margin(model,adapter,x.iloc[::-1])[::-1],
                np.concatenate([scores.margin(model,adapter,chunk) for chunk in [x.iloc[:19],x.iloc[19:]]]),
                scores.margin(model,adapter,pd.concat([x,x.iloc[:1].assign(name_email_similarity=999)]))[:len(x)],
                scores.margin(model,adapter,x.iloc[:1])]
            for value in checks:
                if not np.allclose(value,saved['scores'][:len(value)],atol=1e-12,rtol=0):raise ValueError('Native miniature batch/order invariance differs')
            changed=frame.copy();changed.loc[ev,'fraud_bool']=1-changed.loc[ev,'fraud_bool']
            if not np.allclose(scores.margin(model,adapter,features(changed).loc[ev]),saved['scores'],atol=1e-12,rtol=0):raise ValueError('Current-label invariance differs')
            values[tag]=saved['scores'];artifacts.update({str(p):digest(p) for p in ps});results.append(dict(tag=tag,spec=spec,record=r))
            write_json(out/'miniature_progress.json',dict(fingerprint=fp,attempts=attempts,results=results))
        for spec in repeats:
            if not np.allclose(values[spec['id']],values['repeat_'+spec['id']],atol=1e-12,rtol=0):raise ValueError('Native same-seed repeat differs')
        result=dict(fingerprint=fp,fit_count=len(attempts),attempts=attempts,results=results,artifacts=artifacts,
            same_seed_repeatable=True,inference_invariance=True,raw_columns=29,elapsed_seconds=time.perf_counter()-start)
        write_json(path,result);return result
    except Exception as exc:
        write_json(failure,dict(fingerprint=fp,error=repr(exc),attempts=attempts,artifacts=artifacts,completed_fits=sum(r['status']=='completed' for r in attempts),created_at_utc=now()));raise


def execution_gate(c,fp):
    check_receipt(c,fp);out=Path(c['output_dir'])
    for name in ['resource_preflight.json','source_miniature_gate.json','older_miniature_audit.json','miniature_gate.json']:
        record=read_config(out/name)
        if record['fingerprint']!=fp:raise ValueError('Missing/stale execution gate '+name)
        verify_hashes(record.get('artifacts',{}))
    mini=read_config(out/'miniature_gate.json')
    if mini['fit_count']!=14 or len(mini['attempts'])!=14 or any(r['status']!='completed' for r in mini['attempts']) or not mini['same_seed_repeatable'] or not mini['inference_invariance']:raise ValueError('Fourteen-fit native miniature gate required')


def work_accounting(c,records,derived,work):
    old={s['id']:s for s in required_sources(c)};refs=[r for r in records if r['candidate'] in old];new=[r for r in records if r['candidate'] not in old]
    def kind(r):
        s=old[r['candidate']];return 'bagged' if s['id'].startswith('bagged_') else 'spline' if s.get('representation')=='spline' else 'tree'
    olds=[r for r in derived if r['candidate'] in scores.dependencies([])['old_derived']]
    return dict(base_results=len(records),reference_base_results=len(refs),reference_reuses=sum(bool(r.get('reused')) for r in refs),
        **{'reproduced_'+k+'_fits':sum(kind(r)==k and not r.get('reused',False) for r in refs) for k in ['bagged','spline','tree']},
        native_model_results=len(new),derived_results=len(derived),reference_derived_results=len(olds),reference_derived_reuses=sum(bool(r['reused']) for r in olds),
        new_compositions=len(derived)-len(olds),native_train_row_rounds=sum(r['train_row_rounds'] for r in new),
        native_stored_trees=sum(r['fitted_tree_count'] for r in new),native_original_fit_seconds=sum(r['fit_seconds'] for r in new),
        invocation_new_native_fits=sum(r['work']=='fit native recipe' for r in work),invocation_exact_reference_fits=sum(r['work']=='fit exact recipe' for r in work),invocation_work=work)


def exact_artifact_grid(c,phase,names=None):
    names=[s['id'] for s in scores.strategies()] if names is None else names;dag=scores.dependencies(names)
    wanted_scores=set();wanted_models=set();old={s['id']:s for s in required_sources(c)};objects=0
    for m in [4,5,6] if phase=='validation' else [7]:
        wanted_scores.update(f'{phase}-m{m}-{n}' for n in dag['full_scores'])
        objects+=len(dag['old_bases'])+len(dag['catboost_bases'])
        for n in dag['old_bases']:wanted_models.update(p.name for p in parent.parent.source_paths(c,old[n],m,phase)[2:])
        for n in dag['catboost_bases']:wanted_models.update(p.name for p in scores.model_paths(c,phase,m,n)[2:])
    out=Path(c['output_dir'])
    for folder,suffix in [('runs','.json'),('predictions','.npz')]:
        if {p.stem for p in (out/folder).glob(phase+'-*'+suffix)}!=wanted_scores:raise ValueError('Exact score artifact grid differs: '+folder)
    if {p.name for p in (out/'models').glob(phase+'-*') if p.is_file()}!=wanted_models:raise ValueError('Exact model artifact grid differs')
    return dict(model_artifacts=objects,model_files=len(wanted_models),prediction_artifacts=len(wanted_scores))


def model_work(c,records):
    rows=[];params=[];pre=[]
    for r in records:
        is_native=r['candidate'].startswith('cb_d');n=r['candidate'];m=r['eval_month']
        row=dict(candidate=n,eval_month=m,kind='native' if is_native else 'reference',
            train_rows=r.get('train_rows'),train_fraud=r.get('train_fraud'),reused=bool(r.get('reused')),
            fitted_tree_count=r.get('fitted_tree_count'),train_row_rounds=r.get('train_row_rounds'),
            fit_seconds=r.get('fit_seconds'),peak_process_rss_bytes=r.get('peak_process_rss_bytes'),
            original_record=json.dumps(r,sort_keys=True))
        if is_native:
            row.update(model_bytes=r['model_bytes'],requested_rounds=r['spec']['iterations'],solver_iterations=None,
                resource_note='native fit measurements')
        else:
            spec=next(s for s in required_sources(c) if s['id']==n)
            ps=parent.parent.source_paths(c,spec,m,r['phase']);artifact=joblib.load(ps[-1]);solver=None
            if n=='xgb_d4_w1':
                from xgboost import XGBClassifier
                model=XGBClassifier();model.load_model(ps[2]);trees=int(model.get_booster().num_boosted_rounds());requested=r['fit_params']['n_estimators']
            else:
                model=artifact['model']
                if spec.get('representation')=='spline':
                    trees=0;requested=None;solver=int(np.max(model.n_iter_))
                else:
                    estimator=model.named_steps['model'];trees=estimator.booster_.current_iteration();requested=estimator.n_estimators
            row.update(fitted_tree_count=trees,requested_rounds=requested,solver_iterations=solver,
                train_row_rounds=None if solver is not None else r['train_rows']*trees,
                model_bytes=sum(p.stat().st_size for p in ps[2:]),
                resource_note='original fit measurements; missing historical RSS remains unavailable' if r.get('peak_process_rss_bytes') is None else 'original fit measurements')
        rows.append(row)
        if is_native:
            params.append(dict(candidate=n,eval_month=m,requested=r['requested_model_params'],effective=r['effective_model_params']))
            pre.append(dict(candidate=n,eval_month=m,preprocessing=r['preprocessing']))
    return pd.DataFrame(rows),params,pre


def prepare_development(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Development is frozen')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    records,derived,work=prepare(c,fp,frame,base);result=work_accounting(c,records,derived,work)
    if (result['base_results'],result['reference_base_results'],result['native_model_results'],result['derived_results'])!=(60,24,36,123):raise ValueError('Complete development DAG differs')
    grid=exact_artifact_grid(c,'validation')
    if grid!={'model_artifacts':60,'model_files':99,'prediction_artifacts':183}:raise ValueError('Development artifact accounting differs')
    rows,params,pre=model_work(c,records)
    if (len(rows),len(params),len(pre))!=(60,36,36):raise ValueError('Model work/preprocessing/parameter grid differs')
    rows.to_csv(out/WORK_TABLES[0],index=False);write_json(out/WORK_TABLES[1],params);write_json(out/WORK_TABLES[2],pre)
    result.update(fingerprint=fp,elapsed_seconds=time.perf_counter()-start,**grid,work_files={n:digest(out/n) for n in WORK_TABLES})
    path=out/'preparation.json'
    if path.exists():
        old=read_config(path)
        for k,v in result.items():
            if k not in ['invocation_work','invocation_new_native_fits','invocation_exact_reference_fits','elapsed_seconds'] and old[k]!=v:raise ValueError('Preparation accounting changed')
        return old
    write_json(path,result);return result


def validate(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists() or (out/'validation_complete.json').exists():raise ValueError('Completed validation cannot be overwritten')
    if read_config(out/'preparation.json')['fingerprint']!=fp:raise ValueError('Preparation missing/stale')
    frame=load_data(c);frame=frame[frame.month<7]
    generated=tables(c,frame)
    for name,table in generated.items():table.to_csv(out/(name+'.csv'),index=False)
    write_json(out/'validation_complete.json',dict(fingerprint=fp,created_at_utc=now(),files={n:digest(out/n) for n in TABLES+WORK_TABLES}))
    from .shallow_catboost_report import audit_development
    return audit_development(config_path)


def final_fit_bounds(c,choices):
    dag=scores.dependencies([n for n,k in choices.values()]);specs={s['id']:s for s in required_sources(c)}
    available={};missing=dict(bagged=0,spline=0,tree=0);view=adapter_view(c)
    for n in dag['old_bases']:
        spec=specs[n];found=None
        for source,fp in parent.parent.source_origins(view,spec):
            ps=parent.parent.source_paths(source,spec,7,'exploratory')
            if cache_state(ps):found={str(p):digest(p) for p in ps};break
        available[n]=found
        if found is None:missing['bagged' if n.startswith('bagged_') else 'spline' if spec.get('representation')=='spline' else 'tree']+=1
    return dict(unique_base_results=len(dag['old_bases'])+len(dag['catboost_bases']),physical_scores=len(dag['full_scores']),
        model_files=9+2*len(dag['catboost_bases']),max_new_native_fits=len(dag['catboost_bases']),
        **{'max_reproduced_'+k+'_fits':v for k,v in missing.items()},original_availability=available)


def freeze(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    done=read_config(out/'validation_complete.json');audit=read_config(out/'independent_development_audit.json')
    if done['fingerprint']!=fp or audit['fingerprint']!=fp or audit['validation_sha256']!=digest(out/'validation_complete.json') or audit['saved_model_checks']!=60 or audit['actual_source_models_audited'] is not True:raise ValueError('Complete independent prefreeze audit required')
    verify_hashes({str(out/n):h for n,h in done['files'].items()});w=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0]
    if (audit['winner'],audit['policy'])!=(str(w.candidate),str(w.policy)) or not np.isclose(audit['development_f2'],w.f2,atol=1e-12,rtol=0):raise ValueError('Independently audited winner differs')
    roles=scores.choices_for(str(w.candidate),str(w.policy));h=history(c)
    if list((out/'predictions').glob('exploratory-*')):raise ValueError('New consumed-month scores exist before freeze')
    record=dict(created_at_utc=now(),fingerprint=fp,config=c,selected_candidate=str(w.candidate),selected_policy=str(w.policy),selected_development_f2=float(w.f2),
        choices=roles,rules={a:learn(h[n],k,7) for a,(n,k) in roles.items()},score_dependencies=scores.dependencies([n for n,k in roles.values()]),
        final_fit_bounds=final_fit_bounds(c,roles),validation_files=done['files'],independent_audit_sha256=digest(out/'independent_development_audit.json'),
        artifacts={str(p):digest(p) for folder in ['runs','predictions','models'] for p in (out/folder).glob('validation-*') if p.is_file()},month7_previously_consumed=True,exploratory=True)
    write_json(out/'selection.json',record);return record


def load_selection(config_path=CONFIG):
    c,fp=initialize(config_path);out=Path(c['output_dir']);s=read_config(out/'selection.json')
    if s['fingerprint']!=fp or s['config']!=c:raise ValueError('Frozen execution identity changed')
    verify_hashes(s['artifacts']);verify_hashes({str(out/n):h for n,h in s['validation_files'].items()})
    if s['independent_audit_sha256']!=digest(out/'independent_development_audit.json'):raise ValueError('Frozen independent audit changed')
    w=order_summary(pd.read_csv(out/'crossfit.csv')).iloc[0];roles=scores.choices_for(str(w.candidate),str(w.policy));h=history(c)
    if (s['selected_candidate'],s['selected_policy'])!=(str(w.candidate),str(w.policy)) or not np.isclose(s['selected_development_f2'],w.f2,atol=1e-12,rtol=0):raise ValueError('Actual macro winner differs')
    if s['choices']!=roles or s['score_dependencies']!=scores.dependencies([n for n,k in roles.values()]) or s['final_fit_bounds']!=final_fit_bounds(c,roles):raise ValueError('Frozen roles/DAG/bounds differ')
    if any(s['rules'][a]!=learn(h[n],k,7) for a,(n,k) in roles.items()):raise ValueError('Frozen earlier-only rules differ')
    return c,s


def exploratory(config_path=CONFIG):
    c,s=load_selection(config_path);fp=s['fingerprint'];execution_gate(c,fp);out=Path(c['output_dir']);sha=digest(out/'selection.json')
    if (out/'transfer_complete.json').exists():raise ValueError('Completed transfer cannot be overwritten')
    access=out/'exploratory_access.json'
    if not access.exists():write_json(access,dict(access_at_utc=now(),selection_sha256=sha,month7_previously_consumed=True))
    r=read_config(access)
    if r['selection_sha256']!=sha or not s['created_at_utc']<r['access_at_utc']:raise ValueError('Freeze/access chronology differs')
    frame=load_data(c);base=features(frame);names=sorted({n for n,k in s['choices'].values()});records,derived,work=prepare(c,fp,frame,base,'exploratory',names)
    rows=[]
    for a,(n,k) in s['choices'].items():
        d=prediction(c,'exploratory',7,n);rule=s['rules'][a]
        rows.append(dict(role=a,candidate=n,policy=k,eval_month=7,parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    counts=work_accounting(c,records,derived,work);bounds=s['final_fit_bounds'];grid=exact_artifact_grid(c,'exploratory',names)
    if counts['native_model_results']>bounds['max_new_native_fits'] or counts['base_results']!=bounds['unique_base_results'] or grid['prediction_artifacts']!=bounds['physical_scores'] or grid['model_files']!=bounds['model_files']:raise ValueError('Frozen transfer work bounds exceeded')
    for kind in ['bagged','spline','tree']:
        if counts['reproduced_'+kind+'_fits']>bounds['max_reproduced_'+kind+'_fits']:raise ValueError('Frozen reference reproduction bound exceeded')
    pd.DataFrame(rows).to_csv(out/'exploratory.csv',index=False)
    result=dict(fingerprint=fp,selection_sha256=sha,table_sha256=digest(out/'exploratory.csv'),role_count=len(rows),**grid,**counts)
    write_json(out/'transfer_complete.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('step',choices=['verify_declaration','record_checks','preflight','miniature','prepare_development','validate','freeze','exploratory'])
    parser.add_argument('--config',default=CONFIG);args=parser.parse_args();print(json.dumps(globals()[args.step](args.config),indent=2))



def preflight(config_path=CONFIG):
    c, fp = initialize(config_path); check_receipt(c, fp)
    out = Path(c['output_dir']); path = out / 'resource_preflight.json'
    source_miniature_gate(c, fp)
    if path.exists():
        if read_config(path)['fingerprint'] != fp:
            raise ValueError('Resource gate identity changed')
        return read_config(path)
    failure = out / 'failures' / 'resource-preflight.json'
    if failure.exists():
        raise ValueError('Prior resource failure retained')
    start = time.perf_counter()
    try:
        from .spline_inputs import AdditiveInputs
        from .interaction_inputs import storage
        frame = load_data(c); frame = frame[frame.month < 7]; base = features(frame)
        additive = AdditiveInputs('spline').fit(base); sample = additive.transform(base.iloc[:8192])
        pre = make_model('lightgbm', False, dict(read_config('configs/study.json'), threads=c['threads']), base.columns).named_steps['preprocess'].fit(base)
        tree_sample = pre.transform(base.iloc[:8192]); n = len(base)
        d = max(len(additive.names), tree_sample.shape[1])
        matrix = 12 * n * d + 8 * (n + 1)
        source_workspace = 4 * matrix + 8 * (d + 1)**2 + 4 * 1024**3
        native_inputs = scores.NativeInputs(catboost=True).fit(base).transform(base)
        native_bytes = int(native_inputs.memory_usage(deep=True).sum())
        native_reserve = 4*native_bytes + 16*n*29 + 6*2**6*29*129*16
        workspace = source_workspace + native_reserve
        ram = (int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)) if sys.platform == 'darwin'
               else os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
        result = dict(fingerprint=fp, host_ram_bytes=ram, training_rows=n,
            training_months=list(range(7)), additive_dimension=len(additive.names),
            original_tree_dimension=tree_sample.shape[1], max_dimension=d, max_chunk_rows=8192,
            additive_sample=storage(sample), original_tree_sample=storage(tree_sample),
            matrix_upper_bytes=matrix, source_workspace_upper_bytes=source_workspace,
            native_input_deep_bytes=native_bytes, native_reserve_bytes=native_reserve,
            workspace_upper_bytes=workspace, allowance_fraction=.7,
            predictor_only=True, elapsed_seconds=time.perf_counter()-start,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if workspace > ram * .7:
            raise ValueError('Largest required allocation exceeds70% hostRAM')
        write_json(path, result); return result
    except Exception as exc:
        write_json(failure, dict(fingerprint=fp, error=repr(exc), created_at_utc=now())); raise


if __name__=='__main__':main()
