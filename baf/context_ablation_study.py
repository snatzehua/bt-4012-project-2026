"""Isolated context-ablation declaration, original sources and native-margin controls."""
import argparse
from itertools import combinations,product
import json
from pathlib import Path
import sys
import os
import subprocess
import time
import resource
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from scipy.special import expit

from . import shallow_catboost_study as parent
from . import context_ablation as scores
from .common import read_config,write_json,digest,load_data,features
from .focal_study import now
from .improvements import paths,prediction
from .recall_weighted_metrics import aligned,learn,measure,decisions,POLICIES
from .recall_extension import order_summary,expected_population
from .improvement_metrics import confusion_metrics

CONFIG='configs/context-feature-ablation.json'
CHANGE='test-context-feature-ablation-f2'
PLANNING='openspec/changes/'+CHANGE+'/planning-evidence.json'
SOURCES=parent.SOURCES+['baf/context_ablation.py','baf/context_ablation_study.py','baf/context_ablation_report.py']
OLD=parent.parent.parent
verify_hashes=parent.verify_hashes
cache_state=parent.cache_state


def planning_evidence_path():
    p=Path(PLANNING)
    if p.is_file():return p
    found=list(Path('openspec/changes/archive').glob('*-'+CHANGE+'/planning-evidence.json'))
    if len(found)!=1:raise ValueError('Exactly one original planning record required')
    return found[0]


def context_snapshot(name,entry):
    original=Path(entry['preserved_snapshot']);root=Path('openspec/changes')/CHANGE
    if original.is_file():result=original
    else:result=planning_evidence_path().parent/original.relative_to(root)
    if digest(result)!=entry['original_sha256']:raise ValueError('Historical project-context snapshot changed: '+name)
    return result


def optional_historical_entry(c,name):
    path=Path(name)
    return name not in c['preserve_sha256'] and any(path==Path(root) or Path(root) in path.parents for root in c['historical_output_roots'])


def verify_parent_indices(c):
    source=read_config(parent.CONFIG);dest=Path(source['report_dir']);mapping=c['project_context_snapshots']
    if set(mapping)!={'README.md','AGENTS.md','openspec/config.yaml'}:raise ValueError('Only three declared mutable context paths can be versioned')
    for file in ['report_hashes.json','run_manifest.json','closed_log_manifest.json']:
        for name,h in read_config(dest/file).items():
            if file=='run_manifest.json' and name in mapping:
                if mapping[name]['original_sha256']!=h:raise ValueError('Original context hash differs')
                q=context_snapshot(name,mapping[name])
            else:q=Path(name)
            if file=='run_manifest.json' and optional_historical_entry(c,name) and not q.is_file():continue
            if digest(q)!=h:raise ValueError('Original parent artifact changed: '+name)


def mandatory_preservation(source):
    result=dict(source['preserve_sha256'])
    for p in parent.SOURCES[-3:]+[parent.CONFIG,source['protocol_path'],str(Path(source['output_dir'])/'selection.json'),str(Path(source['report_dir'])/'report_hashes.json')]:result[p]=digest(p)
    result.update(read_config(Path(source['report_dir'])/'report_hashes.json'));return result


def declaration(source):
    p=read_config(planning_evidence_path())
    return dict(parent_config=parent.CONFIG,parent_fingerprint=parent.fingerprint(),
        reference_adapter_config=source['reference_adapter_config'],reference_adapter_fingerprint=source['reference_adapter_fingerprint'],
        preserve_sha256=mandatory_preservation(source),source_paths=SOURCES,strategies=scores.strategies(),
        native_recipes=scores.recipes(),reference_roles=scores.REFERENCE_ROLES,native_alphas=scores.ALPHAS,
        expected_counts=p['expected_counts'],representations=scores.REMOVED,project_context_snapshots=p['project_context_snapshots'],
        planning_evidence_path=PLANNING,planning_evidence_sha256=digest(planning_evidence_path()),
        output_dir='outputs/baf-context-feature-ablation-v1',report_dir='reports/baf-context-feature-ablation-v1',
        protocol_path='docs/context-feature-ablation-protocol.md',historical_output_roots=p['historical_output_roots'])


def guard(c):
    source=read_config(parent.CONFIG);parent.guard(source);expected=declaration(source)
    if set(c)!=set(source)|set(expected) or any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Prospective removal declaration changed')
    if any(c.get(k)!=v for k,v in source.items() if k not in expected):raise ValueError('Original source recipe changed')
    if len(SOURCES)!=64 or len(set(SOURCES))!=64 or len(c['preserve_sha256'])!=2020:raise ValueError('Scientific preservation closure changed')
    verify_hashes(c['preserve_sha256']);verify_parent_indices(c)
    pairs=[p for p in c['preserve_sha256'] if p.startswith('outputs/') and p.endswith('/selection.json')]
    if len(pairs)!=19 or any(c['preserve_sha256'].get(p.replace('outputs/','reports/',1))!=c['preserve_sha256'][p] for p in pairs):raise ValueError('Nineteen matching selections required')
    dest=[Path(c[k]).resolve() for k in ['output_dir','report_dir']]
    for a,b in combinations(dest+[Path(p).resolve() for p in c['historical_output_roots']],2):
        if (a in dest or b in dest) and (a==b or a in b.parents or b in a.parents):raise ValueError('Destination overlap')
    if any(k in c for k in ['primary_fraction','review_quota','precision_floor']):raise ValueError('Undeclared objective constraint')


def fingerprint(path=CONFIG):
    fp=parent.fingerprint(path);fp['sources']={p:digest(p) for p in SOURCES};return fp


def verify_declaration(config_path=CONFIG):
    from importlib.metadata import version
    c=read_config(config_path);guard(c);p=read_config(planning_evidence_path())
    if sys.version.split()[0]!=p['python'] or any(version(k)!=v for k,v in p['verified_pins'].items()):raise ValueError('Pinned software differs')
    if scores.recipes()!=p['recipes'] or scores.strategies()!=p['strategies'] or scores.contrast_factors()!=p['contrasts'] or scores.REMOVED!=p['representations']:
        raise ValueError('Independent prospective model/factor identities differ')
    if c['preserve_sha256']!=p['mandatory_preservation_sha256'] or SOURCES!=p['planned_source_paths']:raise ValueError('Prospective preserved sources differ')
    for case in p['winner_policy_cases']:
        roles=scores.choices_for(case['winner'],case['policy']);dag=scores.dependencies([n for n,k in roles.values()])
        if roles!=case['roles'] or dag!=case['dependencies'] or len(dag['new_ablation_bases'])!=case['new_fit_bound'] or len(dag['full_scores'])!=case['physical_score_bound']:
            raise ValueError('Prospective exact transfer case differs')
    # The prospective enumerator excludes generated Python bytecode, not source/model artifacts.
    actual={str(f) for root in c['historical_output_roots'] for f in Path(root).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    original=p['historical_ignored_file_sha256']
    if not actual<=set(original):raise ValueError('Undeclared historical scientific files appeared')
    present={name:original[name] for name in sorted(actual)};verify_hashes(present)
    out=Path(c['output_dir']);initial=out/'preimplementation_preservation.json'
    bytecode={str(f):digest(f) for root in c['historical_output_roots'] for f in Path(root).rglob('*') if f.is_file() and '__pycache__' in f.parts}
    bytecode_path=out/'historical_bytecode_preservation.json'
    if bytecode_path.exists():verify_hashes(read_config(bytecode_path))
    else:write_json(bytecode_path,bytecode)
    if initial.exists():verify_hashes(read_config(initial))
    else:write_json(initial,present)
    if digest(c['data_path'])!=c['data_sha256']:raise ValueError('Base checksum differs')
    result=dict(config_sha256=digest(config_path),protocol_sha256=digest(c['protocol_path']),planning_sha256=digest(planning_evidence_path()),
        planned_source_paths=SOURCES,mandatory_paths=2020,historical_original_files=5432,historical_present_files=len(present),selection_pairs=19,
        strategies=38,winner_policy_cases=114,contrasts=1192,parent_context_versions_verified=3,new_scientific_fits=0,new_score_evaluations=0)
    q=out/'contract_verification.json'
    if q.exists() and read_config(q)!=result:raise ValueError('Original prospective contract receipt changed')
    if not q.exists():write_json(q,result)
    return result


def adapter_view(c):
    return dict(c,parent_config=c['reference_adapter_config'],parent_fingerprint=c['reference_adapter_fingerprint'])


def required_sources(c):return parent.required_sources(adapter_view(c))


def prepare_source(c,fp,frame,base,spec,month,phase='validation',allow_missing=True):
    if spec not in required_sources(c):raise ValueError('Unrelated source forbidden')
    return OLD.prepare_source(adapter_view(c),fp,frame,base,spec,month,phase,allow_missing)


def derived_reference(c,fp,name,month,phase='validation',allow_missing=True):
    return parent.derived_reference(adapter_view(c),fp,name,month,phase,allow_missing)


def full_margin(c,fp,frame,base,name,month,phase='validation',allow_missing=True,*,miniature=False):
    spec=next((s for s in scores.recipes() if s['id']==name and not s['new_fit']),None)
    if spec is None:raise ValueError('Only a declared full native-margin control allowed')
    rp,pp=paths(c,phase,month,name);present=cache_state([rp,pp])
    if present:verify_hashes(read_config(rp)['source_artifacts'])
    elif not allow_missing:raise ValueError('Required full-native control absent')
    source=next(s for s in required_sources(c) if s['id']==spec['original_full_source'])
    original=prepare_source(c,fp,frame,base,source,month,phase,allow_missing)
    ps=OLD.source_paths(c,source,month,phase);pipe=joblib.load(ps[-1])['model'];x=scores.select(base,spec)
    tr=frame.index[frame.month<month];ev=frame.index[frame.month==month]
    pre=scores.preprocessing(pipe,x.loc[tr]);params=scores.parameters(pipe,spec,miniature=miniature)
    raw=scores.margins(pipe,x.loc[ev]);prob=prediction(c,phase,month,source['id'])
    actual=dict(row_ids=ev.to_numpy(),labels=frame.loc[ev,'fraud_bool'].to_numpy(),scores=raw);aligned(actual,prob)
    if not np.allclose(expit(raw),prob['scores'],atol=1e-12,rtol=0):raise ValueError('Original full probability/native inference differs')
    expected=dict(candidate=name,phase=phase,eval_month=month,fingerprint=fp,spec=spec,source_candidate=source['id'],
        source_artifacts={str(p):digest(p) for p in ps},preprocessing=pre,parameters=params,score_domain='finite native margin',score_only=True)
    if present:
        r=read_config(rp);saved=prediction(c,phase,month,name);aligned(actual,saved)
        if any(r.get(k)!=v for k,v in expected.items()) or digest(pp)!=r['prediction_sha256'] or not np.allclose(saved['scores'],raw,atol=1e-12,rtol=0):raise ValueError('Full native-control identity/inference differs')
        return r
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**actual)
    r=dict(**expected,prediction_sha256=digest(pp));write_json(rp,r);return r


def new_blend(c,fp,name,month,phase='validation',allow_missing=True):
    spec=scores.lookup(name)
    if spec['kind']!='blend':raise ValueError('Only a declared blend allowed')
    rp,pp=paths(c,phase,month,name);present=cache_state([rp,pp])
    if not present and not allow_missing:raise ValueError('Required context blend absent')
    members=[scores.ANCHOR,scores.pure(spec['representation'],spec['weighting'])]
    data=[prediction(c,phase,month,n) for n in members];actual=scores.blend(*data,spec['alpha_numerator']/100)
    expected=dict(candidate=name,phase=phase,eval_month=month,fingerprint=fp,spec=spec,
        dependencies={n:{str(p):digest(p) for p in paths(c,phase,month,n)} for n in members},arithmetic='(1-alpha)*anchor + alpha*native_margin',score_domain='finite native margin')
    if present:
        r=read_config(rp);saved=prediction(c,phase,month,name);aligned(actual,saved)
        if any(r.get(k)!=v for k,v in expected.items()) or digest(pp)!=r['prediction_sha256'] or not np.array_equal(saved['scores'],actual['scores']):raise ValueError('Context blend identity/arithmetic differs')
        return r
    pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**actual);r=dict(**expected,prediction_sha256=digest(pp));write_json(rp,r);return r


def source_miniature_gate(c,fp):
    return parent.source_miniature_gate(adapter_view(c),fp)


TABLES=[name+'.csv' for name in ['crossfit','summary','pooled','contrasts','physical_score_diagnostics','score_diagnostics','diversity','decision_overlap']]
WORK_TABLES=['model_work.csv','native_parameters.json','native_preprocessing.json','full_native_controls.json']


def physical_names():return scores.dependencies([s['id'] for s in scores.strategies()])['full_scores']


def history(c,*,physical=False):
    names=physical_names() if physical else [s['id'] for s in scores.strategies()]
    return {n:{m:prediction(c,'validation',m,n) for m in [4,5,6]} for n in names}


def crossfit(c,h=None):
    h=history(c) if h is None else h;rows=[]
    for m,s,k in product([5,6],scores.strategies(),POLICIES):
        n=s['id'];aligned(h[n][m],h['baseline_balanced'][m]);rule=learn({t:d for t,d in h[n].items() if t<m},k,m)
        rows.append(dict(candidate=n,policy=k,eval_month=m,source_months=','.join(map(str,rule['source_months'])),parameters=json.dumps(rule,sort_keys=True),**measure(h[n][m],rule)))
    return pd.DataFrame(rows)


def overlap_references(s):
    if s['kind']=='reference':return []
    refs=[scores.ANCHOR]
    if s['kind']=='pure':
        if s['representation']!='full':refs.append(scores.pure('full',s['weighting']))
    else:refs.extend([scores.pure(s['representation'],s['weighting']),scores.candidate('full',s['weighting'],s['alpha_numerator'])])
    return refs


def diversity_pairs():
    pairs=[(s['id'],scores.ANCHOR) for s in scores.recipes()]
    pairs.extend((s['id'],scores.pure('full',s['weighting'])) for s in scores.recipes() if s['new_fit'])
    return pairs


def contrast_terms(f):
    return f.get('terms',[dict(candidate=f['candidate'],policy=f['policy'],coefficient=1),dict(candidate=f['comparator'],policy=f['comparator_policy'],coefficient=-1)])


def tables(c,frame,h=None):
    h=history(c,physical=True) if h is None else h
    if set(h)!=set(physical_names()) or any(set(v)!={4,5,6} for v in h.values()):raise ValueError('Complete context physical score grid required')
    for n,m in product(h,[4,5,6]):aligned(h[n][m],expected_population(frame,m))
    cf=crossfit(c,h);summary=order_summary(cf).reset_index(drop=True);index=summary.set_index(['candidate','policy']);contrasts=[]
    for f in scores.contrast_factors():
        terms=contrast_terms(f)
        value={v+'_difference':float(sum(t['coefficient']*index.loc[(t['candidate'],t['policy']),v] for t in terms)) for v in ['f2','recall','precision','flagged_fraction','average_precision']}
        contrasts.append(dict(contrast=f['kind'],candidate=f['candidate'],policy=f['policy'],reference=f['comparator'],reference_policy=f['comparator_policy'],
            factor_details=json.dumps(f['factors'],separators=(',',':')),term_details=json.dumps(terms,separators=(',',':')),**value))
    masks={(r.candidate,r.policy,r.eval_month):decisions(h[r.candidate][r.eval_month],json.loads(r.parameters)) for r in cf.itertuples()}
    pooled=[]
    for (n,k),group in cf.groupby(['candidate','policy']):
        t=group[['tn','fp','fn','tp']].sum();pooled.append(dict(candidate=n,policy=k,**confusion_metrics(*[int(t[v]) for v in ['tn','fp','fn','tp']]),flagged_fraction=float((t.tp+t.fp)/group.n.sum())))
    diagnostic=pd.DataFrame([parent.score_diagnostic(n,m,h[n][m]) for n,m in product(h,[4,5,6])]);diversity=[]
    for (n,b),m in product(diversity_pairs(),[4,5,6]):
        x,y=h[n][m]['scores'],h[b][m]['scores'];constant=bool(np.ptp(x)==0 or np.ptp(y)==0)
        diversity.append(dict(candidate=n,reference=b,eval_month=m,pearson=None if constant else float(pearsonr(x,y).statistic),
            spearman=None if constant else float(spearmanr(x,y).statistic),correlation_undefined=constant,reason='constant member' if constant else 'defined'))
    overlap=[]
    for s,k,m in product(scores.strategies(),POLICIES,[5,6]):
        n=s['id']
        for b in overlap_references(s):
            a_mask,b_mask=masks[n,k,m],masks[b,k,m];y=h[n][m]['labels'].astype(bool);r=dict(candidate=n,policy=k,eval_month=m,reference=b)
            for tag,take in [('fraud',y),('legitimate',~y)]:
                for name,v in [('both_flagged',a_mask&b_mask),('first_only',a_mask&~b_mask),('second_only',~a_mask&b_mask),('neither',~a_mask&~b_mask)]:r[tag+'_'+name]=int(np.sum(take&v))
            overlap.append(r)
    return dict(crossfit=cf,summary=summary,pooled=pd.DataFrame(pooled),contrasts=pd.DataFrame(contrasts),physical_score_diagnostics=diagnostic,
        score_diagnostics=diagnostic[diagnostic.candidate.isin([s['id'] for s in scores.strategies()])].reset_index(drop=True),diversity=pd.DataFrame(diversity),decision_overlap=pd.DataFrame(overlap))


def prepare(c,fp,frame,base,phase='validation',names=None,allow_missing=True,fit_context=None):
    if phase not in ['validation','exploratory']:raise ValueError('Undeclared scientific phase')
    names=[s['id'] for s in scores.strategies()] if names is None else sorted(set(names));dag=scores.dependencies(names)
    specs={s['id']:s for s in required_sources(c)};new={s['id']:s for s in scores.recipes() if s['new_fit']};records=[];derived=[];work=[]
    for month in [4,5,6] if phase=='validation' else [7]:
        for n in dag['old_bases']:
            if fit_context is not None:fit_context.update(candidate=n,eval_month=month,kind='reference')
            print('Preparing original source',phase,month,n,flush=True);present=cache_state(OLD.source_paths(c,specs[n],month,phase))
            r=prepare_source(c,fp,frame,base,specs[n],month,phase,allow_missing);records.append(r)
            work.append(dict(candidate=n,eval_month=month,kind='reference',work='audit existing' if present else 'reuse origin' if r.get('reused') else 'fit exact recipe'))
        for n in OLD.derived_order([n for n,k in scores.REFERENCE_ROLES.values()]):
            present=cache_state(paths(c,phase,month,n));r=derived_reference(c,fp,n,month,phase,allow_missing);derived.append(r)
            work.append(dict(candidate=n,eval_month=month,kind='old_derived',work='audit existing' if present else 'reuse origin' if r['reused'] else 'calculate'))
        for n in dag['new_ablation_bases']:
            if fit_context is not None:fit_context.update(candidate=n,eval_month=month,kind='reduced')
            print('Preparing reduced model',phase,month,n,flush=True);present=cache_state(scores.model_paths(c,phase,month,n))
            r=scores.fit_model(c,fp,frame,base,new[n],month,phase,allow_missing);records.append(r)
            work.append(dict(candidate=n,eval_month=month,kind='reduced',work='audit existing' if present else 'fit reduced recipe'))
            if allow_missing:write_json(Path(c['output_dir'])/(phase+'_progress.json'),dict(fingerprint=fp,work=work))
        for n in dag['full_margin_controls']:
            present=cache_state(paths(c,phase,month,n));r=full_margin(c,fp,frame,base,n,month,phase,allow_missing);derived.append(r)
            work.append(dict(candidate=n,eval_month=month,kind='full_margin',work='audit existing' if present else 'calculate native full margin'))
        for n in dag['new_blends']:
            present=cache_state(paths(c,phase,month,n));r=new_blend(c,fp,n,month,phase,allow_missing);derived.append(r)
            work.append(dict(candidate=n,eval_month=month,kind='new_blend',work='audit existing' if present else 'calculate fixed blend'))
    return records,derived,work


def work_accounting(c,records,derived,work):
    old={s['id']:s for s in required_sources(c)};refs=[r for r in records if r['candidate'] in old];new=[r for r in records if r['candidate'] not in old]
    def kind(r):
        s=old[r['candidate']];return 'bagged' if s['id'].startswith('bagged_') else 'spline' if s.get('representation')=='spline' else 'tree'
    oldderived=[r for r in derived if r['candidate'] in scores.dependencies([])['old_derived']]
    full=[r for r in derived if r.get('score_only') is True];blends=[r for r in derived if r not in oldderived and r not in full]
    return dict(base_results=len(records),reference_base_results=len(refs),reference_reuses=sum(bool(r.get('reused')) for r in refs),
        **{'reproduced_'+k+'_fits':sum(kind(r)==k and not r.get('reused',False) for r in refs) for k in ['bagged','spline','tree']},
        reduced_model_results=len(new),derived_results=len(derived),reference_derived_results=len(oldderived),reference_derived_reuses=sum(bool(r['reused']) for r in oldderived),
        full_native_margin_controls=len(full),new_compositions=len(blends),reduced_train_row_rounds=sum(r['train_row_rounds'] for r in new),
        reduced_stored_trees=sum(r['fitted_tree_count'] for r in new),reduced_original_fit_seconds=sum(r['fit_seconds'] for r in new),
        invocation_new_reduced_fits=sum(r['work']=='fit reduced recipe' for r in work),invocation_exact_reference_fits=sum(r['work']=='fit exact recipe' for r in work),
        invocation_full_margin_calculations=sum(r['work']=='calculate native full margin' for r in work),invocation_blend_calculations=sum(r['work']=='calculate fixed blend' for r in work),invocation_work=work)


def exact_artifact_grid(c,phase,names=None):
    if phase not in ['validation','exploratory']:raise ValueError('Undeclared phase')
    names=[s['id'] for s in scores.strategies()] if names is None else names;dag=scores.dependencies(names);old={s['id']:s for s in required_sources(c)}
    wanted_scores=set();wanted_models=set();objects=0
    for month in [4,5,6] if phase=='validation' else [7]:
        wanted_scores.update(f'{phase}-m{month}-{n}' for n in dag['full_scores']);objects+=len(dag['old_bases'])+len(dag['new_ablation_bases'])
        for n in dag['old_bases']:wanted_models.update(p.name for p in OLD.source_paths(c,old[n],month,phase)[2:])
        for n in dag['new_ablation_bases']:wanted_models.add(scores.model_paths(c,phase,month,n)[-1].name)
    out=Path(c['output_dir'])
    for folder,suffix in [('runs','.json'),('predictions','.npz')]:
        if {p.stem for p in (out/folder).glob(phase+'-*'+suffix)}!=wanted_scores:raise ValueError('Exact context score artifact grid differs: '+folder)
    if {p.name for p in (out/'models').glob(phase+'-*') if p.is_file()}!=wanted_models:raise ValueError('Exact context model-file grid differs')
    return dict(model_artifacts=objects,model_files=len(wanted_models),prediction_artifacts=len(wanted_scores))


def model_work(c,records):
    new=[r for r in records if r['candidate'].startswith('context_')];old=[r for r in records if r not in new]
    references,_,_=parent.model_work(c,old);rows=references.to_dict('records');params=[];pre=[]
    for r in new:
        n,m=r['candidate'],r['eval_month'];rows.append(dict(candidate=n,eval_month=m,kind='reduced',train_rows=r['train_rows'],train_fraud=r['train_fraud'],reused=False,
            fitted_tree_count=r['fitted_tree_count'],train_row_rounds=r['train_row_rounds'],fit_seconds=r['fit_seconds'],peak_process_rss_bytes=r['peak_process_rss_bytes'],
            original_record=json.dumps(r,sort_keys=True),requested_rounds=200,solver_iterations=None,model_bytes=r['model_bytes'],resource_note='reduced native fit measurements; process high-water RSS'))
        params.append(dict(candidate=n,eval_month=m,requested=r['parameters']['requested'],effective=r['parameters']['effective'],actual_rounds=r['parameters']['actual_rounds']))
        pre.append(dict(candidate=n,eval_month=m,preprocessing=r['preprocessing']))
    return pd.DataFrame(rows),params,pre



TEST_SOURCE="tests/test_context_ablation.py"

def initialize(config_path=CONFIG):
    """Freeze the execution identity; never silently update it after code edits."""
    c = read_config(config_path); guard(c); fp = fingerprint(config_path)
    out = Path(c['output_dir']); preservation = out / 'preservation.json'
    expected = dict(mandatory=c['preserve_sha256'],
        optional_present=read_config(out / 'preimplementation_preservation.json'),
        roots=c['historical_output_roots'])
    verify_hashes(expected['optional_present']);verify_hashes(read_config(out/'historical_bytecode_preservation.json'))
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
    own = [t.get('name', '') for t in cases if t.get('classname', '').endswith('test_context_ablation')]
    expected_formats = {'bagged_balanced','bagged_unweighted','spline','xgboost','lightgbm_balanced','lightgbm_unweighted'}
    formats = [n for n in own if n.startswith('test_six_original_formats_and_full_native_controls[')]
    if len(formats) != 6 or {n.split('[')[-1].removesuffix(']') for n in formats} != expected_formats:
        raise ValueError('All six genuine reference-format fits required')
    if sum(n.startswith('test_six_reduced_formats_fit_inference_reuse_and_reject_corruption[') for n in own) != 6:
        raise ValueError('All twelve genuine native-format fits required')
    if sum(n.startswith('test_all_winner_policy_cases_preserve_exact_roles_work_and_dependencies[') for n in own) != 114:
        raise ValueError('All 114 winner/policy cases required')
    actual = [(t.get('classname', ''), t.get('name', '')) for t in cases]
    for path in sorted(Path('tests').glob('test_*.py')):
        prefix = 'tests.' + path.stem
        if not any(cls == prefix or cls.startswith(prefix + '.') for cls, name in actual):
            raise ValueError('Every methodological module required')
    lp = dest / 'test_model_fits.json'; ledger = read_config(lp)
    if ledger.get('scope')!='all' or ledger['exit_code'] != 0 or ledger['junit_sha256'] != digest(xml) or ledger['deselected_nodes']:
        raise ValueError('Complete actual-fit ledger required')
    expected = []
    for node in ledger['collected_nodes']:
        parts = node.split('::')
        expected.append(('.'.join([parts[0].removesuffix('.py').replace('/', '.'), *parts[1:-1]]), parts[-1]))
    if len(actual) != len(set(actual)) or len(expected) != len(set(expected)) or set(actual) != set(expected):
        raise ValueError('Exact collected/executed identities differ')
    sources=set(SOURCES+[CONFIG,c['protocol_path'],'scripts/check_context_ablation.py']+[
        str(p) for p in sorted(Path('tests').glob('test_*.py'))])
    if set(ledger['execution_source_sha256'])!=sources:
        raise ValueError('Complete execution source closure required')
    verify_hashes(ledger['execution_source_sha256'])
    current = [r for r in ledger['fits'] if r['nodeid'].startswith(TEST_SOURCE + '::')]
    if len(current) != 12 or any(r['status'] != 'completed' for r in current):
        raise ValueError('Exactly twelve actual current-study tiny fits required')
    allowed = [TEST_SOURCE + '::test_' + s for s in [
        'six_reduced_formats_fit_inference_reuse_and_reject_corruption[',
        'six_original_formats_and_full_native_controls[']]
    if any(not any(r['nodeid'].startswith(p) for p in allowed) for r in current):
        raise ValueError('Unexpected current-study tiny fit')
    if any(r['nodeid'] not in ledger['collected_nodes'] or r['status'] not in ['completed', 'failed'] for r in ledger['fits']):
        raise ValueError('Unattributed or unterminated classifier fit')
    historical = [r for r in ledger['fits'] if r not in current]
    receipt = dict(fingerprint=fingerprint(config_path), junit_sha256=digest(xml),
        tests_passed=len(cases), own_tests=len(own), actual_fit_ledger_sha256=digest(lp),
        actual_current_test_fits=12, actual_historical_test_fit_attempts=len(historical),
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

@contextmanager
def production_fit_accounting(out,phase,fp):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    directory=Path(out)/'fit_invocations';directory.mkdir(parents=True,exist_ok=True)
    path=directory/(phase+'-'+str(time.time_ns())+'.json');records=[];context={};depth=0;originals=[]
    def save():write_json(path,dict(fingerprint=fp,phase=phase,context=context,records=records,semantics='Outermost actual classifier fits; started/completed/failed calls retained, separate from inference.'))
    def wrapped(original,kind):
        @wraps(original)
        def fit(*args,**kwargs):
            nonlocal depth
            if depth:return original(*args,**kwargs)
            x=args[1] if len(args)>1 else kwargs.get('X');y=args[2] if len(args)>2 else kwargs.get('y')
            item=dict(**context,estimator=kind,status='started',started_at_utc=now(),rows=int(x.shape[0]),columns=int(x.shape[1]),fraud=int(np.sum(y)))
            records.append(item);save();depth+=1;start=time.perf_counter()
            try:result=original(*args,**kwargs);item['status']='completed';return result
            except Exception as exc:item.update(status='failed',error=repr(exc));raise
            finally:depth-=1;item['seconds']=time.perf_counter()-start;save()
        return fit
    for cls in [LGBMClassifier,XGBClassifier,CatBoostClassifier,LogisticRegression,RandomForestClassifier]:
        original=cls.fit;originals.append((cls,original));cls.fit=wrapped(original,cls.__name__)
    save()
    try:yield context,path
    finally:
        for cls,original in originals:cls.fit=original
        save()


@contextmanager
def forbid_classifier_fits():
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    originals=[]
    def blocked(*args,**kwargs):raise ValueError('Predictor/source preflight cannot fit classifiers')
    for cls in [LGBMClassifier,XGBClassifier,CatBoostClassifier,LogisticRegression,RandomForestClassifier]:originals.append((cls,cls.fit));cls.fit=blocked
    try:yield
    finally:
        for cls,original in originals:cls.fit=original


@contextmanager
def inference_accounting():
    """Count actual prediction API calls, including nested APIs; no fitted inputs stored."""
    from lightgbm import Booster as LGBBooster
    from xgboost import Booster as XGBBooster
    from sklearn.linear_model import LogisticRegression
    calls=[];originals=[]
    def wrapper(original,name):
        @wraps(original)
        def counted(*args,**kwargs):
            data=args[1] if len(args)>1 else kwargs.get('data',kwargs.get('X'))
            shape=getattr(data,'shape',None);rows=int(shape[0]) if shape is not None else int(data.num_row()) if hasattr(data,'num_row') else None
            item=dict(api=name,rows=rows,raw_score=bool(kwargs.get('raw_score',False)),status='started');calls.append(item);start=time.perf_counter()
            try:
                result=original(*args,**kwargs);item.update(status='completed',output_shape=list(np.asarray(result).shape));return result
            except Exception as exc:item.update(status='failed',error=repr(exc));raise
            finally:item['inclusive_seconds']=time.perf_counter()-start
        return counted
    for cls,names in [(LGBBooster,['predict']),(XGBBooster,['predict','inplace_predict']),(LogisticRegression,['predict_proba','decision_function'])]:
        for name in names:
            original=getattr(cls,name);originals.append((cls,name,original));setattr(cls,name,wrapper(original,cls.__name__+'.'+name))
    try:yield calls
    finally:
        for cls,name,original in originals:setattr(cls,name,original)


def preflight(config_path=CONFIG):
    c,fp=initialize(config_path);check_receipt(c,fp);out=Path(c['output_dir'])
    with forbid_classifier_fits():source_miniature_gate(c,fp)
    path=out/'resource_preflight.json'
    if path.exists():
        result=read_config(path)
        if result['fingerprint']!=fp:raise ValueError('Frozen resource gate changed')
        return result
    start=time.perf_counter();frame=load_data(c);frame=frame[frame.month<7];base=features(frame);n=len(base)
    from .spline_inputs import AdditiveInputs
    from .engineering import NativeInputs
    source_dimensions=[]
    # Predictor-only fitted representations; classifier fitting is forbidden here.
    for adapter in [AdditiveInputs(),NativeInputs(catboost=False)]:
        adapter.fit(base)
        if isinstance(adapter,AdditiveInputs):width=adapter.transform(base.iloc[:1]).shape[1]
        else:
            pre=scores.constructor(next(s for s in scores.recipes() if s['representation']=='full')).named_steps['preprocess'];pre.fit(base);width=len(pre.get_feature_names_out())
        source_dimensions.append(int(width))
    dimensions={}
    for spec in scores.recipes():
        x=scores.select(base,spec);pre=scores.constructor(spec).named_steps['preprocess'];pre.fit(x);dimensions[spec['id']]=len(pre.get_feature_names_out())
    ds=max(source_dimensions);dn=max(dimensions.values());csr=lambda d:12*n*d+8*(n+1)
    source_workspace=4*csr(ds)+8*(ds+1)**2+4*1024**3;new_workspace=4*csr(dn);score_buffers=64*n;estimated=source_workspace+new_workspace+score_buffers
    if sys.platform=='darwin':ram=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
    else:ram=os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    result=dict(fingerprint=fp,rows=n,source_dimensions=source_dimensions,new_encoded_dimensions=dimensions,d_source=ds,d_new=dn,
        source_workspace_bytes=source_workspace,new_workspace_bytes=new_workspace,score_buffers_bytes=score_buffers,estimated_bytes=estimated,host_ram_bytes=ram,
        limit_fraction=.7,input_deep_bytes=int(base.memory_usage(deep=True).sum()),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        feasible=estimated<=.7*ram,elapsed_seconds=time.perf_counter()-start,interpretation='Conservative feasibility proxy, not a guaranteed allocation peak.')
    if n!=903157 or not result['feasible']:
        write_json(out/'failures'/'resource_preflight.json',result);raise ValueError('Declared actual-host resource gate failed')
    write_json(path,result);return result


def miniature(config_path=CONFIG):
    c,fp=initialize(config_path);check_receipt(c,fp);out=Path(c['output_dir']);path=out/'miniature_gate.json';failure=out/'failures'/'miniature.json'
    for name in ['resource_preflight.json','source_miniature_gate.json','older_miniature_audit.json']:
        if read_config(out/name)['fingerprint']!=fp:raise ValueError('Resource/source gate required')
    if path.exists():
        result=read_config(path)
        if result['fingerprint']!=fp or result['fit_count']!=8:raise ValueError('Frozen miniature changed')
        verify_hashes(result['artifacts']);return result
    if failure.exists() or (out/'miniature_progress.json').exists():raise ValueError('Prior miniature attempt retained')
    frame,base=parent.miniature_data(scores.RAW_COLUMNS);specs=[s for s in scores.recipes() if s['new_fit']]
    repeats=[s for s in specs if s['id'] in ['context_drop_both_balanced','context_drop_velocity_unweighted']]
    attempts=[];results=[];artifacts={};values={};start=time.perf_counter()
    try:
        from .context_ablation_report import audit_reduced
        for i,spec in enumerate(specs+repeats):
            tag=spec['id'] if i<6 else 'repeat_'+spec['id'];attempt=dict(tag=tag,status='started');attempts.append(attempt)
            write_json(out/'miniature_progress.json',dict(fingerprint=fp,attempts=attempts));local=dict(c,output_dir=str(out/'miniature'/tag))
            r=scores.fit_model(local,fp,frame,base,spec,2,miniature=True);attempt['status']='completed';audit_reduced(local,fp,frame,base,r)
            ps=scores.model_paths(local,'validation',2,spec['id']);pipe=joblib.load(ps[-1])['model'];x=scores.select(base,spec);ev=frame.index[frame.month==2];saved=prediction(local,'validation',2,spec['id'])
            checks=[scores.margins(pipe,x.loc[ev].iloc[::-1])[::-1],np.concatenate([scores.margins(pipe,x.loc[ev].iloc[:19]),scores.margins(pipe,x.loc[ev].iloc[19:])]),scores.margins(pipe,x.loc[ev].iloc[:1]),scores.margins(pipe,pd.concat([x.loc[ev],x.loc[ev].iloc[:1].assign(name_email_similarity=999)]))[:len(ev)]]
            changed=frame.copy();changed.loc[ev,'fraud_bool']=1-changed.loc[ev,'fraud_bool'];checks.append(scores.margins(pipe,scores.select(features(changed),spec).loc[ev]))
            changed=base.copy();changed[spec['removed_columns']]=999999;checks.append(scores.margins(pipe,scores.select(changed,spec).loc[ev]))
            for actual in checks:
                if not np.allclose(actual,saved['scores'][:len(actual)],atol=1e-12,rtol=0):raise ValueError('Miniature inference invariance failed')
            values[tag]=saved['scores'];artifacts.update({str(p):digest(p) for p in ps});results.append(dict(tag=tag,spec=spec,record=r));write_json(out/'miniature_progress.json',dict(fingerprint=fp,attempts=attempts,results=results))
        for spec in repeats:
            if not np.allclose(values[spec['id']],values['repeat_'+spec['id']],atol=1e-12,rtol=0):raise ValueError('Same-seed miniature repeat failed')
        result=dict(fingerprint=fp,fit_count=8,attempts=attempts,results=results,artifacts=artifacts,same_seed_repeatable=True,inference_invariance=True,raw_columns=29,elapsed_seconds=time.perf_counter()-start)
        write_json(path,result);return result
    except Exception as exc:
        write_json(failure,dict(fingerprint=fp,error=repr(exc),attempts=attempts,artifacts=artifacts,completed_fits=sum(x['status']=='completed' for x in attempts),created_at_utc=now()));raise


def execution_gate(c,fp):
    check_receipt(c,fp);out=Path(c['output_dir'])
    for name in ['resource_preflight.json','source_miniature_gate.json','older_miniature_audit.json','miniature_gate.json']:
        record=read_config(out/name)
        if record['fingerprint']!=fp:raise ValueError('Missing/stale execution gate '+name)
        verify_hashes(record.get('artifacts',{}))
    mini=read_config(out/'miniature_gate.json')
    if mini['fit_count']!=8 or len(mini['attempts'])!=8 or any(r['status']!='completed' for r in mini['attempts']) or not mini['same_seed_repeatable'] or not mini['inference_invariance']:raise ValueError('Eight-fit reduced miniature gate required')

def prepare_development(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Development is frozen')
    if (out/'preparation.json').exists():raise ValueError('Completed preparation cannot be overwritten')
    frame=load_data(c);frame=frame[frame.month<7];base=features(frame);start=time.perf_counter()
    with production_fit_accounting(out,'validation',fp) as (fit_context,fit_ledger),inference_accounting() as calls:records,derived,work=prepare(c,fp,frame,base,fit_context=fit_context)
    write_json(out/'development_inference_calls.json',dict(fit_ledger=str(fit_ledger),fit_ledger_sha256=digest(fit_ledger),calls=calls,semantics='Actual instrumented prediction APIs, including nested calls; inclusive timing is not summed runtime.'))
    result=work_accounting(c,records,derived,work)
    if (result['base_results'],result['reference_base_results'],result['reduced_model_results'],result['derived_results'])!=(42,24,18,93):raise ValueError('Complete development DAG differs')
    grid=exact_artifact_grid(c,'validation')
    if grid!={'model_artifacts':42,'model_files':45,'prediction_artifacts':135}:raise ValueError('Development artifact accounting differs')
    rows,params,pre=model_work(c,records)
    if (len(rows),len(params),len(pre))!=(42,18,18):raise ValueError('Model work/preprocessing/parameter grid differs')
    rows.to_csv(out/WORK_TABLES[0],index=False);write_json(out/WORK_TABLES[1],params);write_json(out/WORK_TABLES[2],pre);write_json(out/WORK_TABLES[3],[r for r in derived if r.get('score_only') is True])
    result.update(fingerprint=fp,elapsed_seconds=time.perf_counter()-start,**grid,work_files={n:digest(out/n) for n in WORK_TABLES})
    path=out/'preparation.json'
    if path.exists():
        old=read_config(path)
        for k,v in result.items():
            if not k.startswith('invocation_') and k!='elapsed_seconds' and old[k]!=v:raise ValueError('Preparation accounting changed')
        return old
    write_json(path,result);return result

def validate(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists() or (out/'validation_complete.json').exists():raise ValueError('Completed validation cannot be overwritten')
    if read_config(out/'preparation.json')['fingerprint']!=fp:raise ValueError('Preparation missing/stale')
    frame=load_data(c);frame=frame[frame.month<7]
    generated=tables(c,frame)
    for name,table in generated.items():table.to_csv(out/(name+'.csv'),index=False)
    write_json(out/'validation_complete.json',dict(fingerprint=fp,created_at_utc=now(),files={n:digest(out/n) for n in TABLES+WORK_TABLES+['development_inference_calls.json']}))
    from .context_ablation_report import audit_development
    return audit_development(config_path)

def final_fit_bounds(c,choices):
    dag=scores.dependencies([n for n,k in choices.values()]);specs={s['id']:s for s in required_sources(c)};available={};missing=dict(bagged=0,spline=0,tree=0);view=adapter_view(c)
    for n in dag['old_bases']:
        spec=specs[n];found=None
        origins=[(c,fingerprint())]+list(OLD.source_origins(view,spec))
        for source,fp in origins:
            ps=OLD.source_paths(source,spec,7,'exploratory')
            if cache_state(ps):found={str(p):digest(p) for p in ps};break
        available[n]=found
        if found is None:missing['bagged' if n.startswith('bagged_') else 'spline' if spec.get('representation')=='spline' else 'tree']+=1
    own={n:({str(p):digest(p) for p in scores.model_paths(c,'exploratory',7,n)} if cache_state(scores.model_paths(c,'exploratory',7,n)) else None) for n in dag['new_ablation_bases']}
    return dict(unique_base_results=8+len(dag['new_ablation_bases']),physical_scores=len(dag['full_scores']),model_files=9+len(dag['new_ablation_bases']),
        required_reduced_models=len(dag['new_ablation_bases']),max_new_reduced_fits=sum(v is None for v in own.values()),
        **{'max_reproduced_'+k+'_fits':v for k,v in missing.items()},original_availability=available,reduced_availability=own)


def freeze(config_path=CONFIG):
    c,fp=initialize(config_path);execution_gate(c,fp);out=Path(c['output_dir'])
    if (out/'selection.json').exists():raise ValueError('Selection is immutable')
    done=read_config(out/'validation_complete.json');audit=read_config(out/'independent_development_audit.json')
    if done['fingerprint']!=fp or audit['fingerprint']!=fp or audit['validation_sha256']!=digest(out/'validation_complete.json') or audit['saved_model_checks']!=42 or audit['actual_source_models_audited'] is not True:raise ValueError('Complete independent prefreeze audit required')
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
    frame=load_data(c);base=features(frame);names=sorted({n for n,k in s['choices'].values()});
    with production_fit_accounting(out,'exploratory',fp) as (fit_context,fit_ledger),inference_accounting() as calls:records,derived,work=prepare(c,fp,frame,base,'exploratory',names,fit_context=fit_context)
    write_json(out/'transfer_inference_calls.json',dict(fit_ledger=str(fit_ledger),fit_ledger_sha256=digest(fit_ledger),calls=calls,semantics='Actual prediction API calls including nested calls.'))
    rows=[]
    for a,(n,k) in s['choices'].items():
        d=prediction(c,'exploratory',7,n);rule=s['rules'][a]
        rows.append(dict(role=a,candidate=n,policy=k,eval_month=7,parameters=json.dumps(rule,sort_keys=True),**measure(d,rule)))
    counts=work_accounting(c,records,derived,work);bounds=s['final_fit_bounds'];grid=exact_artifact_grid(c,'exploratory',names)
    if counts['reduced_model_results']!=bounds['required_reduced_models'] or counts['invocation_new_reduced_fits']>bounds['max_new_reduced_fits'] or counts['base_results']!=bounds['unique_base_results'] or grid['prediction_artifacts']!=bounds['physical_scores'] or grid['model_files']!=bounds['model_files']:raise ValueError('Frozen transfer work bounds exceeded')
    for kind in ['bagged','spline','tree']:
        if counts['reproduced_'+kind+'_fits']>bounds['max_reproduced_'+kind+'_fits']:raise ValueError('Frozen reference reproduction bound exceeded')
    pd.DataFrame(rows).to_csv(out/'exploratory.csv',index=False)
    result=dict(fingerprint=fp,selection_sha256=sha,table_sha256=digest(out/'exploratory.csv'),role_count=len(rows),**grid,**counts)
    write_json(out/'transfer_complete.json',result);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('step',choices=['verify_declaration','record_checks','preflight','miniature','prepare_development','validate','freeze','exploratory']);parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args();print(json.dumps(globals()[args.step](args.config),indent=2))


if __name__=='__main__':main()
