"""Isolated context-ablation declaration, original sources and native-margin controls."""
import argparse
from itertools import combinations,product
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from scipy.special import expit

from . import shallow_catboost_study as parent
from . import context_ablation as scores
from .common import read_config,write_json,digest
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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('step',choices=['verify_declaration']);parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args();print(json.dumps(globals()[args.step](args.config),indent=2))


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


def prepare(c,fp,frame,base,phase='validation',names=None,allow_missing=True):
    if phase not in ['validation','exploratory']:raise ValueError('Undeclared scientific phase')
    names=[s['id'] for s in scores.strategies()] if names is None else sorted(set(names));dag=scores.dependencies(names)
    specs={s['id']:s for s in required_sources(c)};new={s['id']:s for s in scores.recipes() if s['new_fit']};records=[];derived=[];work=[]
    for month in [4,5,6] if phase=='validation' else [7]:
        for n in dag['old_bases']:
            print('Preparing original source',phase,month,n,flush=True);present=cache_state(OLD.source_paths(c,specs[n],month,phase))
            r=prepare_source(c,fp,frame,base,specs[n],month,phase,allow_missing);records.append(r)
            work.append(dict(candidate=n,eval_month=month,kind='reference',work='audit existing' if present else 'reuse origin' if r.get('reused') else 'fit exact recipe'))
        for n in OLD.derived_order([n for n,k in scores.REFERENCE_ROLES.values()]):
            present=cache_state(paths(c,phase,month,n));r=derived_reference(c,fp,n,month,phase,allow_missing);derived.append(r)
            work.append(dict(candidate=n,eval_month=month,kind='old_derived',work='audit existing' if present else 'reuse origin' if r['reused'] else 'calculate'))
        for n in dag['new_ablation_bases']:
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


if __name__=='__main__':main()
