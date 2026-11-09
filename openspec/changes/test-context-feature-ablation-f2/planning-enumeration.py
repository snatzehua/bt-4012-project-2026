"""Read-only ancestry/cache inspection and finite planning enumeration; no fitting/inference."""
from itertools import product,combinations
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path.cwd()))
from importlib.metadata import version
from baf.common import read_config,digest,write_json
from baf import shallow_catboost_study as parent

ROOT=Path('openspec/changes/test-context-feature-ablation-f2')
if not ROOT.exists():
    found=list(Path('openspec/changes/archive').glob('*-test-context-feature-ablation-f2'))
    assert len(found)==1;ROOT=found[0]
c=read_config(parent.CONFIG)
refs=parent.scores.REFERENCE_ROLES
rep={'full':[], 'drop_velocity':['velocity_6h','velocity_24h','velocity_4w'], 'drop_counts':['zip_count_4w','bank_branch_count_8w','date_of_birth_distinct_emails_4w']}
rep['drop_both']=rep['drop_velocity']+rep['drop_counts']
weights=['balanced','unweighted'];policies=['pooled','latest','rank_fraction'];alphas=[10,25,50]
full={'balanced':'baseline_balanced','unweighted':'baseline_lgbm'}
def pure(r,w):return f'context_{r}_{w}'
def blend(r,w,a):return f'context_anchor_{r}_{w}_a{a:02d}'
recipes=[dict(id=pure(r,w),representation=r,weighting=w,removed_columns=rep[r],raw_columns=29-len(rep[r]),new_fit=r!='full',original_full_source=full[w]) for r,w in product(rep,weights)]
strategies=[dict(id=n,kind='reference') for n,k in refs.values()]
strategies += [dict(id=pure(r,w),kind='pure',representation=r,weighting=w) for r,w in product(rep,weights)]
strategies += [dict(id=blend(r,w,a),kind='blend',representation=r,weighting=w,alpha_numerator=a) for r,w,a in product(rep,weights,alphas)]
lookup={s['id']:s for s in strategies};assert len(lookup)==len(strategies)==38
oldplan=read_config(parent.planning_evidence_path());oldcase=next(x for x in oldplan['winner_policy_cases'] if x['winner']==refs['bagged_anchor'][0] and x['policy']=='pooled')
oldbases=oldcase['dependencies']['old_bases'];oldderived=oldcase['dependencies']['old_derived']
assert len(oldbases)==8 and len(oldderived)==5
anchor=refs['bagged_anchor'][0]
def at(r,w,a):return pure(r,w) if a==100 else blend(r,w,a)
def choices(n,k):
 s=lookup[n];roles={'selected':[n,k],**{a:list(v) for a,v in refs.items()},'same_strategy':[n,'pooled']}
 if s['kind']=='reference':return roles
 r,w=s['representation'],s['weighting'];a=s.get('alpha_numerator',100)
 roles.update({f'representation_{q}':[at(q,w,a),k] for q in rep})
 roles.update({f'weight_{q}':[at(r,q,a),k] for q in weights})
 roles.update({f'alpha_a{q:03d}':[anchor if q==0 else at(r,w,q),k] for q in [0,10,25,50,100]})
 assert len(roles)==19;return roles
def dag(names):
 requested=set(names);raw=set();bs=set()
 for n in requested:
  s=lookup[n]
  if s['kind']=='reference':continue
  raw.add(pure(s['representation'],s['weighting']))
  if s['kind']=='blend':bs.add(n)
 fm=sorted(n for n in raw if lookup[n]['representation']=='full');new=sorted(raw-set(fm))
 return dict(old_bases=oldbases,old_derived=oldderived,full_margin_controls=fm,new_ablation_bases=new,new_blends=sorted(bs),full_scores=sorted(set(oldbases+oldderived)|raw|bs))
cases=[]
for s,k in product(strategies,policies):
 roles=choices(s['id'],k);d=dag([n for n,p in roles.values()]);nf=len(d['new_ablation_bases'])
 assert nf==(0 if s['kind']=='reference' else 3 if s['representation']=='full' else 4)
 assert len(d['full_scores'])=={'reference':13,'pure':21,'blend':25}[s['kind']]
 cases.append(dict(winner=s['id'],policy=k,roles=roles,dependencies=d,new_fit_bound=nf,physical_score_bound=len(d['full_scores']),model_object_bound=8+nf,model_file_bound=9+nf,unique_operating_pairs=len({tuple(v) for v in roles.values()}),paired_interval_rows=4*(len(roles)-1)))
all_dag=dag(list(lookup));assert len(all_dag['full_scores'])==45 and len(all_dag['new_ablation_bases'])==6
# Exact factor identities, including four-term performance-removal interactions.
contrasts=[]
def add(kind,n,k,b,bk,factors,terms=None):
 row=dict(kind=kind,candidate=n,policy=k,comparator=b,comparator_policy=bk,factors=factors)
 if terms is not None:row['terms']=terms
 contrasts.append(row)
for n,k in product(lookup,policies):
 for role,(b,bk) in refs.items():add('fixed_reference',n,k,b,bk,dict(reference_role=role))
 if k!='pooled':add('own_policy',n,k,n,'pooled',{})
for r,w,a,k in product(rep,weights,[10,25,50,100],policies):
 n=at(r,w,a)
 if r in ['full','drop_counts']:
  b='drop_velocity' if r=='full' else 'drop_both';add('remove_velocity',at(b,w,a),k,n,k,dict(counts_removed=r=='drop_counts',weighting=w,alpha=a))
 if r in ['full','drop_velocity']:
  b='drop_counts' if r=='full' else 'drop_both';add('remove_counts',at(b,w,a),k,n,k,dict(velocity_removed=r=='drop_velocity',weighting=w,alpha=a))
 if r=='full':
  add('remove_both',at('drop_both',w,a),k,n,k,dict(weighting=w,alpha=a))
  add('removal_interaction',at('drop_both',w,a),k,n,k,dict(weighting=w,alpha=a),terms=[dict(candidate=at(q,w,a),policy=k,coefficient=z) for q,z in [('drop_both',1),('drop_velocity',-1),('drop_counts',-1),('full',1)]])
 if w=='unweighted':add('balanced_minus_unweighted',at(r,'balanced',a),k,n,k,dict(representation=r,alpha=a))
 if a==100:add('pure_vs_anchor',n,k,anchor,k,dict(representation=r,weighting=w))
 else:
  add('blend_vs_pure',n,k,pure(r,w),k,dict(representation=r,weighting=w,alpha=a))
  add('blend_vs_anchor',n,k,anchor,k,dict(representation=r,weighting=w,alpha=a))
for r,w,k,(a,b) in product(rep,weights,policies,combinations(alphas,2)):
 add('higher_alpha',blend(r,w,b),k,blend(r,w,a),k,dict(representation=r,weighting=w,alpha_first=b,alpha_second=a))
keys=[json.dumps(x,sort_keys=True) for x in contrasts];assert len(set(keys))==len(keys)==1192
parent_manifest=Path(c['report_dir'])/'report_hashes.json';mandatory=dict(c['preserve_sha256'])
for q in parent.SOURCES[-3:]+[parent.CONFIG,c['protocol_path'],str(Path(c['output_dir'])/'selection.json'),str(parent_manifest)]:mandatory[q]=digest(q)
mandatory.update(read_config(parent_manifest))
assert len(mandatory)==2020
for q,h in mandatory.items():assert digest(q)==h,q
pairs=[q for q in mandatory if q.startswith('outputs/') and q.endswith('/selection.json')];assert len(pairs)==19
assert all(mandatory[q]==mandatory[q.replace('outputs/','reports/',1)] for q in pairs)
history_roots=sorted(set(c['historical_output_roots']+[c['output_dir']]))
history={str(p):digest(p) for root in history_roots for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
ctx={};oldrun=read_config(Path(c['report_dir'])/'run_manifest.json')
for q in ['README.md','AGENTS.md','openspec/config.yaml']:
 target=ROOT/'project-context-snapshot'/q;target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():assert digest(target)==oldrun[q]
 else:target.write_bytes(Path(q).read_bytes())
 assert digest(target)==oldrun[q]
 ctx[q]=dict(original_sha256=oldrun[q],preserved_snapshot=str(target))
oldreport=read_config(parent_manifest)
for q,h in oldreport.items():assert digest(q)==h,q
pins=oldplan['verified_pins'];assert sys.version.split()[0]==oldplan['python']
assert all(version(k)==v for k,v in pins.items());assert digest(c['data_path'])==c['data_sha256']
# Required original source set presence only; this is not inference verification.
view=parent.adapter_view(c);availability={}
for spec in parent.required_sources(c):
 for month in [4,5,6,7]:
  phase='exploratory' if month==7 else 'validation';sets=[]
  for source,fp in parent.parent.parent.source_origins(view,spec):
   ps=parent.parent.parent.source_paths(source,spec,month,phase);present=[p.is_file() for p in ps]
   sets.append(dict(paths=[str(p) for p in ps],present=present))
   if any(present):
    assert all(present),f'Inspected partial original source: {spec["id"]} {month}'
    break
  availability[f'{spec["id"]}/m{month}']=sets
result=dict(parent_config=parent.CONFIG,parent_fingerprint=read_config(Path(c['output_dir'])/'selection.json')['fingerprint'],parent_selection_sha256=digest(Path(c['output_dir'])/'selection.json'),parent_report_manifest_sha256=digest(parent_manifest),reference_adapter_config=c['reference_adapter_config'],reference_adapter_fingerprint=c['reference_adapter_fingerprint'],python=oldplan['python'],verified_pins=pins,data_sha256=c['data_sha256'],representations=rep,recipes=recipes,strategies=strategies,reference_roles=refs,winner_policy_cases=cases,contrasts=contrasts,development_dependencies=all_dag,mandatory_preservation_sha256=mandatory,historical_output_roots=history_roots,historical_ignored_file_sha256=history,project_context_snapshots=ctx,original_cache_presence=availability,planned_source_paths=parent.SOURCES+['baf/context_ablation.py','baf/context_ablation_study.py','baf/context_ablation_report.py'],expected_counts=dict(strategies=38,winner_policy_cases=114,new_development_fits=18,reference_base_results=24,full_margin_controls=6,reference_derived_results=15,new_blends=72,models=42,model_files=45,physical_scores=135,crossfit=228,summary=114,pooled=114,contrasts=1192,physical_score_diagnostics=135,score_diagnostics=114,diversity=42,decision_overlap=516),new_model_fits=0,new_model_inferences=0,new_policy_evaluations=0)
assert len(result['planned_source_paths'])==64
write_json(ROOT/'planning-evidence.json',result)
print(json.dumps(dict(expected_counts=result['expected_counts'],mandatory_paths=len(mandatory),selection_pairs=len(pairs),historical_files=len(history),source_paths=64,cache_model_sets=len(availability),parent_report_entries_verified=len(oldreport),project_context_snapshots=3,new_model_fits=0,new_model_inferences=0,new_policy_evaluations=0),indent=2))
