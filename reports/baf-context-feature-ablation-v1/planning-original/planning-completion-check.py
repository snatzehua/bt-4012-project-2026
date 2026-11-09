"""Audit proposal artifacts and unchanged execution state; no scientific evaluation."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from baf.common import read_config,digest,write_json
R=Path(__file__).resolve().parent
p=read_config(R/'planning-evidence.json');status=read_config(R/'planning-status.json')
assert status['isPlanningComplete'] is True and all(a['status']=='done' for a in status['artifacts'])
spec=(R/'specs/baf-context-feature-ablation-study/spec.md').read_text();tasks=(R/'tasks.md').read_text()
assert spec.count('### Requirement: ')==9 and spec.count('#### Scenario: ')==13
assert tasks.count('- [ ] ')==14 and '- [x] ' not in tasks
assert len(p['strategies'])==38 and len(p['winner_policy_cases'])==114 and len(p['contrasts'])==1192
assert len(p['planned_source_paths'])==64 and len(set(p['planned_source_paths']))==64
for name in p['planned_source_paths'][-3:]+['configs/context-feature-ablation.json','docs/context-feature-ablation-protocol.md','tests/test_context_ablation.py','scripts/check_context_ablation.py']:
 assert not Path(name).exists(),f'Planning-only boundary violated: {name}'
assert not Path('outputs/baf-context-feature-ablation-v1').exists() and not Path('reports/baf-context-feature-ablation-v1').exists()
assert p['new_model_fits']==p['new_model_inferences']==p['new_policy_evaluations']==0
for key,n in [('mandatory_preservation_sha256',2020),('historical_ignored_file_sha256',5432)]:
 assert len(p[key])==n
 for path,h in p[key].items():assert digest(path)==h,path
for path,r in p['project_context_snapshots'].items():
 assert digest(path)==r['original_sha256']==digest(r['preserved_snapshot'])
parent=read_config(p['parent_config']);D=Path(parent['report_dir']);O=Path(parent['output_dir'])
assert digest(O/'selection.json')==p['parent_selection_sha256'] and digest(D/'report_hashes.json')==p['parent_report_manifest_sha256']
for name in ['report_hashes.json','run_manifest.json','closed_log_manifest.json']:
 for path,h in read_config(D/name).items():assert digest(path)==h,(name,path)
assert '0 failed' in (R/'planning-validation.log').read_text()
files={str(q.relative_to(Path.cwd())):digest(q) for q in sorted(R.rglob('*')) if q.is_file() and q.name not in ['planning-completion.json','planning-completion.log']}
result=dict(planning_complete=True,implementation_complete=False,implementation_tasks_complete=0,implementation_tasks_total=14,requirements=9,scenarios=13,prospective_cases=114,contrasts=1192,mandatory_paths=2020,selection_pairs=19,preserved_historical_files=5432,planned_sources=64,parent_scientific_execution_and_report_indices_unchanged=True,project_context_current_and_snapshot_hashes_equal=True,new_fits=0,new_inferences=0,new_policy_evaluations=0,broader_goal_achieved=False,artifact_sha256=files)
write_json(R/'planning-completion.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2))
