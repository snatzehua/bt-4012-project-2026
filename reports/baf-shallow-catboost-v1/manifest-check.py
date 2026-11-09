"""Index closed logs and finished run/report artifacts, excluding self indices."""
import json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from baf import shallow_catboost_study as run
from baf.common import read_config,write_json,digest
D=Path('reports/baf-shallow-catboost-v1');O=Path('outputs/baf-shallow-catboost-v1')
logs=sorted(Path('reports').glob('baf-shallow-catboost-v1-*.log'))
for p in logs:
    q=D/'closed-logs'/p.name;q.parent.mkdir(parents=True,exist_ok=True)
    if q.exists():assert digest(q)==digest(p),f'Closed log changed: {p}'
    else:shutil.copyfile(p,q)
closed={str(p):digest(p) for p in sorted(set(logs)|set(D.rglob('*.log'))) if p.is_file()}
write_json(D/'closed_log_manifest.json',closed);run.verify_hashes(closed)
change=run.planning_evidence_path().parent
files=[p for p in O.rglob('*') if p.is_file()]
files += [Path(p) for p in run.SOURCES]+[Path(run.CONFIG),Path('docs/shallow-catboost-protocol.md'),Path('scripts/check_shallow_catboost.py')]
files += list(Path('tests').glob('test_*.py'))+[Path('README.md'),Path('AGENTS.md'),Path('openspec/config.yaml'),Path('openspec/specs/baf-shallow-catboost-study/spec.md')]
files += [p for p in change.rglob('*') if p.is_file()]
files += logs
index={str(p):digest(p) for p in sorted(set(files))}
write_json(D/'run_manifest.json',index);run.verify_hashes(index)
write_json(D/'manifest-verification.json',dict(checked_at_utc=run.now(),closed_log_entries=len(closed),run_entries=len(index),all_closed_logs_and_run_entries_verified=True,report_index_excludes_itself=True,report_index_verified_by_successful_command_exit=True))
report={str(p):digest(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='report_hashes.json' and '__pycache__' not in p.parts}
write_json(D/'report_hashes.json',report);run.verify_hashes(report)
print(json.dumps(dict(closed_logs=len(closed),run_files=len(index),report_files=len(report),all_indexed_members_verified=True,report_manifest_sha256=digest(D/'report_hashes.json')),indent=2))
