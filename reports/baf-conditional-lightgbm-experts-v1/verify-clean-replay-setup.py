"""Exact restore/rejection and cache-free archive-path setup in a disposable copy."""
from pathlib import Path
import importlib.metadata
import platform
import shutil
import sys

root=Path.cwd();sys.path.insert(0,str(root))
from baf import conditional_expert_study as run
from baf.common import read_config,write_json,digest

c=read_config(run.CONFIG);dest=Path(c['report_dir'])
pairs=[(Path(p),Path(p.replace('outputs/','reports/',1)),h)
       for p,h in c['preserve_sha256'].items() if p.startswith('outputs/') and p.endswith('/selection.json')]
assert len(pairs)==17
def restore():
    for original,exported,expected in pairs:
        assert exported.is_file() and digest(exported)==expected,str(exported)
        if original.exists():assert digest(original)==expected,str(original)
    for original,exported,expected in pairs:
        if not original.exists():
            original.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(exported,original)
restore();rejected=[]
for original,exported,expected in pairs:
    saved=original.read_bytes();original.write_bytes(b'{"changed fixture": true}\n')
    try:
        try:restore()
        except AssertionError as exc:
            assert str(original) in str(exc);rejected.append(str(original))
        else:raise AssertionError('Changed selection was accepted')
    finally:original.write_bytes(saved)
restore()
assert all(digest(original)==h for original,exported,h in pairs)
assert not list(Path('outputs').rglob('*.joblib')) and not list(Path('outputs').rglob('*.npz')) and not list(Path('outputs').rglob('*.ubj'))
assert platform.python_version()=='3.12.9'
pins={}
for path in ['requirements.txt','requirements-improvements.txt','requirements-xgboost.txt']:
    for line in Path(path).read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            package,version=line.strip().split('==');assert importlib.metadata.version(package)==version
            pins[package]=version
assert len(pins)==27 and digest(c['data_path'])==c['data_sha256']
declaration=run.verify_declaration()
assert run.present_history(c,read_config(run.planning_evidence_path())['historical_ignored_file_sha256'])=={
    str(original):h for original,exported,h in pairs}
active=Path('openspec/changes/test-conditional-lightgbm-experts-f2')
archived=Path('openspec/changes/archive/2026-10-12-test-conditional-lightgbm-experts-f2')
assert active.is_dir() and not archived.exists()
active.rename(archived)
assert run.planning_evidence_path()==archived/'planning-evidence.json'
assert digest(run.planning_evidence_path())==c['planning_evidence_sha256']
result=dict(copy_root=str(root),selection_pairs_restored=17,changed_restores_rejected=rejected,
    data_sha256=c['data_sha256'],python=platform.python_version(),pins=pins,
    source_sha256={p:digest(p) for p in run.SOURCES},prior_models=0,prior_predictions=0,
    archive_path_simulated=True,archived_planning_sha256=digest(run.planning_evidence_path()),
    full_size_models_retrained=False,setup_verified=True,created_at_utc=run.now())
write_json(dest/'clean-replay-setup.json',result)
print('Verified seventeen restores/rejections, 27 pins, Base checksum, zero model/prediction caches and archived planning path.')
