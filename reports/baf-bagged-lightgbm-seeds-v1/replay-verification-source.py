from pathlib import Path
import re,subprocess,json,sys,shutil,importlib.metadata
from baf.common import digest,read_config,write_json
root=Path.cwd();dest=root/'reports/baf-bagged-lightgbm-seeds-v1'
text=(root/'README.md').read_text().split('### Bagged LightGBM seed study (in progress)',1)[1].split('## Acknowledging assistance',1)[0]
blocks=re.findall(r'```bash\n(.*?)\n```',text,re.S)
assert len(blocks)==3
run=subprocess.run(['bash','-c',blocks[0]+'\npwd\n'],text=True,capture_output=True,check=True)
replay=Path(run.stdout.strip().splitlines()[-1]);assert replay!=root and replay.name.startswith('baf-bagged-replay.')
assert not list((replay/'outputs').rglob('*')) if (replay/'outputs').exists() else True
restore=blocks[1];first=subprocess.run(['bash','-c',restore],cwd=replay,text=True,capture_output=True)
assert first.returncode==0,first.stderr
c=read_config(root/'configs/bagged-lightgbm-seeds.json');pairs=[(p,h) for p,h in c['preserve_sha256'].items() if p.startswith('outputs/') and p.endswith('/selection.json')];assert len(pairs)==15
for p,h in pairs:assert digest(replay/p)==h
p=replay/pairs[0][0];old=p.read_bytes()
try:
    p.write_bytes(old+b' ')
    rejected=subprocess.run(['bash','-c',restore],cwd=replay,text=True,capture_output=True)
    assert rejected.returncode!=0
finally:p.write_bytes(old)
assert not list((replay/'outputs').rglob('*.npz')) and not list((replay/'outputs').rglob('*.joblib'))
pins={}
for file in ['requirements.txt','requirements-improvements.txt','requirements-xgboost.txt']:
    for line in (root/file).read_text().splitlines():
        if '==' in line:
            name,version=line.split('==');assert importlib.metadata.version(name)==version;pins[name]=version
assert len(pins)==27 and sys.version.split()[0]=='3.12.9'
assert digest(replay/'data/raw/baf/Base.csv')=='7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809'
subprocess.run([str(replay/'.venv/bin/python'),'-c','from baf import bagged_boosting as r;from baf.common import read_config;r.guard(read_config(r.CONFIG));print("Replay preservation and source identities valid")'],cwd=replay,check=True)
write_json(dest/'replay_setup.json',dict(replay_root=str(replay),readme_sha256=digest(root/'README.md'),selection_pairs=15,changed_selection_rejected=True,prior_model_prediction_caches=0,python=sys.version.split()[0],pins=pins,data_sha256=digest(replay/'data/raw/baf/Base.csv'),no_production_fit=True))
(dest/'executed-replay-instructions.md').write_text(text)
print('Isolated README setup,15selection restores,rejection,27pins,dataset and preservation guard verified:',replay,flush=True)
