"""Synthetic score-only reconstruction of the complete declared score/subset grid."""
from pathlib import Path
from itertools import product
from unittest.mock import patch
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from baf import conditional_expert_study as run
from baf import conditional_expert_report as audit
from baf.common import read_config, write_json, digest
from baf.improvements import prediction

root = Path(__file__).resolve().parent
out = root/'synthetic-score-grid'
if out.exists():
    raise ValueError('Preserve existing verification; do not overwrite the fixture')
c = dict(read_config(run.CONFIG), output_dir=str(out))
cells = list(product(['AA','AB','AC','AD'],['BA','BB','BC','BD','BE']))
frame = pd.DataFrame(dict(month=np.zeros(60000,dtype=int),
    payment_type=[x for x,y in cells for _ in range(3000)],
    housing_status=[y for x,y in cells for _ in range(3000)],
    fraud_bool=(np.arange(60000)%9==0).astype(int)))
future = []
for m in [4,5,6]:
    future.append(pd.DataFrame(dict(month=[m]*42,
        payment_type=np.resize(np.array(['AA','AB','AC','AD','AE',None,'unknown'],dtype=object),42),
        housing_status=np.resize(np.array(['BA','BB','BC','BD','BE','BF','BG',None,'unknown'],dtype=object),42),
        fraud_bool=(np.arange(42)%7==0).astype(int))))
frame = pd.concat([frame,*future],ignore_index=True)
fp = {'fixture':'synthetic scores only; no classifier, preprocessor or operational-result claim'}
rng = np.random.default_rng(383)
def save(name,m,ids,labels,values):
    rp,pp=run.paths(c,'validation',m,name);pp.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(pp,row_ids=ids,labels=labels,scores=values)
    write_json(rp,dict(fixture=fp,candidate=name,eval_month=m,prediction_sha256=digest(pp)))
old_names=[n for n,k in run.scores.REFERENCE_ROLES.values()]
old_dag=run.parent.scores.dependencies(old_names)
subset_names=set()
with patch('lightgbm.LGBMClassifier.fit',side_effect=AssertionError('No model fits in a score-only audit')):
    for m in [4,5,6]:
        ids=frame.index[frame.month==m].to_numpy();y=frame.loc[ids,'fraud_bool'].to_numpy()
        for name in old_dag['bases']:
            save(name,m,ids,y,rng.uniform(.01,.99,len(ids)))
        for name in run.parent.derived_order(old_names):
            spec=run.parent.derived_spec(name);items={n:prediction(c,'validation',m,n) for n in spec['members']}
            d=run.parent.parent.old_compose(spec,items) if spec['kind']=='anchor' else run.parent.scores.compose(spec,items)
            save(name,m,ids,y,d['scores'])
        for w in run.scores.WEIGHTINGS:
            save(run.scores.capacity(w),m,ids,y,rng.uniform(.01,.99,len(ids)))
        for spec in run.scores.model_specs(frame,m,capacities=[]):
            subset=ids[frame.loc[ids,spec['partition']].eq(spec['level']).to_numpy()]
            save(spec['id'],m,subset,frame.loc[subset,'fraud_bool'].to_numpy(),rng.uniform(.01,.99,len(subset)))
            subset_names.add((m,spec['id']))
        for p,w in product(run.scores.PARTITIONS,run.scores.WEIGHTINGS):
            for a in [100,25,50,75]:
                run.new_derived(c,fp,frame,run.scores.candidate(p,w,a),m)
    h=run.history(c,physical=True)
    tables=run.tables(c,frame,h)
    for name,table in tables.items():table.to_csv(out/(name+'.csv'),index=False)
    compositions=audit.audit_new_compositions(c,fp,frame,[4,5,6])
    receipt=audit.audit_tables(c,frame,h)
    assert compositions['new_full_compositions']==48 and compositions['member_subsets']==54
    assert len(subset_names)==54 and sum(len(v) for v in h.values())==93
    assert not receipt['actual_source_models_audited'] and not compositions['actual_source_models_audited']
    receipt.update(new_compositions=compositions,new_model_fits=0,
                   scope='Synthetic scores/memberships only; no actual model or performance claim',
                   source_sha256={n:digest(n) for n in run.SOURCES+[str(Path(__file__))]})
    write_json(root/'synthetic-score-grid-verification.json',receipt)
print(json.dumps(receipt,indent=2))
