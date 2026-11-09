"""Observe all real classifier fits in focused reduced/original/native-control tests."""
from pathlib import Path
import functools,sys,time
sys.path.insert(0,str(Path.cwd()))
import pytest
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from baf import context_ablation_study as run
from baf.common import digest,write_json
from baf.focal_study import now
D=Path('reports/baf-context-feature-ablation-v1');LEDGER=D/'core-focused-04-fits.json';XML=D/'core-focused-04.xml'
assert not LEDGER.exists() and not XML.exists()
SOURCES=run.SOURCES+[run.CONFIG,'docs/context-feature-ablation-protocol.md','tests/test_context_ablation.py',str(Path(__file__))]
class Ledger:
 def __init__(self):self.fits=[];self.nodeid='collection';self.nodes=[];self.started=now();self.sources={p:digest(p) for p in SOURCES};self.depth=0
 def pytest_collection_finish(self,session):self.nodes=[item.nodeid for item in session.items]
 def pytest_runtest_setup(self,item):self.nodeid=item.nodeid
 def save(self,code=None):write_json(LEDGER,dict(started_at_utc=self.started,recorded_at_utc=now(),scope='focused current tests; not the complete methodological suite',source_sha256=self.sources,collected_nodes=self.nodes,fits=self.fits,exit_code=code,junit_sha256=digest(XML) if code is not None and XML.exists() else None))
a=Ledger();originals=[]
def wrap(original,kind):
 @functools.wraps(original)
 def fit(*args,**kwargs):
  if a.depth:return original(*args,**kwargs)
  r=dict(nodeid=a.nodeid,estimator=kind,status='started',started_at_utc=now());a.fits.append(r);a.save();t=time.perf_counter();a.depth+=1
  try:result=original(*args,**kwargs);r['status']='completed';return result
  except Exception as e:r.update(status='failed',error=repr(e));raise
  finally:a.depth-=1;r['seconds']=time.perf_counter()-t;a.save()
 return fit
for cls in [LGBMClassifier,XGBClassifier,LogisticRegression]:
 original=cls.fit;originals.append((cls,original));cls.fit=wrap(original,cls.__name__)
a.save()
try:code=pytest.main(['tests/test_context_ablation.py','-q','--junitxml='+str(XML)],plugins=[a])
finally:
 for cls,original in originals:cls.fit=original
a.save(int(code));print('Actual current tiny fits:',len(a.fits));sys.exit(int(code))
