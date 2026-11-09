"""Observe every real fit in the first focused core verification invocation."""
from pathlib import Path
import functools,json,sys,time
sys.path.insert(0,str(Path.cwd()))
import pytest
from lightgbm import LGBMClassifier
from baf.common import digest,write_json
from baf.focal_study import now
D=Path('reports/baf-context-feature-ablation-v1')
LEDGER=D/'core-focused-01-fits.json';XML=D/'core-focused-01.xml'
assert not LEDGER.exists() and not XML.exists()
SOURCES=['baf/context_ablation.py','tests/test_context_ablation.py',str(Path(__file__))]
class Ledger:
 def __init__(self):self.fits=[];self.nodeid='collection';self.nodes=[];self.started=now();self.sources={p:digest(p) for p in SOURCES}
 def pytest_collection_finish(self,session):self.nodes=[item.nodeid for item in session.items]
 def pytest_runtest_setup(self,item):self.nodeid=item.nodeid
 def save(self,code=None):write_json(LEDGER,dict(started_at_utc=self.started,recorded_at_utc=now(),scope='focused core tests only; not the complete methodological suite',source_sha256=self.sources,collected_nodes=self.nodes,fits=self.fits,exit_code=code,junit_sha256=digest(XML) if code is not None and XML.exists() else None))
a=Ledger();original=LGBMClassifier.fit
@functools.wraps(original)
def fit(*args,**kwargs):
 r=dict(nodeid=a.nodeid,status='started',started_at_utc=now());a.fits.append(r);a.save();t=time.perf_counter()
 try:result=original(*args,**kwargs);r['status']='completed';return result
 except Exception as e:r.update(status='failed',error=repr(e));raise
 finally:r['seconds']=time.perf_counter()-t;a.save()
LGBMClassifier.fit=fit
a.save()
try:code=pytest.main(['tests/test_context_ablation.py','-q','--junitxml='+str(XML)],plugins=[a])
finally:LGBMClassifier.fit=original
a.save(int(code));print('Actual current tiny fits:',len(a.fits));sys.exit(int(code))
