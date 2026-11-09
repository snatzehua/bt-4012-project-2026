"""Run methodology tests with transparent, separately scoped model-fit accounting."""
from __future__ import annotations

import argparse
from functools import wraps
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from baf.common import digest,write_json
from baf.focal_study import now
from baf import context_ablation_study as run


class FitAccounting:
    def __init__(self,path,junit):
        self.scope='all';self.path=path;self.junit=junit;self.nodeid='collection';self.depth=0;self.fits=[];self.originals=[]
        files=run.SOURCES+[run.CONFIG,'docs/context-feature-ablation-protocol.md',str(Path(__file__).relative_to(REPO))]
        files += [str(p) for p in sorted(Path('tests').glob('test_*.py'))]
        self.execution_sources={p:digest(p) for p in files}
        self.started_at=now()
        self.collected_nodes=[];self.deselected_nodes=[]

    def save(self,exit_code=None):
        write_json(self.path,dict(started_at_utc=self.started_at,recorded_at_utc=now(),
            execution_source_sha256=self.execution_sources,scope=self.scope,fits=self.fits,exit_code=exit_code,
            collected_nodes=self.collected_nodes,deselected_nodes=self.deselected_nodes,
            junit_sha256=digest(self.junit) if exit_code is not None and self.junit.exists() else None,
            semantics='Outermost real classifier fit calls; preprocessing fits and mocked methods excluded; current/historical test node IDs retained.'))

    def install(self):
        for cls in [LogisticRegression,RandomForestClassifier,LGBMClassifier,XGBClassifier,CatBoostClassifier]:
            original=cls.fit;self.originals.append((cls,original))
            setattr(cls,'fit',self.wrap(original,cls.__module__+'.'+cls.__name__))
        self.save()

    def wrap(self,original,kind):
        @wraps(original)
        def recorded(*args,**kwargs):
            if self.depth:
                return original(*args,**kwargs)
            row=dict(nodeid=self.nodeid,estimator=kind,status='started',started_at_utc=now())
            self.fits.append(row);self.save();start=time.perf_counter();self.depth+=1
            try:
                result=original(*args,**kwargs);row['status']='completed';return result
            except BaseException as exc:
                row.update(status='failed',error=repr(exc));raise
            finally:
                self.depth-=1;row['elapsed_seconds']=time.perf_counter()-start;self.save()
        return recorded

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self,item,nextitem):
        self.nodeid=item.nodeid
        yield
        self.nodeid='between tests'

    def pytest_collection_finish(self,session):
        self.collected_nodes=[item.nodeid for item in session.items]
        self.save()

    def pytest_deselected(self,items):
        self.deselected_nodes.extend(item.nodeid for item in items)

    def restore(self):
        for cls,original in self.originals:
            cls.fit=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--junit',type=Path,default=Path('reports/baf-context-feature-ablation-v1/tests.xml'))
    parser.add_argument('--ledger',type=Path,default=Path('reports/baf-context-feature-ablation-v1/test_model_fits.json'))
    parser.add_argument('--scope',choices=['all','current'],default='all')
    args,extra=parser.parse_known_args()
    if args.ledger.exists() or args.junit.exists():raise ValueError('Retain existing ledger; choose a fresh path before a new invocation')
    accounting=FitAccounting(args.ledger,args.junit);accounting.scope=args.scope;accounting.install()
    try:
        result=pytest.main(['tests' if args.scope=='all' else 'tests/test_context_ablation.py','-q','--junitxml='+str(args.junit),*extra],plugins=[accounting])
    finally:
        accounting.restore()
    accounting.save(int(result))
    print('Actual classifier-fit calls:',len(accounting.fits),flush=True)
    return int(result)


if __name__=='__main__':
    raise SystemExit(main())
