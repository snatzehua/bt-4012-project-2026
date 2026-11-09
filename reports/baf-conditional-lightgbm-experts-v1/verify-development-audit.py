"""Verify the completed independent audit with explicit subrecord scope."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baf import conditional_expert_study as run
from baf.common import read_config,write_json,digest
c,fp=run.initialize();out=Path(c['output_dir']);dest=Path(c['report_dir'])
a=read_config(out/'independent_development_audit.json')
assert a['fingerprint']==fp and a['validation_sha256']==digest(out/'validation_complete.json')
assert a['saved_model_checks']==84 and a['actual_source_models_audited']
assert a['independent_preprocessing_inference_costs_memberships'] and a['independent_policies_metrics_factors']
assert a['compositions']['new_full_compositions']==48 and a['compositions']['member_subsets']==54
assert a['compositions']['actual_source_models_audited'] is False  # score-only subroutine
assert a['counts']==dict(crossfit=144,summary=72,pooled=72,contrasts=738,eligibility=72,
    routing=96,stratum_errors=2304,physical_score_diagnostics=93,score_diagnostics=72)
assert a['winner']=='bagged_anchor_balanced_logit_a25' and a['policy']=='pooled'
run.verify_hashes({str(out/p):h for p,h in read_config(out/'validation_complete.json')['files'].items()})
run.guard(c)
write_json(dest/'development-audit-verification.json',dict(status='passed',fingerprint=fp,
    independent_audit_sha256=digest(out/'independent_development_audit.json'),
    model_checks=84,full_compositions=48,subsets=54,counts=a['counts'],
    source_model_audit_separately_verified=True,composition_only_scope_disclosed=True,
    supplementary_initial_assertion_retained=True,new_fits=0,created_at_utc=run.now()))
p=Path('openspec/changes/test-conditional-lightgbm-experts-f2/tasks.md')
t=p.read_text();assert '- [ ] 3.3 ' in t;p.write_text(t.replace('- [ ] 3.3 ','- [x] 3.3 '))
print('Verified completed 84-model independent audit and all table/composition scopes; retained supplementary reader assertion correction, zero new fits.')
