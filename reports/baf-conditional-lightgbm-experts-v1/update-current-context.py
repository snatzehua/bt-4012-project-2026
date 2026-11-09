"""Document measured results after verification; preserve both closure snapshots."""
from pathlib import Path
import argparse
import importlib.metadata
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baf import conditional_expert_study as run
from baf.common import read_config,write_json,digest

parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
c,s=run.load_selection();dest=Path(c['report_dir']);science=read_config(dest/'scientific-results-verification.json')
assert science['status']=='passed' and not science['broader_goal_achieved']
change=run.planning_evidence_path().parent
phase='final' if args.final else 'prearchive'
if args.final:assert change.parent.name=='archive' and '- [ ]' not in (change/'tasks.md').read_text()
target=str(change)
status='All fourteen tasks are verified and requirements are synchronized; the change is archived.' if args.final else 'Scientific execution and audits are complete; final OpenSpec closure remains tracked in the task checklist.'
paragraph=f'''The completed [conditional-model experiment]({target}/proposal.md) has checked [findings](reports/baf-conditional-lightgbm-experts-v1/findings.md), [requirements]({target}/specs/baf-conditional-lightgbm-expert-study/spec.md) and [tasks]({target}/tasks.md). Its 24 strategies used 54 local/six global400 development fits, 24 base/fifteen derived reuses, 93 full scores and 54 separate subsets. All 84 saved models, 144 earlier-only operating rows, 738 contrasts and 2304 group-error rows passed independent auditing. No new strategy improved the existing bagged-anchor/pooled winner: macro F2 remains **{science['development_f2']:.9f}**, versus **{science['best_new_f2']:.9f}** for the best new housing/balanced/25% blend with latest policy. All twelve pure-expert versus global200 effects are negative. Eight frozen roles/eight models/thirteen scores transferred with zero new fits; consumed F2 remains **{science['consumed_f2']:.9f}**, below tree fusion and unweighted references. All 1500 paired draws/28 interval rows and the rendered figure passed checks. {status} Primary and no-prior-cache copied suites each pass 1915 tests (414 own). Sixty cumulative current tiny fits, 300 historical test calls and sixteen production miniature fits are separate from the 60 full development fits. Preserve all seventeen predecessors, the 58-source/config/software/protocol/selection/access identity, 1053 mandatory paths, seventeen selection pairs, 4369 historical files and the stopped attempt. See the retained [miniature metadata correction](reports/baf-conditional-lightgbm-experts-v1/miniature-metadata-note.md) and [work measurement scope](reports/baf-conditional-lightgbm-experts-v1/work-accounting-note.md). F2 beta=2 has no quota or precision floor; settings/partition choices are adaptive analyst selections, month 7 is consumed and the broader goal remains active.'''
p=Path('README.md');lines=p.read_text().splitlines()
indices=[i for i,v in enumerate(lines) if v.startswith('The [conditional-model study](') or v.startswith('The completed [conditional-model experiment](')]
assert len(indices)==1;lines[indices[0]]=paragraph
text='\n'.join(lines)+'\n'
text=text.replace('These commands are an execution procedure, not evidence that the full experiment has completed.','These commands reproduced the checked scientific execution in the primary destination. Use a new destination for further tuning; do not overwrite the completed frozen run.')
text=text.replace('Current synthetic checks do not establish a model-performance gain.','The independently checked finite experiment found no new F2 gain; the broader performance goal remains active.')
p.write_text(text)
p=Path('AGENTS.md');lines=p.read_text().splitlines();ix=[i for i,v in enumerate(lines) if v.startswith('`test-conditional-lightgbm-experts-f2`')];assert len(ix)==1
lines[ix[0]]=f'''`test-conditional-lightgbm-experts-f2` has completed scientific execution and independent audits under configs/conditional-lightgbm-experts.json, docs/conditional-lightgbm-experts-protocol.md and reports/baf-conditional-lightgbm-experts-v1/. {status} All 24 strategies/72 cases, 84 saved development models, 93 full scores/54 subsets, 144 chronological operating rows, 738 contrasts, 96 routing/2304 stratum rows and measured work are checked. Sixty new full development fits (54 local/six capacity) reused 24 bases/fifteen derived scores. Existing bagged-anchor/pooled remains selected at macro F2 {science['development_f2']:.9f}; best new housing/balanced/25% latest blend is {science['best_new_f2']:.9f}; all twelve pure-vs-global200 effects are negative. Eight frozen roles/eight bases/thirteen scores transferred with zero new fits and eight base/five derived reuses; consumed F2 {science['consumed_f2']:.9f} fails tree-fusion/unweighted comparisons and ties the anchor. All 1500 draws/28 intervals and inspected/pixel-regenerated figure passed. Primary/clean-copy suites each pass 1915 tests (414 own), ten current/75 historical tiny fits per full invocation; cumulative 60 current/300 historical calls are separate from sixteen production miniature and sixty full development fits. Seventeen restores/rejections, Base SHA, Python 3.12.9/27 pins and archived-path clean-copy setup pass without prior model/prediction caches; no full-size retraining there is claimed. Resource and preserved 9/8-fit gates pass. Preserve the getter-only miniature metadata defect/correction and supplementary composition-record assertion correction; frozen producers/models/scores were unchanged and numerical recipes were not relaxed. Work row-rounds are logical per-method exposure, not physical cache/audit call counts or equal-runtime/storage evidence. Keep all seventeen predecessors, 1053 mandatory paths, seventeen selection pairs, 4369 historical files, separate Newton identities and stopped attempt intact. The 58-source/config/software/protocol/selection/access identity is frozen. Further tuning needs a separate study. F2 beta=2 has no quota/floor; settings/partitions are adaptive analyst selections, month7 remains consumed and the broader goal remains active.'''
p.write_text('\n'.join(lines)+'\n')
p=Path('openspec/config.yaml');text=p.read_text();start=text.index('  The separate test-conditional-lightgbm-experts-f2 ');end=text.index('  First-study settings are analyst-selected',start)
replacement=f'''  The separate test-conditional-lightgbm-experts-f2 completed scientific execution
  and independent audits. {status}
  Its 24 strategies used 54 local/six capacity full development fits and 24 base/
  fifteen derived reuses. All 84 saved models, 93 full scores/54 subsets, 144
  operating rows, 738 contrasts, 96 routing/2304 group-error rows and actual work
  passed. Existing anchor/pooled remains selected at macro F2 {science['development_f2']:.9f};
  best new housing/balanced/25% latest blend scored {science['best_new_f2']:.9f}. All twelve
  pure-vs-global200 comparisons are negative. Eight frozen roles/eight models/
  thirteen scores transferred with zero new fits; consumed F2 {science['consumed_f2']:.9f}
  fails tree-fusion/unweighted comparisons and ties the existing anchor.
  All 1500 paired fixed-decision draws/28 intervals and the inspected, identical
  regenerated figure passed. Primary/clean suites each pass 1915 tests (414 own),
  ten current/75 historical tiny fits per full invocation. Sixty cumulative
  current tiny fits and 300 historical calls are separate from sixteen production
  miniature and sixty full development fits. Seventeen selection restores/rejections,
  Python3.12.9/27 pins/Base SHA and no-prior-cache archived-path copy pass;
  full-size clean-copy retraining is not claimed. Preserve all seventeen ancestors,
  1053 mandatory paths, 4369 historical files, stopped attempt and separate Newton
  identities. Keep the frozen 58-source/config/software/protocol/selection/access
  identity and getter-only miniature metadata/supplementary assertion corrections.
  No producer/model/prediction was edited or numerical recipe relaxed. Work
  row-rounds describe logical method exposure, not physical prediction-call counts.
  Further tuning requires a separate study. F2 beta=2 has no quota/floor;
  partitions/settings are adaptive analyst selections, month7 remains consumed,
  the broader performance goal remains active and no independent confirmation is claimed.
'''
p.write_text(text[:start]+replacement+text[end:])
readme=Path('README.md').read_text()
commands=['record-checks','preflight','miniature','prepare-development','validate','freeze','exploratory']
positions=[readme.index('baf.conditional_expert_study '+a) for a in commands]
assert positions==sorted(positions) and readme.index('baf.conditional_expert_report report')>positions[-1]
for phrase in ['assert len(pairs) == 17','7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809','Python 3.12.9','requirements-xgboost.txt','Wholly absent required references','partial or incompatible sets','Codex','no public notebook','miniature-metadata-note.md','work-accounting-note.md']:
    assert phrase.lower() in readme.lower(),phrase
setup=read_config(dest/'clean-replay-setup.json');pins=setup['pins']
assert len(pins)==27
for name,version in pins.items():assert importlib.metadata.version(name)==version
copy=read_config(dest/'clean-replay-verification.json');assert copy['source_fingerprint']==s['fingerprint']
assert copy['original_selection_restores']==copy['changed_restore_rejections']==17
assert copy['no_prior_models_or_predictions'] and not copy['full_size_retraining_claimed']
snapshot=dest/(phase+'-documentation-source');assert not snapshot.exists();snapshot.mkdir()
files={}
for name in ['README.md','AGENTS.md','openspec/config.yaml']:
    p=snapshot/(name+'.source');p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,p);files[str(p)]=digest(p)
receipt='final-documentation-verification.json' if args.final else 'documentation-verification.json'
assert not (dest/receipt).exists()
write_json(dest/receipt,dict(status='passed',phase=phase,snapshot_files=files,
    current_root_files={n:digest(n) for n in ['README.md','AGENTS.md','openspec/config.yaml']},
    ordered_commands_checked=True,pins_verified=27,selection_restores_verified=17,
    methodology_full_tests=1915,current_tests=414,full_size_copy_retraining_claimed=False,
    frozen_execution_identity_unchanged=True,assistance_disclosed=True,broader_goal_achieved=False,
    created_at_utc=run.now()))
print('Updated and verified',phase,'project context, ordered commands, pins/restores and assistance; original frozen scientific identities unchanged.')
