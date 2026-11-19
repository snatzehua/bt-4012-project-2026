# BT4012 submission working pack

This pack consolidates completed frozen studies without changing modelling recipes or opening another search. The canonical code is `baf/`; the read-only submission interface is `scripts/build_submission_evidence.py`. See `docs/submission-code-guide.md` for export versus full model-replay commands and `docs/submission-branch-audit.md` for pinned remote comparisons.

## Build order

```bash
python3 scripts/build_submission_evidence.py --check
python3 scripts/build_submission_evidence.py
.venv/bin/python -m pytest tests/test_submission_evidence.py -q
.venv/bin/python scripts/build_report_pdf.py
python3 scripts/build_slide_skeleton.py
```

Report and slide scripts are separate build steps: evidence export validates historical CSVs, not model retraining. Existing study tests and prior clean-copy receipts are historical checks; only checks listed in `verification.json` were executed for this delivery.

## Outputs

- `report.md`: editable report text.
- `reportXX-draft.pdf`: report PDF; replace XX with the real group number after review.
- `slides.md`: slide-by-slide content, timing and speaker cues.
- `presentationXX-skeleton.pptx`: plain editable deck; aesthetics intentionally preliminary.
- `generated/`: measured tables and source hash manifest.

## Required human handoff

1. Fill the actual group number and all member names. Search for `XX`, `[GROUP` and `[MEMBER` in sources; do not infer names from commit authors.
2. Review the findings and AI/external-code acknowledgement together; every member must explain their material.
3. Refine slide aesthetics and divide speaking time reasonably. Record a playable video no longer than 15 minutes, with readable opening title/member names and clear audio.
4. Rename final outputs `reportXX.pdf`, `presentationXX.pptx`, `videoXX.mp4` using a two-digit group number.
5. Ensure the teaching team can access the GitHub URL included in the report. Local Git metadata/network restrictions blocked publication in this session; no push or merge is claimed.
6. Submit on Canvas by 20 November 2026, 23:59 Asia/Singapore. Report: Project > Final Report. Slides/video: Project > Presentation.

No video is generated here; recording requires the group's actual speakers. The report is a completed draft for group review, not a claim of final group approval. Month 7 is consumed and exploratory. The unfinished context-feature-ablation study and stopped L-BFGS attempt are excluded from new performance claims.

## Scope of scientific synthesis

Use the first study for exact size/class-matched recency comparisons, improvements for feature/operating-policy evidence and selected completed F2 studies as a compact exploratory synthesis. Remote ZS output is a separate supplementary study with different training schedules, capacity rounding and already-exposed evaluation months. No unified branch leaderboard or independent replication is claimed.
