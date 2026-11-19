# Group 8 submission assets

The repository contains report and presentation drafts built from completed studies. The [code and evidence guide](../docs/submission-code-guide.md) explains table export and model reproduction; the [branch audit](../docs/submission-branch-audit.md) records the separate supplementary workflow.

## Available files and sources

| Asset | Editable source and status |
|---|---|
| [reportXX-draft.pdf](reportXX-draft.pdf) | Generated from [report.md](report.md); tracked Markdown draft, not the separately edited Word report |
| [presentationXX-skeleton.pptx](presentationXX-skeleton.pptx) | Preliminary deck generated from content embedded in `scripts/build_slide_skeleton.py` |
| [slides.md](slides.md) | Generated outline with slide content, timing and speaker cues; edits are overwritten by the skeleton generator |
| [generated/](generated/) | Ten evidence tables and a source/hash manifest |
| [verification.json](verification.json), [slide-validation.json](slide-validation.json) | Historical checks of the tracked drafts, not approval of later edits or final submission |

The separately edited `report08.docx` is maintained outside this repository. The Markdown/PDF builder does not reproduce its edits. No final `report08.pdf`, final `presentation08.pptx` or `video08.mp4` is included in this checkout. The `XX` filenames and member placeholders remain in the historical drafts; Group 8 is known, but final approval and complete member details have not been verified.

## Rebuild the tracked drafts

Run evidence checks in a checkout with original LF file bytes; see the [checkout instructions](../docs/submission-code-guide.md#preserve-original-checkout-bytes). From the repository root:

```sh
python scripts/build_submission_evidence.py --check
python scripts/build_submission_evidence.py
```

The exporter uses only the standard library. It overwrites `submission/generated/` and does not retrain models. The PDF builder additionally needs matplotlib and Pillow from the pinned modelling environment:

```sh
.venv/bin/python scripts/build_report_pdf.py
```

This reads `submission/report.md` and overwrites `submission/reportXX-draft.pdf` and its `.layout.json` receipt. `--source` and `--output` select other paths; keep preview outputs inside the repository, for example under ignored `tmp/`. It does not convert `report08.docx`. A fresh build checks its own main-page count; the previous draft's layout receipt does not establish the final Word/PDF page count.

```sh
python scripts/build_slide_skeleton.py
```

This standard-library draft generator overwrites `presentationXX-skeleton.pptx`, `slides.md` and `slide-validation.json`. Save manually edited slides under a separate filename before running it. Editing `slides.md` does not update the deck; slide content lives in the script. The generator does not reproduce a subsequently designed final presentation.

## Before submission

Confirm the final report source and export it as `report08.pdf`; supply the reviewed presentation as `presentation08.pptx` and the recorded video as `video08.mp4`. The report must have 8-12 pages excluding references/appendices, and the video must be playable and no longer than 15 minutes. Verify member names/admin numbers, references, assistance acknowledgement and the teaching team's repository access. No final page-count, video or Canvas submission check is claimed here.

Submit by **20 November 2026, 23:59 Asia/Singapore**: report under Project > Final Report; slides/video under Project > Presentation. These are the remaining submission checks, not evidence that delivery has occurred.

## Scientific scope

The original controlled study supports recency/model comparisons; improvements support feature and policy analysis. Later F2 and supplementary operational results are exploratory because month 7 has already been evaluated. Keep different policies and branch timelines separate. The incomplete context-feature-ablation study and stopped L-BFGS attempt contribute no completed performance claims.
