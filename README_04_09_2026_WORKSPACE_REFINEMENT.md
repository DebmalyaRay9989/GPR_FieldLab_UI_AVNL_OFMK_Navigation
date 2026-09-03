# GPR FieldLab workspace refinement

- Removed the Mission / Survey Information block from the post-login single-scan workspace.
- Reorganized Single Scan into acquisition, AI inference, detection review, exports, and processing flow.
- Reorganized Batch Scan into multi-file acquisition/queue, shared processing profile, run controls, and batch results.
- Existing Dash IDs and processing callbacks are retained.
- The drone background remains active behind the authenticated workspace.

## Final post-login cleanup

- Removed the redundant post-login ENGINE ONLINE / MODEL / INPUT status strip.
- Removed the static CONFIDENCE 35% / OVERLAP 30% mini-stat row from the single-scan console.
- Removed the equivalent batch status strip and replaced both with compact workflow guidance.
- Kept the actual Detection Engine controls and inference configuration functional in the sidebar and console.
- Simplified the authenticated header to reduce duplicated technical status information.

## 03 Sep 2026 — Batch UI refinement

- Reworked the Batch Results area into a compact result center with clearer hierarchy.
- Replaced oversized blank `Details` bars with light-blue file preview accordions showing filename, type, detections, confidence and status at a glance.
- Added a cleaner annotated-output area inside each expandable preview.
- Tightened batch controls, gain preset chips, action/status row and report download treatment.
- Applied a light-blue visual treatment to Dash dropdown controls and menus for consistency.
- Kept the dark Command Center visual language while using brighter blue interaction states for scan-level controls.
