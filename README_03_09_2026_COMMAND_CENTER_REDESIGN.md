# GPR FieldLab — 03 Sep 2026 Command Center Redesign

The authenticated/post-login workspace has been reorganized as a field-operations command center.

## Structure
1. Persistent UAV/GPR drone survey photograph as the post-login background.
2. Dark operational rail for workspace navigation and AI-engine controls.
3. Sticky command header for engine/session state.
4. Workspace switcher for Scan Lab, Batch Lab, Scan History and Methods.
5. Scan Lab hero with survey status.
6. Two-column acquisition + inference console.
7. Dedicated full-width detection review area for annotated output, metrics, cards, table and JSON.
8. Export controls remain available.
9. Processing-flow strip communicates Acquire → Condition → Detect → Review.

## Functional preservation
Existing Dash callback IDs were intentionally retained. Upload, SEG-Y preprocessing, Roboflow inference, tiling, multiscale inference, confidence/overlap controls, display preferences, batch processing, history, methods, API-key configuration and exports are not removed.
