# GPR FieldLab — Post-Login UI Redesign

The authenticated application shell has been restructured with a clearer operational hierarchy:

- Premium fixed operations sidebar with branded identity, workspace navigation and compact AI controls.
- Cleaner sticky header with live engine status, workspace identity and session indicator.
- Dedicated workspace selector bar separating navigation from analytical content.
- More consistent card system, spacing, shadows, typography and responsive behavior.
- Background imagery is now extremely subtle after login so analytical data remains the visual focus.
- Display preferences are collapsed by default to reduce sidebar clutter.
- Existing Dash component IDs and processing callbacks are retained, so preprocessing, inference, batch analysis, history and export logic remain compatible.

Run with `RUN_GPR_FIELDLAB.bat` or `python gpr_dash_app.py` after installing `requirements_dash.txt`.
