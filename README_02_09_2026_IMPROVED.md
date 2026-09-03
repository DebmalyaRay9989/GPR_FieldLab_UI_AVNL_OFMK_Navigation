# GPR FieldLab UI — Improved Build (02 September 2026)

This package consolidates the latest GPR FieldLab interface with the supplied
24 August 2026 Dash application improvements.

## Included improvements
- Professional FieldLab login experience with drone-field background.
- Modern responsive application styling applied directly through `assets/style.css`.
- Dedicated FieldLab launcher.
- Retained Single Scan, Batch Analysis, Scan History and Guide workflows.
- Retained Roboflow inference, tiled/multiscale inference, NMS, CSV export and
  annotated-result workflows.
- More defensive SEG-Y input validation and cleanup.
- Safer band-pass validation with clearer errors for invalid frequency ranges or
  scans too short for zero-phase filtering.
- Cleaned distribution structure without cached Python artifacts.
- Added `.gitignore` to prevent credentials and local runtime files from being
  accidentally included in future source-control copies.

## Run
1. Install Python 3.10+.
2. Configure `ROBOFLOW_API_KEY` in the environment, `.streamlit/secrets.toml`,
   or the temporary administrator session-key field.
3. Double-click `RUN_GPR_FIELDLAB.bat`, or run:

   python -m pip install -r requirements_dash.txt
   python gpr_dash_app.py

Then open `http://127.0.0.1:8050`.

The Roboflow API key is intentionally not included in this package.

## Post-login drone background refinement

After successful authentication, the login hero is fully hidden and the
authenticated FieldLab workspace uses the supplied `gpr_drone_background.png`
as a restrained full-screen field-survey backdrop. A light glass/gradient
overlay keeps charts, tables, controls and detection results readable while
allowing the drone survey image to remain visibly present.
