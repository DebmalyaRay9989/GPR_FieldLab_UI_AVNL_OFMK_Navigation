# GPR Dash Application — Streamlit-Compatible Improved Build

This package is an improved Plotly Dash version of the supplied AVNL-OFMK GPR buried-object detection Streamlit application.

It now uses the same core engine details as the Streamlit app:
- Roboflow model: `rawgprburieobjectdetection/1`
- YOLOv26 Nano object detection
- SEG-Y → dewow → bandpass → background removal → gain → 640×640 JPEG
- 100–900 MHz bandpass, order 4
- linear 30 dB gain
- tiled inference: 640 px tiles with 128 px overlap
- multiscale inference and square padding
- landmine / mine / IED / threat / metal / pipe / cable / utility / rock / root / void / clutter classes

## Fixes and improvements

### Pages and navigation
- The complete Dash component tree is mounted from the first request.
- Authentication is a non-destructive overlay instead of replacing the root layout.
- Single Scan, Batch Analysis, Scan History and Guide pages remain mounted and navigable.
- Session history and authentication state are preserved without destroying the page tree.

### Roboflow configuration
The Dash application now follows the credential conventions of the supplied Streamlit app without hard-coding the API key into the Dash source.

Key lookup order:
1. temporary administrator session key;
2. `ROBOFLOW_API_KEY` environment variable;
3. `.streamlit/secrets.toml`;
4. `secrets.toml` beside the Dash application.

Example `.streamlit/secrets.toml`:

```toml
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_KEY"
```

Windows Command Prompt:

```bat
set ROBOFLOW_API_KEY=YOUR_ROBOFLOW_KEY
python gpr_dash_app.py
```

PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_KEY"
python gpr_dash_app.py
```

After administrator login, the sidebar also provides a temporary session-key field. This is useful when the Dash application is being tested independently of the Streamlit deployment.

### GPR preprocessing
The Dash preprocessing follows the supplied Streamlit defaults:
- Dewow: enabled, window 39
- Bandpass: enabled, 100–900 MHz, order 4
- Background removal: mean
- Trace normalization: available
- Gain: linear, 30 dB
- AGC support retained
- Output: 640×640 JPEG, quality 95

### Inference
- Adaptive upscaling for small B-scans
- Multi-scale inference
- Optional 640 px tiled inference with 128 px overlap
- Reflective square padding
- NMS merging across tiles/scales
- Configurable confidence and overlap thresholds
- Better HTTP, timeout and malformed-response diagnostics

### Results
- Annotated image download
- Single-scan CSV report
- Consolidated batch CSV report
- Detection cards
- Detection table
- Optional raw Roboflow JSON
- Session scan history
- Batch progress and per-file errors

## Installation

```bat
python -m pip install -r requirements_dash.txt
```

## Run

Double-click `RUN_GPR_DASH.bat`, or:

```bat
python gpr_dash_app.py
```

Open:

```text
http://127.0.0.1:8050
```

## Diagnostics

Open:

```text
http://127.0.0.1:8050/health
```

The health endpoint reports the server state, model ID and whether a Roboflow credential is available. It never returns the credential itself.

## Troubleshooting the "Roboflow API key not configured" error

If the Dash app reports that the key is missing:

1. Create `.streamlit/secrets.toml` beside `gpr_dash_app.py` and add `ROBOFLOW_API_KEY`.
2. Or set the `ROBOFLOW_API_KEY` environment variable before starting Dash.
3. Or log in as `GPRAdmin` and use the temporary API-key field in the sidebar.
4. Refresh `/health` and confirm the API key status is `ENVIRONMENT / SECRETS` or `SESSION`.
5. Run one Single Scan inference.

Do not commit a real API key into source control.

## Notes

The application requires a valid Roboflow credential and access to model `rawgprburieobjectdetection/1` for cloud inference. SEG-Y preprocessing requires the packages listed in `requirements_dash.txt`.

## 02 Sep 2026 bandpass stability fix

If a SEG-Y file reports a sampling interval whose resulting Nyquist frequency is lower than the user-selected bandpass range, the preprocessing pipeline no longer aborts. The app records a clear warning and continues with the remaining preprocessing stages. This specifically handles files such as `2026-04-24-11-27-03-gpr_017.sgy`, where the reported `dt` makes a 100–900 MHz filter mathematically impossible. The source SEG-Y timing metadata should still be corrected when physically calibrated frequency filtering is required.

## 03 Sep 2026 AVANI logo integration

The login screen and authenticated sidebar now use the supplied **AVANI — Armoured Vehicles and High-Explosives** logo from `assets/avani_logo.png`. The previous text/icon brand mark has been removed from those two primary brand locations, while the AVNL-OFMK application naming remains in the workspace and authentication text where it is useful for product identification.

The supplied logo asset is packaged locally so the application does not depend on an external image host. A compact mobile fallback keeps the AVANI mark visible when the desktop hero content is hidden on narrow screens.
