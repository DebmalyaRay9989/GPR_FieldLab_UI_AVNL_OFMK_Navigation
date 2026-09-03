@echo off
cd /d "%~dp0"
python -m pip install -r requirements_dash.txt
python gpr_dash_app.py
pause
