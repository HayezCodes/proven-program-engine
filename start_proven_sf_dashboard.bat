@echo off
echo Starting Proven S/F Dashboard...
cd /d "%~dp0"
py -m streamlit run src/dashboard/app.py
pause
