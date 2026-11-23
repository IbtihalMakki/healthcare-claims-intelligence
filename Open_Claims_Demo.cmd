@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Copy this file into healthcare_claims_code, beside the existing .venv folder.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 claims_demo.py --serve
if errorlevel 1 pause
