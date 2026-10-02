@echo off
setlocal
cd /d "%~dp0"
if not exist "profiles\local-sm75.json" (
  echo Generate profiles\local-sm75.json using tools\sm75\configure.py first.
  echo See README.md for the release engine and model paths.
  pause
  exit /b 1
)
set "STRATA_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "STRATA_PYTHON=%CD%\.venv\Scripts\python.exe"
"%STRATA_PYTHON%" "tools\sm75\start.py" --config "profiles\local-sm75.json"
if errorlevel 1 pause
