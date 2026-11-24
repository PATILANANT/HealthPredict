@echo off
setlocal
title HealthPredict
cd /d "%~dp0"

rem Portable build: bundled Python runtime with every package pre-installed
if exist "%~dp0python\python.exe" (
  set "PYTHONNOUSERSITE=1"
  set "PYTHONHOME="
  set "PYTHONPATH="
  "%~dp0python\python.exe" -s start.py %*
  echo.
  pause
  exit /b
)

rem Find Python 3.12 / 3.11 / 3.10 (the Windows "py" launcher first, then python on PATH)
set "PYCMD="
for %%V in (3.12 3.11 3.10) do (
  if not defined PYCMD (
    py -%%V -c "import sys" >nul 2>&1 && set "PYCMD=py -%%V"
  )
)
if not defined PYCMD (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
  echo.
  echo [!] Python 3.10, 3.11 or 3.12 was not found.
  echo     Install Python 3.12 from https://www.python.org/downloads/release/python-3128/
  echo     and tick "Add python.exe to PATH" during installation, then run start.bat again.
  echo.
  pause
  exit /b 1
)

%PYCMD% start.py %*
echo.
pause
