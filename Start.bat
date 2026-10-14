@echo off
rem ============================================================
rem  Comic Translator - double-click to install (first time) and run
rem ============================================================
setlocal
cd /d "%~dp0"
title Comic Translator

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo  Python was not found.
  echo  1. Download Python 3.11 or newer from https://www.python.org/downloads/
  echo  2. In the installer TICK "Add python.exe to PATH"
  echo  3. Run this file again.
  echo.
  start "" https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating private Python environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not create the environment.
    pause
    exit /b 1
  )
)

if not exist ".venv\.installed" (
  echo.
  echo Installing required packages - first run only, about 2-5 minutes...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Installation failed. Check your internet connection and run this file again.
    pause
    exit /b 1
  )
  echo installed> ".venv\.installed"
)

".venv\Scripts\python.exe" run.py %*
if errorlevel 1 pause
endlocal
