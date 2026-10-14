@echo off
chcp 65001 >nul
title Comic Translator - Installer
echo ==== Comic Translator: download and install ====
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((Get-Content -Raw -Encoding UTF8 '%~dp0packaging\install.ps1'))"
echo.
pause
