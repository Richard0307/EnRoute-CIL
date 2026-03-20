@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0upload_github.ps1" %*
endlocal
