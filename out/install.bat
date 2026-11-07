@echo off
setlocal DisableDelayedExpansion
set "SCRIPT=%~dp0..\mod\asr\install.ps1"
if "%~1"=="" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" -Install -Engine all
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
)
exit /b %ERRORLEVEL%
