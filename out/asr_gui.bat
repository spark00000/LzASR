@echo off
setlocal DisableDelayedExpansion
set "GUI=%~dp0..\mod\asr\gui.py"
set "BIN=%LZASR_APP_HOME%"
if not defined BIN set "BIN=%~d0\APP\LzASR"
set "PY=%BIN%\.venv-qwen\Scripts\pythonw.exe"
if exist "%PY%" (
    start "" "%PY%" -B "%GUI%" %*
    exit /b %ERRORLEVEL%
)
where pyw.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] GUI Python is unavailable. Run "%~dp0install.bat" first.
    exit /b 2
)
start "" pyw.exe -3.12 -B "%GUI%" %*
exit /b %ERRORLEVEL%
