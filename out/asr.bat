@echo off
setlocal DisableDelayedExpansion
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0.."
set "BIN=%LZASR_APP_HOME%"
if not defined BIN set "BIN=%~d0\APP\LzASR"
set "PY=%BIN%\.venv-qwen\Scripts\python.exe"
if exist "%PY%" goto :VENV
where py.exe >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python is unavailable. Run "%~dp0install.bat" first.
  exit /b 2
)
py.exe -3.12 -B "%ROOT%\mod\asr\cli.py" %*
exit /b %ERRORLEVEL%
:VENV
"%PY%" -B "%ROOT%\mod\asr\cli.py" %*
exit /b %ERRORLEVEL%
