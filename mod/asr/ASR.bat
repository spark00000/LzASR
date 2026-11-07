@echo off
setlocal DisableDelayedExpansion

rem Local standalone ASR: Whisper / Qwen3-ASR-1.7B / Qwen3-ASR-0.6B.
rem Drag video/audio files onto ASR.bat and choose an engine.
rem Or run: ASR.bat --engine qwen17 "path-to-video.mp4"
rem ASR output stays BESIDE the input media; existing subtitles are preserved.
rem Model-specific Qwen output names allow side-by-side ASR comparison.

set "APP=%~dp0"
set "BIN=%LZASR_APP_HOME%"
if not defined BIN set "BIN=%~d0\APP\LzASR"
if not "%BIN:~-1%"=="\" set "BIN=%BIN%\"
set "WHISPER_ENGINE=%BIN%engine\Faster-Whisper-XXL\faster-whisper-xxl.exe"
set "WHISPER_MODELS=%BIN%models"
set "QWEN_PY=%BIN%.venv-qwen\Scripts\python.exe"
set "QWEN_SCRIPT=%APP%qwen_asr_to_srt.py"
set "HF_HOME=%BIN%models\.hf"
set "HF_HUB_CACHE=%BIN%models\.hf\hub"
set "HF_XET_CACHE=%BIN%models\.hf\xet"
set "TORCH_HOME=%BIN%models\.torch"
set "UV_CACHE_DIR=%BIN%.uv-cache"

if /I "%~1"=="--help" goto :USAGE
set "ASR_ENGINE=%WHISPER_ASR_ENGINE%"
if /I "%~1"=="--engine" (
    if "%~2"=="" goto :USAGE
    set "ASR_ENGINE=%~2"
    shift
    shift
)
if "%~1"=="" goto :USAGE

if not defined ASR_ENGINE set "ASR_ENGINE=whisper"
if /I "%ASR_ENGINE%"=="1" set "ASR_ENGINE=whisper"
if /I "%ASR_ENGINE%"=="2" set "ASR_ENGINE=qwen17"
if /I "%ASR_ENGINE%"=="3" set "ASR_ENGINE=qwen06"
if /I "%ASR_ENGINE%"=="qwen" set "ASR_ENGINE=qwen17"

if /I "%ASR_ENGINE%"=="whisper" goto :CHECK_WHISPER
if /I "%ASR_ENGINE%"=="qwen17" goto :CHECK_QWEN
if /I "%ASR_ENGINE%"=="qwen06" goto :CHECK_QWEN
echo [ERROR] Invalid engine: "%ASR_ENGINE%"
exit /b 2

:CHECK_WHISPER
if not exist "%WHISPER_ENGINE%" goto :NO_WHISPER
if not exist "%WHISPER_MODELS%\faster-whisper-large-v3-turbo\model.bin" goto :NO_WHISPER_MODEL
goto :BEGIN

:CHECK_QWEN
if not exist "%QWEN_PY%" goto :NO_QWEN
if not exist "%QWEN_SCRIPT%" goto :NO_QWEN
goto :BEGIN

:BEGIN
echo [ENGINE] %ASR_ENGINE%
set "ERRORS=0"

:NEXT
if "%~1"=="" goto :DONE
if not exist "%~f1" goto :INPUT_MISSING
set "INPUT=%~f1"
if exist "%INPUT%\NUL" goto :DIRECTORY

if /I "%ASR_ENGINE%"=="whisper" for %%F in ("%INPUT%") do set "OUTPUT_SRT=%%~dpnF_whisper-large-v3-turbo.srt"
if /I "%ASR_ENGINE%"=="qwen17" for %%F in ("%INPUT%") do set "OUTPUT_SRT=%%~dpnF_qwen3-asr-1.7b.srt"
if /I "%ASR_ENGINE%"=="qwen06" for %%F in ("%INPUT%") do set "OUTPUT_SRT=%%~dpnF_qwen3-asr-0.6b.srt"
if exist "%OUTPUT_SRT%" goto :EXISTS
rem The Whisper standalone tool always writes the generic <stem>.srt first.
rem Refuse to touch a preexisting generic subtitle; never overwrite user data.
if /I "%ASR_ENGINE%"=="whisper" for %%F in ("%INPUT%") do if exist "%%~dpnF.srt" goto :LEGACY_COLLISION

echo.
echo [START] "%INPUT%"
if /I "%ASR_ENGINE%"=="whisper" (
    "%WHISPER_ENGINE%" "%INPUT%" --model large-v3-turbo --model_dir "%WHISPER_MODELS%" --device cuda --compute_type float16 --beam_size 5 --vad_filter True --word_timestamps True --task transcribe --output_format srt --output_dir source --beep_off
) else (
    if /I "%ASR_ENGINE%"=="qwen17" (
        "%QWEN_PY%" "%QWEN_SCRIPT%" "%INPUT%" --model "1.7B" --output "%OUTPUT_SRT%"
    ) else (
        "%QWEN_PY%" "%QWEN_SCRIPT%" "%INPUT%" --model "0.6B" --output "%OUTPUT_SRT%"
    )
)

if errorlevel 1 goto :FAIL
if /I "%ASR_ENGINE%"=="whisper" (
    for %%F in ("%INPUT%") do ren "%%~dpnF.srt" "%%~nF_whisper-large-v3-turbo.srt"
    if errorlevel 1 goto :FAIL
)
if not exist "%OUTPUT_SRT%" goto :NO_OUTPUT
echo [OK] "%OUTPUT_SRT%"
shift
goto :NEXT

:LEGACY_COLLISION
echo [ERROR] Generic subtitle already exists; refusing to overwrite it: "%INPUT%"
exit /b 3

:EXISTS
echo [SKIP] Existing SRT preserved: "%OUTPUT_SRT%"
shift
goto :NEXT

:INPUT_MISSING
echo [ERROR] Input not found: "%~1"
set "ERRORS=1"
shift
goto :NEXT

:DIRECTORY
echo [ERROR] Expected a media file: "%INPUT%"
set "ERRORS=1"
shift
goto :NEXT

:FAIL
echo [ERROR] ASR failed: "%INPUT%"
set "ERRORS=1"
shift
goto :NEXT

:NO_OUTPUT
echo [ERROR] No subtitle file: "%OUTPUT_SRT%"
set "ERRORS=1"
shift
goto :NEXT

:DONE
echo.
if "%ERRORS%"=="0" echo [DONE] All files processed or skipped.
if "%ERRORS%"=="0" exit /b 0
echo [DONE] One or more files failed.
exit /b 1

:NO_WHISPER
echo [ERROR] Faster-Whisper-XXL executable is missing: "%WHISPER_ENGINE%"
exit /b 2

:NO_WHISPER_MODEL
echo [ERROR] Whisper model.bin is missing.
exit /b 2

:NO_QWEN
echo [ERROR] Qwen isolated environment or script is missing.
exit /b 2

:USAGE
echo Usage: "%~f0" [--engine whisper^|qwen17^|qwen06] "video1" ["video2"...]
echo Drag files onto the BAT to select the engine.
echo Whisper:  video_whisper-large-v3-turbo.srt
echo Qwen17:   video_qwen3-asr-1.7b.srt
echo Qwen06:   video_qwen3-asr-0.6b.srt
echo Results are created next to the input videos without overwriting.
exit /b 2
