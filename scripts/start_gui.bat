@echo off
setlocal
chcp 65001 >nul

REM ============================================================
REM  Auto Screen Reader - GUI (interactive reading board)
REM
REM  Requires the GPT-SoVITS TTS API to be running:
REM    runtime\python.exe -s api_v2.py -a 127.0.0.1 -p 9880 ^
REM        -c <dsh>\data\tts_infer_voice01.yaml
REM
REM  NOTE: pure ASCII on purpose. Chinese text in a .bat can break
REM  cmd parsing (some GBK trailing bytes collide with & | > etc).
REM ============================================================

set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] venv python not found: %PY%
  pause
  exit /b 1
)

set "PYTHONPATH=%ROOT%\src"
set "PYTHONIOENCODING=utf-8"
cd /d "%ROOT%"

echo Starting screen reader GUI ...
"%PY%" -m screenreader gui
if errorlevel 1 pause
