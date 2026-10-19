@echo off
setlocal
chcp 65001 >nul
REM ============================================================
REM  SubFix Audio Labeling WebUI launcher
REM
REM  Sets GRADIO_ALLOWED_PATHS automatically so the sliced wavs can
REM  be served. Without it Gradio returns 403 for every audio file
REM  and the labeler errors out when you click "Submit Text".
REM
REM  NOTE: this file is intentionally pure ASCII. Chinese text in a
REM  .bat can break cmd parsing, because some GBK trailing bytes
REM  collide with cmd special characters (& | > etc).
REM ============================================================

set "GSV=D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604"
set "TOOLS=%~dp0..\tools"
set "LIST=%~dp0..\data\voice01_train_clean.list"

if not exist "%GSV%\runtime\python.exe" (
  echo [ERROR] Integrated package python not found:
  echo         %GSV%\runtime\python.exe
  pause
  exit /b 1
)
if not exist "%LIST%" (
  echo [ERROR] List file not found: %LIST%
  echo         For another list, run manually:
  echo         "%GSV%\runtime\python.exe" -u -s "%TOOLS%\subfix_launch.py" --list your.list
  pause
  exit /b 1
)

echo.
echo   Starting SubFix audio labeling WebUI ...
echo.
echo   List file : %LIST%
echo   Open      : http://127.0.0.1:9871
echo.
echo   Close this window to stop the service.
echo.

"%GSV%\runtime\python.exe" -u -s "%TOOLS%\subfix_launch.py" --list "%LIST%" --port 9871

echo.
echo   Service stopped.
pause
