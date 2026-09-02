@echo off
title ISRO BAS-HAR — Operator Mission Dashboard
cd /d "%~dp0"

echo ===============================================================================
echo                ISRO BAS-HAR: OPERATOR MISSION DASHBOARD
echo        AI/ML Activity Recognition and Protocol Monitoring Prototype
echo ===============================================================================
echo.

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Python executable not found in virtual environment:
    echo         %CD%\.venv\Scripts\python.exe
    echo.
    echo Please verify that the project virtual environment is set up.
    echo.
    pause
    exit /b 1
)

echo [INFO] Project Root: %CD%
echo [INFO] Python Interpreter: .\.venv\Scripts\python.exe
echo [INFO] Initializing unified perception, temporal HAR, and protocol FSM...
echo [INFO] Auto-starting demonstration video: data/raw/videos/SES_001_NOMINAL.mp4
echo [INFO] Dashboard URL: http://127.0.0.1:8000
echo.
echo Press Ctrl+C in this terminal window to stop the server.
echo ===============================================================================
echo.

.\.venv\Scripts\python.exe -m src.ui.server --host 127.0.0.1 --port 8000 --auto_start --video data/raw/videos/SES_001_NOMINAL.mp4

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server terminated with error code %ERRORLEVEL%.
    echo.
)

pause
