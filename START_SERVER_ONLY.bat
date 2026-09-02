@echo off
title ISRO BAS-HAR — Server (Standby Mode)
cd /d "%~dp0"

echo ===============================================================================
echo                ISRO BAS-HAR: OPERATOR DASHBOARD SERVER
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
echo [INFO] Initializing server in STANDBY mode (use UI to start Webcam/Demo)...
echo [INFO] Dashboard URL: http://127.0.0.1:8000
echo.
echo Press Ctrl+C in this terminal window to stop the server.
echo ===============================================================================
echo.

.\.venv\Scripts\python.exe -m src.ui.server --host 127.0.0.1 --port 8000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server terminated with error code %ERRORLEVEL%.
    echo.
)

pause
