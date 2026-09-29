@echo off
echo ========================================================
echo Launching Complete MI Sense Prototype (Backend + Frontend)
echo Problem Statement: SIH 26053
echo ========================================================

start "MI Sense Backend Server" cmd /k "%~dp0start_backend.bat"
timeout /t 2 /nobreak >nul
start "MI Sense Frontend UI" cmd /k "%~dp0start_frontend.bat"

echo.
echo Both servers are starting up:
echo Backend API & Stream: http://localhost:8000
echo Frontend Dashboard:   http://localhost:5173
echo.
