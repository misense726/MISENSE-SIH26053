@echo off
echo ========================================================
echo Starting MI Sense Backend (FastAPI + WebSocket Stream)
echo Problem Statement: SIH 26053
echo ========================================================

cd /d "%~dp0\.."
set PYTHONPATH=%CD%
call backend\venv\Scripts\activate
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
pause
