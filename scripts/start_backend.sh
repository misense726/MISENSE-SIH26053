#!/usr/bin/env bash
echo "========================================================"
echo "Starting MI Sense Backend (FastAPI + WebSocket Stream)"
echo "Problem Statement: SIH 26053"
echo "========================================================"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/.."
export PYTHONPATH="$PWD"
source backend/venv/Scripts/activate 2>/dev/null || source backend/venv/bin/activate 2>/dev/null
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
