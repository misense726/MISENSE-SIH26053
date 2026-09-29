#!/usr/bin/env bash
echo "========================================================"
echo "Launching Complete MI Sense Prototype (Backend + Frontend)"
echo "Problem Statement: SIH 26053"
echo "========================================================"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Start backend in background
"$DIR/start_backend.sh" &
BACKEND_PID=$!

sleep 2

# Start frontend
"$DIR/start_frontend.sh" &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID, Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to terminate both servers."

trap "kill $BACKEND_PID $FRONTEND_PID" SIGINT SIGTERM
wait
