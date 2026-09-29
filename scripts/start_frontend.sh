#!/usr/bin/env bash
echo "========================================================"
echo "Starting MI Sense Frontend (React + Vite + Three.js)"
echo "Problem Statement: SIH 26053"
echo "========================================================"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/../frontend"
npm run dev
