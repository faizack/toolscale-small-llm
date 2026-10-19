#!/usr/bin/env bash
# Runs every experiment in order. Resumable: just run it again after an interruption.
# Small "learning" scale: ~3-4 h on an RTX 3050 (4 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
$PY scripts/env_snapshot.py
$PY scripts/retrieval_recall.py                              # RQ3a  (embeddings only)
$PY scripts/run_experiment.py configs/rq1_scaling.yaml       # RQ1/RQ2
$PY scripts/run_experiment.py configs/rq3_retrieval.yaml     # RQ3
echo "CORE EXPERIMENTS FINISHED"
