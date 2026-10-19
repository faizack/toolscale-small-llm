#!/usr/bin/env bash
# Experiments on native datasets (run after scripts/run_all.sh). Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
$PY scripts/run_native.py configs/rq4_native.yaml          # RQ4: BFCL live_multiple / live_irrelevance
$PY scripts/run_multiturn.py configs/rq5_multiturn.yaml    # RQ5: agentic multi-turn (official checker)
$PY scripts/toolret_retrieval.py                           # RQ3b: ToolRet 44k-tool retrieval
echo "EXTRA EXPERIMENTS FINISHED"
