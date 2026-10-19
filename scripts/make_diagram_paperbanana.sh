#!/usr/bin/env bash
# Generates paper/figures/fig_overview.png with PaperBanana (needs GOOGLE_API_KEY or OPENROUTER_API_KEY).
# PaperBanana is cloned at ../tools/PaperBanana (commit 491fa29). Costs API credits; we use 4 candidates.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PB="$HERE/../tools/PaperBanana"
OUT="$HERE/paper/figures/paperbanana"
cd "$PB"
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
"$PY" skill/run.py \
  --content-file "$OUT/method_overview.txt" \
  --caption "$(cat "$OUT/caption.txt")" \
  --task diagram --aspect-ratio 21:9 \
  --num-candidates 4 --max-critic-rounds 2 \
  --output "$OUT/overview.png" | tee "$OUT/run_$(date +%Y%m%d-%H%M%S).log"
echo "Pick the best candidate, copy it to paper/figures/fig_overview.png, and log the choice in notes/lab_notebook.md"
