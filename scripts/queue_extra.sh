#!/usr/bin/env bash
# Wait for the core run to finish, then start the extra experiments (avoids GPU model swapping).
cd "$(dirname "$0")/.."
until grep -q "CORE EXPERIMENTS FINISHED" logs/run_all_console.log 2>/dev/null; do sleep 30; done
bash scripts/run_extra.sh > logs/run_extra_console.log 2>&1
