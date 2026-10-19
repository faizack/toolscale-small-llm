#!/usr/bin/env bash
# Core + extra experiments, then one retry pass for calls that failed (timeouts, resets).
# Run in its own terminal window.
cd "$(dirname "$0")/.."
echo "started $(date)"
bash scripts/run_all.sh   > logs/run_all_console.log   2>&1
bash scripts/run_extra.sh > logs/run_extra_console.log 2>&1
echo "retry pass $(date)"
bash scripts/run_all.sh   > logs/run_all_retry.log   2>&1
bash scripts/run_extra.sh > logs/run_extra_retry.log 2>&1
echo "exit=$? at $(date)" | tee -a logs/run_everything_status.log
read -p "Press Enter to close"
