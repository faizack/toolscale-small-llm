@echo off
title Extra experiments - agent_survey
"C:\Program Files\Git\bin\bash.exe" -c "cd /c/Users/faiza/Videos/parul/agent_survey && bash scripts/run_extra.sh > logs/run_extra_console.log 2>&1; echo exit=$? >> logs/run_everything_status.log"
