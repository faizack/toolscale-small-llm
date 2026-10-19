@echo off
rem Starts Ollama with low-memory settings (one model, one request slot) for the experiments.
title Ollama (low-memory) - agent_survey
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=1
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
