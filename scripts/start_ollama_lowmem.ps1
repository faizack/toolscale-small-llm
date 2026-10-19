# Starts Ollama with low-memory settings (one model, one request slot). Used for the experiments.
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve 2>&1 | Tee-Object -FilePath "$PSScriptRoot\..\logs\ollama_serve.log"
